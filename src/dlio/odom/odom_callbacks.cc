#include "dlio/odom/odom.h"
#include "dlio/odom/utils.h"
#include "dlio/odom/kfdb_io.h"
#if DLIO_HAS_CUDA
#include "dlio/cuda/preprocess_cuda.cuh"
#include "dlio/cuda/common.cuh"
#endif

#include <queue>
#include <filesystem>

#include "rclcpp/qos.hpp"

void dlio::OdomNode::resetGtsamNavState()
{
  // Re-seed the GTSAM preintegration nav state from the current state so the
  // IMU thread predicts forward from the (relocalized/corrected) pose instead
  // of a stale one. Without this, after relocalization the IMU dead-reckoning
  // keeps integrating from the old pose and "flies" the robot away.
  if (this->imu_preintegration_mode_ != "gtsam" || !this->gtsam_imu_initialized_ || !this->imu_preintegration_)
    return;
  std::lock_guard<std::mutex> state_lock(this->state_mtx_);
  this->gtsam_nav_state_ = gtsam::NavState(
      gtsam::Pose3(gtsam::Rot3(this->state.q.cast<double>()), this->state.p.cast<double>()),
      this->state.v.lin.w.cast<double>());
  // Bias = zero because imu_meas is already bias-corrected in callbackImu()
  this->gtsam_bias_ = gtsam::imuBias::ConstantBias();
  this->imu_preintegration_->resetIntegrationAndSetBias(this->gtsam_bias_);
}

void dlio::OdomNode::getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr &pc)
{

  pcl::PointCloud<PointType>::Ptr original_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*pc, *original_scan_);

  // Handle "time_stamp" field that PCL can't auto-map (e.g. Hesai Pandar XT32)
  // Manually copy uint32 time_stamp → double timestamp per point
  for (const auto &field : pc->fields)
  {
    if (field.name == "time_stamp" && field.datatype == sensor_msgs::msg::PointField::UINT32)
    {
      uint32_t point_step = pc->point_step;
      uint32_t offset = field.offset;
      const uint8_t *raw = pc->data.data();
      size_t n_pts = std::min<size_t>(original_scan_->points.size(),
                                      pc->width * pc->height);
      for (size_t i = 0; i < n_pts; i++)
      {
        uint32_t ts_ns;
        memcpy(&ts_ns, raw + i * point_step + offset, sizeof(uint32_t));
        original_scan_->points[i].timestamp = static_cast<double>(ts_ns) * 1e-9;
      }
      break;
    }
  }

  // Remove NaNs
  std::vector<int> idx;
  original_scan_->is_dense = false;
  pcl::removeNaNFromPointCloud(*original_scan_, *original_scan_, idx);

  // Crop Box Filter
  this->crop.setInputCloud(original_scan_);
  this->crop.filter(*original_scan_);

  // automatically detect sensor type
  this->sensor = dlio::SensorType::UNKNOWN;
  for (auto &field : pc->fields)
  {
    if (field.name == "t")
    {
      this->sensor = dlio::SensorType::OUSTER;
      break;
    }
    else if (field.name == "time")
    {
      this->sensor = dlio::SensorType::VELODYNE;
      break;
    }
    else if (field.name == "timestamp")
    {
      double ts0 = original_scan_->points[0].timestamp;
      if (ts0 > 1e14)
      {
        this->sensor = dlio::SensorType::LIVOX; // nanoseconds since epoch
      }
      else if (ts0 > 1e6)
      {
        this->sensor = dlio::SensorType::HESAI; // absolute seconds since epoch
      }
      else
      {
        this->sensor = dlio::SensorType::ROBOSENSE; // relative offset in seconds from scan start
      }
      break;
    }
    else if (field.name == "time_stamp")
    {
      // time_stamp field (e.g. Hesai Pandar XT32): uint32 nanoseconds from scan start
      // After manual copy above, timestamp = relative offset in seconds
      // Use ROBOSENSE path which adds sweep_ref_time to relative offset
      this->sensor = dlio::SensorType::ROBOSENSE;
      break;
    }
  }

  if (this->sensor == dlio::SensorType::UNKNOWN)
  {
    this->deskew_ = false;
  }

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = original_scan_;
}

void dlio::OdomNode::preprocessPoints()
{

  // Deskew the original dlio-type scan
  if (this->deskew_)
  {

    this->deskewPointcloud();

    if (!this->first_valid_scan)
    {
      return;
    }
  }
  else
  {

    this->scan_stamp = rclcpp::Time(this->scan_header_stamp).seconds();

    // don't process scans until IMU data is present (skip check in pure LiDAR mode)
    if (!this->first_valid_scan)
    {
      if (this->use_imu_)
      {
        if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp)
        {
          return;
        }
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan
      this->T_prev_ = this->T; // init for constant velocity model
    }
    else
    {
      if (this->use_imu_)
      {
        // IMU prior for second scan onwards
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
        frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                                    this->geo.prev_vel.cast<float>(), {this->scan_stamp});

        if (frames.size() > 0)
        {
          this->T_prior = frames.back();
        }
        else
        {
          this->T_prior = this->T;
        }
      }
      else
      {
        // Pure LiDAR: constant velocity model (like KISS-ICP)
        this->T_prior = this->T * this->T_prev_.inverse() * this->T;
      }
    }

    pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud(*this->original_scan, *deskewed_scan_,
                             this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // Voxel Grid Filter
  if (this->vf_use_)
  {
    if (this->gpu_preprocess_)
    {
#if DLIO_HAS_CUDA
      // ── GPU voxel filter path ──
      size_t N = this->deskewed_scan->points.size();
      std::vector<float4> h_in(N);
      for (size_t i = 0; i < N; i++)
      {
        const auto &pt = this->deskewed_scan->points[i];
        h_in[i] = make_float4(pt.x, pt.y, pt.z, pt.intensity);
      }

      dlio::cuda::DeviceBuffer<float4> d_in, d_out;
      d_in.upload(h_in.data(), N);
      d_out.allocate(N);

      size_t n_out = dlio::cuda::voxelGridFilter(
          d_in.data(), N, d_out.data(), nullptr,
          static_cast<float>(this->vf_res_));

      // Download filtered points
      pcl::PointCloud<PointType>::Ptr current_scan_ =
          std::make_shared<pcl::PointCloud<PointType>>();
      current_scan_->points.resize(n_out);
      std::vector<float4> h_out(n_out);
      d_out.download(h_out.data(), n_out);
      for (size_t i = 0; i < n_out; i++)
      {
        current_scan_->points[i].x = h_out[i].x;
        current_scan_->points[i].y = h_out[i].y;
        current_scan_->points[i].z = h_out[i].z;
        current_scan_->points[i].intensity = h_out[i].w;
      }
      current_scan_->width = n_out;
      current_scan_->height = 1;
      current_scan_->is_dense = true;
      this->current_scan = current_scan_;
#endif
    }
    else
    {
      // ── CPU voxel filter path (PCL) ──
      pcl::PointCloud<PointType>::Ptr current_scan_ =
          std::make_shared<pcl::PointCloud<PointType>>(*this->deskewed_scan);
      this->voxel.setInputCloud(current_scan_);
      this->voxel.filter(*current_scan_);
      this->current_scan = current_scan_;
    }
  }
  else
  {
    this->current_scan = this->deskewed_scan;
  }

  // Prefilter (noise removal — after deskew + voxel, before registration)
  if (this->prefilter_.isEnabled())
  {
    this->current_scan = this->prefilter_.filter(this->current_scan);
  }
}

void dlio::OdomNode::deskewPointcloud()
{

  pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>(1, this->original_scan->points.size());
  // deskewed_scan_->points.resize(this->original_scan->points.size());
  // individual point timestamps should be relative to this time
  double sweep_ref_time = rclcpp::Time(this->scan_header_stamp).seconds();

  // sort points by timestamp and build list of timestamps
  std::function<bool(const PointType &, const PointType &)> point_time_cmp;
  std::function<bool(boost::range::index_value<PointType &, long>,
                     boost::range::index_value<PointType &, long>)>
      point_time_neq;
  std::function<double(boost::range::index_value<PointType &, long>)> extract_point_time;

  if (this->sensor == dlio::SensorType::OUSTER)
  {

    point_time_cmp = [](const PointType &p1, const PointType &p2)
    { return p1.t < p2.t; };
    point_time_neq = [](boost::range::index_value<PointType &, long> p1,
                        boost::range::index_value<PointType &, long> p2)
    { return p1.value().t != p2.value().t; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType &, long> pt)
    { return sweep_ref_time + pt.value().t * 1e-9f; };
  }
  else if (this->sensor == dlio::SensorType::VELODYNE)
  {

    point_time_cmp = [](const PointType &p1, const PointType &p2)
    { return p1.time < p2.time; };
    point_time_neq = [](boost::range::index_value<PointType &, long> p1,
                        boost::range::index_value<PointType &, long> p2)
    { return p1.value().time != p2.value().time; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType &, long> pt)
    { return sweep_ref_time + pt.value().time; };
  }
  else if (this->sensor == dlio::SensorType::HESAI)
  {

    point_time_cmp = [](const PointType &p1, const PointType &p2)
    { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType &, long> p1,
                        boost::range::index_value<PointType &, long> p2)
    { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType &, long> pt)
    { return pt.value().timestamp; };
  }
  else if (this->sensor == dlio::SensorType::ROBOSENSE)
  {
    // Robosense: timestamp field is relative offset in seconds from scan start
    point_time_cmp = [](const PointType &p1, const PointType &p2)
    { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType &, long> p1,
                        boost::range::index_value<PointType &, long> p2)
    { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType &, long> pt)
    { return sweep_ref_time + pt.value().timestamp; };
  }
  else if (this->sensor == dlio::SensorType::LIVOX)
  {
    point_time_cmp = [](const PointType &p1, const PointType &p2)
    { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType &, long> p1,
                        boost::range::index_value<PointType &, long> p2)
    { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType &, long> pt)
    { return pt.value().timestamp * 1e-9f; };
  }

  // copy points into deskewed_scan_ in order of timestamp
  std::partial_sort_copy(this->original_scan->points.begin(), this->original_scan->points.end(),
                         deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

  // filter unique timestamps
  auto points_unique_timestamps = deskewed_scan_->points | boost::adaptors::indexed() | boost::adaptors::adjacent_filtered(point_time_neq);

  // extract timestamps from points and put them in their own list
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;

  // compute offset between sweep reference time and first point timestamp
  double offset = 0.0;
  if (this->time_offset_)
  {
    offset = sweep_ref_time - extract_point_time(*points_unique_timestamps.begin());
  }

  // build list of unique timestamps and indices of first point with each timestamp
  for (auto it = points_unique_timestamps.begin(); it != points_unique_timestamps.end(); it++)
  {
    timestamps.push_back(extract_point_time(*it) + offset);
    unique_time_indices.push_back(it->index());
  }
  unique_time_indices.push_back(deskewed_scan_->points.size());

  int median_pt_index = timestamps.size() / 2;
  this->scan_stamp = timestamps[median_pt_index]; // set this->scan_stamp to the timestamp of the median point

  if (this->deep_debug_ && !timestamps.empty())
  {
    RCLCPP_INFO(this->get_logger(),
                "[DEEP] deskew: sensor=%d, sweep_ref=%.6f, first_pt_time=%.6f, scan_stamp=%.6f, raw_timestamp[0]=%.9f, imu_front=%.6f",
                (int)this->sensor, sweep_ref_time, timestamps.front(), this->scan_stamp,
                deskewed_scan_->points[0].timestamp,
                this->imu_buffer.empty() ? 0.0 : this->imu_buffer.front().stamp);
  }

  // don't process scans until IMU data is present
  if (!this->first_valid_scan)
  {
    if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp)
    {
      return;
    }

    this->first_valid_scan = true;
    this->T_prior = this->T; // assume no motion for the first scan
    pcl::transformPointCloud(*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return;
  }

  // IMU prior & deskewing for second scan onwards
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
  frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                              this->geo.prev_vel.cast<float>(), timestamps);
  this->deskew_size = frames.size(); // if integration successful, equal to timestamps.size()

  // if there are no frames between the start and end of the sweep
  // that probably means that there's a sync issue
  if (frames.size() != timestamps.size())
  {
    RCLCPP_FATAL(this->get_logger(), "Bad time sync between LiDAR and IMU!");

    this->T_prior = this->T;
    pcl::transformPointCloud(*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
    return;
  }

  // update prior to be the estimated pose at the median time of the scan (corresponds to this->scan_stamp)
  this->T_prior = frames[median_pt_index];

  if (this->gpu_preprocess_)
  {
#if DLIO_HAS_CUDA
    // ── GPU deskewing path ──
    size_t N = deskewed_scan_->points.size();
    int num_frames = static_cast<int>(frames.size());

    // Pre-multiply each frame with baselink2lidar_T and flatten to row-major
    std::vector<float> transforms_flat(num_frames * 16);
    for (int i = 0; i < num_frames; i++)
    {
      Eigen::Matrix4f T = frames[i] * this->extrinsics.baselink2lidar_T;
      // Eigen is column-major → transpose to row-major for GPU
      Eigen::Matrix4f T_row = T.transpose();
      memcpy(&transforms_flat[i * 16], T_row.data(), 16 * sizeof(float));
    }

    // Build per-point frame index array from unique_time_indices
    std::vector<int> frame_indices(N);
    for (int i = 0; i < num_frames; i++)
    {
      int start = unique_time_indices[i];
      int end = unique_time_indices[i + 1];
      for (int k = start; k < end; k++)
        frame_indices[k] = i;
    }

    // Upload points as float4 (x, y, z, intensity)
    std::vector<float4> h_points(N);
    for (size_t i = 0; i < N; i++)
    {
      auto &pt = deskewed_scan_->points[i];
      h_points[i] = make_float4(pt.x, pt.y, pt.z, pt.intensity);
    }

    dlio::cuda::DeviceBuffer<float4> d_points;
    dlio::cuda::DeviceBuffer<float> d_transforms;
    dlio::cuda::DeviceBuffer<int> d_frame_indices;
    d_points.upload(h_points.data(), N);
    d_transforms.upload(transforms_flat.data(), num_frames * 16);
    d_frame_indices.upload(frame_indices.data(), N);

    // Run GPU deskew kernel (in-place)
    dlio::cuda::deskewPointCloud(d_points.data(), d_points.data(), N,
                                 d_transforms.data(), d_frame_indices.data(),
                                 num_frames);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results
    d_points.download(h_points.data(), N);
    for (size_t i = 0; i < N; i++)
    {
      auto &pt = deskewed_scan_->points[i];
      pt.x = h_points[i].x;
      pt.y = h_points[i].y;
      pt.z = h_points[i].z;
      // intensity preserved by kernel
    }
#endif
  }
  else
  {
    // ── CPU deskewing path (OpenMP) ──
#pragma omp parallel for num_threads(this->num_threads_)
    for (int i = 0; i < static_cast<int>(timestamps.size()); i++)
    {
      Eigen::Matrix4f T = frames[i] * this->extrinsics.baselink2lidar_T;

      // transform point to world frame
      for (int k = unique_time_indices[i]; k < unique_time_indices[i + 1]; k++)
      {
        auto &pt = deskewed_scan_->points[k];
        pt.getVector4fMap()[3] = 1.;
        pt.getVector4fMap() = T * pt.getVector4fMap();
      }
    }
  }

  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;
}

void dlio::OdomNode::initializeDLIO()
{

  // Wait for IMU (skip if pure LiDAR mode)
  if (this->use_imu_ && (!this->first_imu_received || !this->imu_calibrated))
  {
    return;
  }

  this->dlio_initialized = true;
  std::cout << std::endl
            << " DLIO initialized!" << std::endl;
}

void dlio::OdomNode::callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc)
{

  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(main_loop_running_mutex);
  this->main_loop_running = true;
  lock.unlock();

  double then = this->now().seconds();

  if (this->first_scan_stamp == 0.)
  {
    this->first_scan_stamp = rclcpp::Time(pc->header.stamp).seconds();
  }

  // DLIO Initialization procedures (IMU calib, gravity align)
  if (!this->dlio_initialized)
  {
    this->initializeDLIO();
  }

  // Set initial pose for prior map (non-relocalize mode)
  if (this->use_prior_map_ && !this->relocalize_ && !this->prior_map_pose_set_ && this->dlio_initialized)
  {
    this->state.p = this->initial_position_;

    // Apply initial yaw on top of gravity-aligned orientation
    Eigen::Quaternionf yaw_q(Eigen::AngleAxisf(this->initial_yaw_, Eigen::Vector3f::UnitZ()));
    this->state.q = yaw_q * this->state.q;

    this->T = Eigen::Matrix4f::Identity();
    this->T.block<3, 3>(0, 0) = this->state.q.toRotationMatrix();
    this->T.block<3, 1>(0, 3) = this->state.p;
    this->T_prior = this->T;
    this->lidarPose.p = this->state.p;
    this->lidarPose.q = this->state.q;
    this->prior_map_pose_set_ = true;

    // Must set main_loop_running = false before buildSubmap to avoid deadlock
    this->main_loop_running = false;

    // Rebuild submap around initial pose
    this->buildSubmap(this->state);
    this->submap_hasChanged = true;
    this->new_submap_is_ready = true;

    RCLCPP_INFO(this->get_logger(), "Prior map initial pose set: [%.1f, %.1f, %.1f] yaw=%.1f deg",
                this->state.p[0], this->state.p[1], this->state.p[2], this->initial_yaw_ * 180.0 / M_PI);
  }

  // Convert incoming scan into DLIO format
  this->getScanFromROS(pc);

  // Preprocess points
  this->preprocessPoints();

  if (!this->first_valid_scan)
  {
    return;
  }

  // Fresh relocalization request (RViz 2D Pose Estimate). Wipe live odometry
  // state so the post-reloc submap can't collide with stale float-keyframes,
  // then arm a guess-only relocalization around the clicked pose. Done here on
  // the odometry thread so we don't race the keyframe/submap structures.
  if (this->request_fresh_reloc_.exchange(false))
  {
    size_t kf_before = 0, kf_after = 0;
    {
      std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
      kf_before = this->keyframes.size();
      int keep = this->num_prior_keyframes_; // keep only the virtual prior-map keyframes
      if (static_cast<int>(this->keyframes.size()) > keep)
      {
        this->keyframes.resize(keep);
        this->keyframe_timestamps.resize(keep);
        this->keyframe_normals.resize(keep);
        this->keyframe_transformations.resize(keep);
      }
      this->num_processed_keyframes = keep;
      kf_after = this->keyframes.size();
    }
    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(),
                  "  [RELOC] fresh-reset: keyframes %zu->%zu (prior=%d), initial_position_=[%.2f,%.2f,%.2f]",
                  kf_before, kf_after, this->num_prior_keyframes_,
                  this->initial_position_[0], this->initial_position_[1], this->initial_position_[2]);
    this->submap_kf_idx_prev.clear();
    this->submap_kf_idx_curr.clear();
    this->voxel_map_last_kf_idx_ = 0;
    this->roi_initialized_ = false; // rebuild prior-map ROI at the relocalized pose

    // Zero velocity / observer so stale motion can't fling the pose.
    this->state.v.lin.w.setZero();
    this->state.v.lin.b.setZero();
    this->state.v.ang.w.setZero();
    this->state.v.ang.b.setZero();
    {
      std::lock_guard<std::mutex> lock(this->geo.mtx);
      this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);
    }
    {
      std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
      this->T_map_odom_ = Eigen::Matrix4f::Identity();
      this->bayes_posterior_.clear();
      this->bayes_consecutive_accepts_ = 0;
    }
    // Stop the IMU thread from dead-reckoning off a stale (flung) nav state
    // while relocalization runs.
    this->resetGtsamNavState();

    // Arm guess-only relocalization around the clicked pose.
    this->relocalize_ = true;
    this->relocalized_ = false;
    this->reloc_guess_only_ = true;
    this->prior_map_pose_set_ = false;
    this->sc_attempt_count_ = 0;
    this->last_reloc_fitness_ = -1.0;

    RCLCPP_INFO(this->get_logger(),
                "[/initialpose] fresh relocalization: live keyframes cleared, odom reset, "
                "refining around [%.1f, %.1f]",
                this->initial_position_[0], this->initial_position_[1]);
  }

  // Scan Context Relocalization (retry up to sc_max_attempts_ scans)
  if (this->use_prior_map_ && this->relocalize_ && !this->relocalized_ && this->dlio_initialized && this->first_valid_scan)
  {

    this->sc_attempt_count_++;

    if (this->runRelocalization(this->current_scan))
    {
      if (this->deep_debug_)
        RCLCPP_INFO(this->get_logger(),
                    "  [RELOC] callback sees state.p=[%.2f,%.2f,%.2f] yaw=%.1f (should match ACCEPT best_pos)",
                    this->state.p[0], this->state.p[1], this->state.p[2],
                    std::atan2(2.f * (this->state.q.w() * this->state.q.z() + this->state.q.x() * this->state.q.y()),
                               1.f - 2.f * (this->state.q.y() * this->state.q.y() + this->state.q.z() * this->state.q.z())) *
                        180.f / M_PI);

      // Scan Context matched — set pose
      this->T = Eigen::Matrix4f::Identity();
      this->T.block<3, 3>(0, 0) = this->state.q.toRotationMatrix();
      this->T.block<3, 1>(0, 3) = this->state.p;
      this->T_prior = this->T;
      this->T_corr = Eigen::Matrix4f::Identity();
      this->lidarPose.p = this->state.p;
      this->lidarPose.q = this->state.q;
      this->prior_map_pose_set_ = true;
      this->relocalized_ = true;

      // Reset velocity and geometric observer to prevent drift from stale state
      this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
      this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
      this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);
      this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);

      {
        std::lock_guard<std::mutex> lock(this->geo.mtx);
        this->geo.prev_p = this->state.p;
        this->geo.prev_q = this->state.q;
        this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);
      }

      // Re-seed GTSAM nav state at the relocalized pose so the IMU thread does
      // NOT keep dead-reckoning from the old pose and fling the robot away.
      this->resetGtsamNavState();

      // Must set main_loop_running = false before buildSubmap to avoid deadlock
      this->main_loop_running = false;

      // Rebuild submap around relocalized pose
      this->buildSubmap(this->state);
      this->submap_hasChanged = true;
      this->new_submap_is_ready = true;

      // Reset map→odom correction and Bayesian state (odom just jumped to correct map position)
      {
        std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
        this->T_map_odom_ = Eigen::Matrix4f::Identity();
        this->bayes_posterior_.clear();
        this->bayes_consecutive_accepts_ = 0;
      }

      // Free relocalization resources (keep if continuous localization or submap localization needs them)
      if (!this->continuous_localize_ && !this->submap_loc_enabled_)
      {
        std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
        this->sc_database_.clear();
        this->prior_map_cloud_.reset();
        this->prior_map_kdtree_.reset();
      }

      RCLCPP_INFO(this->get_logger(),
                  "ScanContext relocalization succeeded after %d attempt(s)! Pose: [%.1f, %.1f, %.1f] yaw=%.1f deg",
                  this->sc_attempt_count_,
                  this->state.p[0], this->state.p[1], this->state.p[2],
                  std::atan2(2.f * (this->state.q.w() * this->state.q.z() + this->state.q.x() * this->state.q.y()),
                             1.f - 2.f * (this->state.q.y() * this->state.q.y() + this->state.q.z() * this->state.q.z())) *
                      180.f / M_PI);

      // Skip normal odometry for this scan — it was deskewed at the OLD pose.
      // Next scan will be properly deskewed at the new relocalized pose.
      this->prev_scan_stamp = this->scan_stamp;
      return;
    }
    else if (this->sc_attempt_count_ >= this->sc_max_attempts_)
    {
      // All attempts exhausted — fallback to configured initial pose
      RCLCPP_WARN(this->get_logger(),
                  "ScanContext: no match after %d attempts, falling back to initial pose [%.1f, %.1f, %.1f]",
                  this->sc_max_attempts_,
                  this->initial_position_[0], this->initial_position_[1], this->initial_position_[2]);

      this->state.p = this->initial_position_;
      this->origin = this->initial_position_;
      Eigen::Quaternionf yaw_q(Eigen::AngleAxisf(this->initial_yaw_, Eigen::Vector3f::UnitZ()));
      this->state.q = yaw_q * this->state.q;

      this->T = Eigen::Matrix4f::Identity();
      this->T.block<3, 3>(0, 0) = this->state.q.toRotationMatrix();
      this->T.block<3, 1>(0, 3) = this->state.p;
      this->T_prior = this->T;
      this->lidarPose.p = this->state.p;
      this->lidarPose.q = this->state.q;
      this->prior_map_pose_set_ = true;
      this->relocalized_ = true;

      // Zero velocity + re-seed GTSAM so the IMU thread doesn't fling the pose.
      this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
      this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
      this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);
      this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
      {
        std::lock_guard<std::mutex> lock(this->geo.mtx);
        this->geo.prev_p = this->state.p;
        this->geo.prev_q = this->state.q;
        this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);
      }
      this->resetGtsamNavState();

      this->main_loop_running = false;

      this->buildSubmap(this->state);
      this->submap_hasChanged = true;
      this->new_submap_is_ready = true;

      // Reset map→odom correction and Bayesian state
      {
        std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
        this->T_map_odom_ = Eigen::Matrix4f::Identity();
        this->bayes_posterior_.clear();
        this->bayes_consecutive_accepts_ = 0;
      }

      if (!this->continuous_localize_ && !this->submap_loc_enabled_)
      {
        std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
        this->sc_database_.clear();
        this->prior_map_cloud_.reset();
        this->prior_map_kdtree_.reset();
      }

      // Skip normal odometry for this scan — deskewed at old pose
      this->prev_scan_stamp = this->scan_stamp;
      return;
    }
    else
    {
      // Not matched yet, skip odometry and retry on next scan
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "ScanContext: attempt %d/%d, no match yet...",
                           this->sc_attempt_count_, this->sc_max_attempts_);
      this->prev_scan_stamp = this->scan_stamp;
      this->main_loop_running = false;
      return;
    }
  }

  if (this->current_scan->points.size() <= this->gicp_min_num_points_)
  {
    RCLCPP_FATAL(this->get_logger(), "Low number of points in the cloud!");
    return;
  }

  // Compute Metrics (detached thread — pass scan copy so original_scan can be safely overwritten)
  pcl::PointCloud<PointType>::ConstPtr scan_for_metrics = this->original_scan;
  this->metrics_thread = std::thread(&dlio::OdomNode::computeMetrics, this, scan_for_metrics);
  this->metrics_thread.detach();

  // Set Adaptive Parameters
  if (this->adaptive_params_)
  {
    this->setAdaptiveParams();
  }

  // Set new frame as input source
  this->setInputSource();

  // Set initial frame as first keyframe
  if (this->keyframes.size() == 0)
  {
    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(), "[DEEP] first keyframe: initializeInputTarget start, scan pts=%zu", this->current_scan->points.size());
    this->initializeInputTarget();
    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(), "[DEEP] first keyframe: initializeInputTarget done, keyframes=%zu", this->keyframes.size());
    this->main_loop_running = false;
    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(), "[DEEP] first keyframe: launching buildKeyframesAndSubmap");
    this->submap_future =
        std::async(std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state);
    this->submap_future.wait(); // wait until completion
    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(), "[DEEP] first keyframe: buildKeyframesAndSubmap done");
    return;
  }

  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] scan #%zu: past first kf, pts=%zu, submap_cloud=%zu",
                this->keyframes.size(), this->current_scan->points.size(),
                this->submap_cloud ? this->submap_cloud->points.size() : 0);

  // First real scan with prior map — register against prior map submap,
  // add as keyframe, and set prev_scan_stamp
  if (this->use_prior_map_ && this->prev_scan_stamp == 0.)
  {
    this->prev_scan_stamp = this->scan_stamp;
    // In localization we don't add live keyframes; buildKeyframesAndSubmap still
    // builds the prior-map ROI target below.
    if (this->keyframing_enabled_)
      this->initializeInputTarget();
    this->main_loop_running = false;
    this->submap_future =
        std::async(std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state);
    this->submap_future.wait();
    return;
  }

  // Save previous lidarPose for gate comparison (scan-to-scan jump detection)
  this->prev_lidarPose_ = this->lidarPose;

  // Registration: Get the next pose via IMU + Scan-to-Map
  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose start");
  {
    std::lock_guard<std::mutex> state_lock(this->state_mtx_);
    this->getNextPose(); // GICP align + propagateGICP only (no state update)
  }
  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose done, lidarPose=[%.2f,%.2f,%.2f]",
                this->lidarPose.p[0], this->lidarPose.p[1], this->lidarPose.p[2]);

  // Gate: evaluate scan matching quality before accepting
  bool gate_ok = this->evaluatePoseGate();

  if (gate_ok)
  {
    this->consecutive_gate_rejects_ = 0;
    this->last_gate_passed_ = true;

    // Save last known good forward speed from GICP-corrected state
    Eigen::Vector3f v_body = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;
    this->last_good_forward_speed_ = v_body[0];
  }
  else
  {
    // Gate rejected: use IMU-propagated state instead of bad GICP result
    this->consecutive_gate_rejects_++;
    this->last_gate_passed_ = false;

    this->lidarPose.p = this->state.p;
    this->lidarPose.q = this->state.q;
    this->geo.prev_vel = this->state.v.lin.w;
    this->T.block(0, 0, 3, 3) = this->state.q.toRotationMatrix();
    this->T.block(0, 3, 3, 1) = this->state.p;
    this->T_corr = Eigen::Matrix4f::Identity();
  }

  // Always: update state, keyframes, submap (with GICP pose or IMU pose)
  {
    std::lock_guard<std::mutex> state_lock(this->state_mtx_);
    this->updateState();
  }

  // Reset GTSAM preintegration with corrected state (after LiDAR correction)
  this->resetGtsamNavState();

  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] gate=%s, updateState done",
                gate_ok ? "PASS" : "REJECT(IMU)");

  // Motion model pose filter: constraint scan-to-scan displacement
  // Runs at scan rate (10Hz), after all corrections — doesn't fight GICP
  if (this->motion_model_type_ != dlio::MotionModelType::NONE)
  {
    Eigen::Vector3f dp_world = this->state.p - this->prev_state_p_;
    Eigen::Matrix3f R_inv = this->state.q.toRotationMatrix().inverse();
    Eigen::Vector3f dp_body = R_inv * dp_world;

    float scan_dt = std::max(0.01f, static_cast<float>(this->scan_stamp - this->prev_scan_stamp));
    const auto &p = this->mm_params_;

    switch (this->motion_model_type_)
    {
    case dlio::MotionModelType::ACKERMANN:
      dp_body[0] = std::clamp(dp_body[0], -p.ack_max_rev_vel * scan_dt, p.ack_max_fwd_vel * scan_dt);
      dp_body[1] = std::clamp(dp_body[1], -p.ack_max_lat_vel * scan_dt, p.ack_max_lat_vel * scan_dt);
      dp_body[2] = std::clamp(dp_body[2], -p.ack_max_vert_vel * scan_dt, p.ack_max_vert_vel * scan_dt);
      break;
    case dlio::MotionModelType::DIFF_DRIVE:
      dp_body[0] = std::clamp(dp_body[0], -p.dd_max_rev_vel * scan_dt, p.dd_max_fwd_vel * scan_dt);
      dp_body[1] = std::clamp(dp_body[1], -p.dd_max_lat_vel * scan_dt, p.dd_max_lat_vel * scan_dt);
      dp_body[2] = std::clamp(dp_body[2], -p.dd_max_vert_vel * scan_dt, p.dd_max_vert_vel * scan_dt);
      break;
    case dlio::MotionModelType::HOLONOMIC:
      dp_body[0] = std::clamp(dp_body[0], -p.holo_max_horiz_vel * scan_dt, p.holo_max_horiz_vel * scan_dt);
      dp_body[1] = std::clamp(dp_body[1], -p.holo_max_horiz_vel * scan_dt, p.holo_max_horiz_vel * scan_dt);
      dp_body[2] = std::clamp(dp_body[2], -p.holo_max_vert_vel * scan_dt, p.holo_max_vert_vel * scan_dt);
      break;
    default:
      break;
    }

    this->state.p = this->prev_state_p_ + this->state.q.toRotationMatrix() * dp_body;
    this->lidarPose.p = this->state.p;
    this->T.block(0, 3, 3, 1) = this->state.p;
  }
  this->prev_state_p_ = this->state.p;

  this->updateKeyframes();

  // Build keyframe normals and submap if needed
  if (this->new_submap_is_ready)
  {
    this->main_loop_running = false;
    this->submap_future =
        std::async(std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state);
  }
  else
  {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
  }

  // Update trajectory
  this->trajectory.push_back(std::make_pair(this->state.p, this->state.q));

  // Update length traversed (needed by publishCloud when waitUntilMove is true)
  if (this->trajectory.size() >= 2)
  {
    Eigen::Vector3f prev_p = this->trajectory[this->trajectory.size() - 2].first;
    double dl = (this->state.p - prev_p).norm();
    if (dl >= 0.05)
      this->length_traversed += dl;
  }

  // Update time stamps
  this->lidar_rates.push_back(1. / (this->scan_stamp - this->prev_scan_stamp));
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // Publish stuff to ROS
  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->densemap_filtered_)
  {
    published_cloud = this->current_scan;
  }
  else
  {
    published_cloud = this->deskewed_scan;
  }
  // Capture raw (unfiltered) deskewed scan for graph SLAM SC computation
  pcl::PointCloud<PointType>::ConstPtr raw_deskewed = this->deskewed_scan;
  // Publish to ROS (detached thread, publish_mtx_ prevents concurrent push_back on path_ros.poses)
  this->publish_thread = std::thread(&dlio::OdomNode::publishToROS, this, published_cloud, raw_deskewed, this->T_corr);
  this->publish_thread.detach();

  // Store latest scan (sensor/body frame) + its T for continuous localization
  // IMPORTANT: use original_scan (sensor frame), NOT current_scan (odom/world frame).
  // The SC computation in continuousLocalize() assumes body-frame input and applies
  // its own gravity rotation. Using odom-frame scans causes double-rotation → SC fails.
  if (this->continuous_localize_ || this->occupancy_grid_enabled_ || this->submap_loc_enabled_)
  {
    std::lock_guard<std::mutex> lock(this->latest_scan_mtx_);
    this->latest_scan_ = this->original_scan;
    this->latest_scan_T_ = this->T;
    this->latest_scan_time_ = this->now().seconds();
  }

  // Update some statistics
  double comp_time = this->now().seconds() - then;
  this->comp_times.push_back(comp_time);
  this->gicp_hasConverged = this->engine_.hasConverged();

  if (this->debug_)
  {
    size_t n_kf = this->keyframes.size();
    RCLCPP_INFO(this->get_logger(),
                "[odom] scan=%zu pts | keyframes=%zu | converged=%d | fitness=%.4f | dt=%.3fs | pos=[%.2f,%.2f,%.2f]",
                this->current_scan->points.size(), n_kf,
                static_cast<int>(this->gicp_hasConverged.load()),
                this->last_fitness_,
                comp_time, this->state.p[0], this->state.p[1], this->state.p[2]);
  }

  this->geo.first_opt_done = true;
}

void dlio::OdomNode::callbackExternalOdom(const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg)
{
  Eigen::Vector3f vel_odom(
      static_cast<float>(msg->twist.twist.linear.x),
      static_cast<float>(msg->twist.twist.linear.y),
      static_cast<float>(msg->twist.twist.linear.z));

  vel_odom = vel_odom.cwiseProduct(this->ext_odom_scale_);

  {
    std::lock_guard<std::mutex> lock(this->ext_odom_mtx_);
    this->ext_odom_vel_body_ = this->ext_baselink2odom_.R * vel_odom;
  }

  this->ext_odom_received_ = true;
  this->ext_odom_stamp_ = rclcpp::Time(msg->header.stamp).seconds();
}

void dlio::OdomNode::callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu_raw)
{

  this->first_imu_received = true;

  // If IMU publishes accel in g units, scale to m/s^2 first so downstream
  // (gravity reconstruction, calibration, propagation) sees consistent SI units.
  if (this->imu_accel_in_g_)
  {
    imu_raw->linear_acceleration.x *= this->gravity_;
    imu_raw->linear_acceleration.y *= this->gravity_;
    imu_raw->linear_acceleration.z *= this->gravity_;
  }

  // If IMU driver already removed gravity from accel:
  // Simply add gravity back as [0, 0, +g] in sensor frame (assuming Z-up IMU).
  // propagateState() will subtract gravity in world frame, cancelling it out.
  // This avoids circular dependency on state.q for gravity reconstruction.
  if (this->imu_gravity_removed_)
  {
    imu_raw->linear_acceleration.z += this->gravity_;
  }

  // Differential orientation mode: derive angular velocity from orientation quaternion delta
  if (this->imu_differential_orientation_)
  {
    Eigen::Quaternionf q_curr(
        static_cast<float>(imu_raw->orientation.w),
        static_cast<float>(imu_raw->orientation.x),
        static_cast<float>(imu_raw->orientation.y),
        static_cast<float>(imu_raw->orientation.z));

    if (q_curr.squaredNorm() > 0.1f)
    {
      double stamp = rclcpp::Time(imu_raw->header.stamp).seconds();
      if (this->imu_prev_orientation_valid_)
      {
        float dt = static_cast<float>(stamp - this->imu_prev_orientation_stamp_);
        if (dt > 0.f && dt < 1.0f)
        {
          Eigen::Quaternionf dq = this->imu_prev_orientation_.conjugate() * q_curr;
          dq.normalize();
          if (dq.w() < 0.f)
            dq.coeffs() = -dq.coeffs(); // shortest path
          Eigen::Vector3f omega = 2.0f * dq.vec() / dt;
          imu_raw->angular_velocity.x = omega.x();
          imu_raw->angular_velocity.y = omega.y();
          imu_raw->angular_velocity.z = omega.z();
        }
      }
      this->imu_prev_orientation_ = q_curr;
      this->imu_prev_orientation_stamp_ = stamp;
      this->imu_prev_orientation_valid_ = true;
    }
  }

  sensor_msgs::msg::Imu::SharedPtr imu = this->transformImu(imu_raw);
  this->imu_stamp = imu->header.stamp;
  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();

  Eigen::Vector3f lin_accel;
  Eigen::Vector3f ang_vel;

  // Get IMU samples
  ang_vel[0] = imu->angular_velocity.x;
  ang_vel[1] = imu->angular_velocity.y;
  ang_vel[2] = imu->angular_velocity.z;

  lin_accel[0] = imu->linear_acceleration.x;
  lin_accel[1] = imu->linear_acceleration.y;
  lin_accel[2] = imu->linear_acceleration.z;

  if (this->first_imu_stamp == 0.)
  {
    this->first_imu_stamp = imu_stamp_secs;
  }

  // IMU calibration procedure - do for three seconds
  if (!this->imu_calibrated)
  {

    static int num_samples = 0;
    static Eigen::Vector3f gyro_avg(0., 0., 0.);
    static Eigen::Vector3f accel_avg(0., 0., 0.);
    static bool print = true;

    if ((imu_stamp_secs - this->first_imu_stamp) < this->imu_calib_time_)
    {

      num_samples++;

      gyro_avg[0] += ang_vel[0];
      gyro_avg[1] += ang_vel[1];
      gyro_avg[2] += ang_vel[2];

      accel_avg[0] += lin_accel[0];
      accel_avg[1] += lin_accel[1];
      accel_avg[2] += lin_accel[2];

      if (print)
      {
        std::cout << std::endl
                  << " Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
        std::cout.flush();
        print = false;
      }
    }
    else
    {

      std::cout << "done" << std::endl
                << std::endl;

      gyro_avg /= num_samples;
      accel_avg /= num_samples;

      Eigen::Vector3f grav_vec(0., 0., this->gravity_);

      // Compute gravity quaternion from IMU calibration (used for KFDB SC frame)
      // Only set during mapping; in localization mode, kfdb_gravity_q_ is loaded from the KFDB file
      if (this->map_mode_ == "mapping")
      {
        Eigen::Vector3f grav_est = (accel_avg - this->state.b.accel).normalized() * abs(this->gravity_);
        this->kfdb_gravity_q_ = Eigen::Quaternionf::FromTwoVectors(grav_est, Eigen::Vector3f(0., 0., this->gravity_));
      }

      if (this->gravity_align_)
      {

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        grav_vec = (accel_avg - this->state.b.accel).normalized() * abs(this->gravity_);
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(grav_vec, Eigen::Vector3f(0., 0., this->gravity_));

        // set gravity aligned orientation
        this->state.q = grav_q;
        this->T.block(0, 0, 3, 3) = this->state.q.toRotationMatrix();
        this->lidarPose.q = this->state.q;

        // rpy
        auto euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0] * (180.0 / M_PI);
        double pitch = euler[1] * (180.0 / M_PI);
        double roll = euler[2] * (180.0 / M_PI);

        // use alternate representation if the yaw is smaller
        if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw))
        {
          yaw = remainder(yaw + 180.0, 360.0);
          pitch = remainder(180.0 - pitch, 360.0);
          roll = remainder(roll + 180.0, 360.0);
        }
        std::cout << " Estimated initial attitude:" << std::endl;
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << std::endl;
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << std::endl;
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << std::endl;
        std::cout << std::endl;
      }

      if (this->calibrate_accel_)
      {

        // subtract gravity from avg accel to get bias
        this->state.b.accel = accel_avg - grav_vec;

        std::cout << " Accel biases [xyz]: " << to_string_with_precision(this->state.b.accel[0], 8) << ", "
                  << to_string_with_precision(this->state.b.accel[1], 8) << ", "
                  << to_string_with_precision(this->state.b.accel[2], 8) << std::endl;
      }

      if (this->calibrate_gyro_)
      {

        this->state.b.gyro = gyro_avg;

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(this->state.b.gyro[0], 8) << ", "
                  << to_string_with_precision(this->state.b.gyro[1], 8) << ", "
                  << to_string_with_precision(this->state.b.gyro[2], 8) << std::endl;
      }

      this->prev_imu_stamp = imu_stamp_secs;
      this->imu_calibrated = true;

      // Initialize GTSAM nav state after calibration
      if (this->imu_preintegration_mode_ == "gtsam")
      {
        gtsam::Rot3 R0(this->state.q.cast<double>());
        gtsam::Point3 p0(this->state.p.cast<double>());
        gtsam::Velocity3 v0 = gtsam::Velocity3::Zero();
        this->gtsam_nav_state_ = gtsam::NavState(gtsam::Pose3(R0, p0), v0);
        // Bias = zero because imu_meas is already bias-corrected in callbackImu()
        this->gtsam_bias_ = gtsam::imuBias::ConstantBias();
        this->imu_preintegration_->resetIntegrationAndSetBias(this->gtsam_bias_);
        this->gtsam_imu_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "[odom] GTSAM nav state initialized after IMU calibration");
      }
    }
  }
  else
  {

    double dt = imu_stamp_secs - this->prev_imu_stamp;
    if (dt <= 0 || dt > 1.0)
    {
      dt = 1.0 / 200.0;
    }
    this->imu_rates.push_back(1. / dt);

    // Apply the calibrated bias to the new IMU measurements
    this->imu_meas.stamp = imu_stamp_secs;
    this->imu_meas.dt = dt;
    this->prev_imu_stamp = this->imu_meas.stamp;

    Eigen::Vector3f lin_accel_corrected = (this->imu_accel_sm_ * lin_accel) - this->state.b.accel;
    Eigen::Vector3f ang_vel_corrected = ang_vel - this->state.b.gyro;

    // use_2d_imu: constrain IMU to 2D plane
    // - Accel Z: replace with expected gravity in body frame (not zero! zero causes freefall)
    // - Gyro: keep only yaw (Z), zero roll/pitch
    if (this->use_2d_imu_)
    {
      // Replace Z accel with gravity projection in body frame so gravity cancels perfectly
      Eigen::Vector3f gravity_body = this->state.q.conjugate()._transformVector(
          Eigen::Vector3f(0.f, 0.f, this->gravity_));
      lin_accel_corrected[2] = gravity_body[2];

      ang_vel_corrected[0] = 0.f; // zero roll rate
      ang_vel_corrected[1] = 0.f; // zero pitch rate
    }

    this->imu_meas.lin_accel = lin_accel_corrected;
    this->imu_meas.ang_vel = ang_vel_corrected;

    // Store calibrated IMU measurements into imu buffer for manual integration later.
    this->mtx_imu.lock();
    this->imu_buffer.push_front(this->imu_meas);
    this->mtx_imu.unlock();

    // Notify the callbackPointCloud thread that IMU data exists for this time
    this->cv_imu_stamp.notify_one();

    if (this->use_imu_ && this->geo.first_opt_done)
    {
      // Geometric Observer: Propagate State
      this->propagateState();
    }
  }
}

sensor_msgs::msg::Imu::SharedPtr dlio::OdomNode::transformImu(const sensor_msgs::msg::Imu::SharedPtr &imu_raw)
{

  auto imu = std::make_shared<sensor_msgs::msg::Imu>();

  // Copy header
  imu->header = imu_raw->header;

  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();
  if (this->imu_transform_prev_stamp_ == 0.0)
    this->imu_transform_prev_stamp_ = imu_stamp_secs;
  double dt = imu_stamp_secs - this->imu_transform_prev_stamp_;
  this->imu_transform_prev_stamp_ = imu_stamp_secs;

  if (dt == 0)
  {
    dt = 1.0 / 200.0;
  }

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  Eigen::Vector3f ang_vel(imu_raw->angular_velocity.x,
                          imu_raw->angular_velocity.y,
                          imu_raw->angular_velocity.z);

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  // Transform linear acceleration (need to account for component due to translational difference)
  Eigen::Vector3f lin_accel(imu_raw->linear_acceleration.x,
                            imu_raw->linear_acceleration.y,
                            imu_raw->linear_acceleration.z);

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg = lin_accel_cg + ((ang_vel_cg - this->imu_transform_ang_vel_prev_) / dt).cross(-this->extrinsics.baselink2imu.t) + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  this->imu_transform_ang_vel_prev_ = ang_vel_cg;

  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;
}
