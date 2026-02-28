/***********************************************************
 *                                                         *
 * Submap-based relocalization for GPS-denied environments *
 *                                                         *
 * Partitions prior map keyframes into submaps, performs    *
 * brute-force GICP registration (Stage 1) then motion     *
 * prediction validation (Stage 2) before applying TF      *
 * correction.                                             *
 *                                                         *
 ***********************************************************/

#include "dlio/odom.h"

void dlio::OdomNode::initSubmapLocalization()
{
  if (this->sc_database_.empty())
  {
    RCLCPP_WARN(this->get_logger(), "[submap_loc] SC database empty, cannot init submaps");
    return;
  }
  if (!this->prior_map_cloud_ || !this->prior_map_kdtree_)
  {
    RCLCPP_WARN(this->get_logger(), "[submap_loc] Prior map not loaded, cannot init submaps");
    return;
  }

  const int N = static_cast<int>(this->sc_database_.size());
  const int group_size = this->submap_loc_group_size_;
  const int num_submaps = (N + group_size - 1) / group_size;

  float search_radius_sq = static_cast<float>(
      this->submap_loc_search_radius_ * this->submap_loc_search_radius_);

  this->submap_entries_.clear();
  this->submap_entries_.reserve(num_submaps);

  for (int si = 0; si < num_submaps; si++)
  {
    SubmapEntry entry;
    entry.submap_id = si;
    entry.probability = 1.0f / static_cast<float>(num_submaps);

    // Collect KF indices for this submap
    int start_idx = si * group_size;
    int end_idx = std::min(start_idx + group_size, N);
    for (int ki = start_idx; ki < end_idx; ki++)
      entry.kf_indices.push_back(ki);

    // Compute centroid from member KF positions
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (int ki : entry.kf_indices)
      centroid += this->sc_database_[ki].position;
    centroid /= static_cast<float>(entry.kf_indices.size());
    entry.centroid = centroid;

    // Build submap cloud via radius search from centroid
    PointType center_pt;
    center_pt.x = centroid[0];
    center_pt.y = centroid[1];
    center_pt.z = centroid[2];

    std::vector<int> indices;
    std::vector<float> dists;
    this->prior_map_kdtree_->radiusSearch(center_pt, search_radius_sq, indices, dists);

    if (indices.size() < 50)
    {
      RCLCPP_WARN(this->get_logger(),
                  "[submap_loc] Submap %d: too few points (%zu) at centroid [%.1f,%.1f,%.1f], skipping",
                  si, indices.size(), centroid[0], centroid[1], centroid[2]);
      continue;
    }

    entry.cloud = std::make_shared<pcl::PointCloud<PointType>>();
    entry.cloud->points.resize(indices.size());
    for (size_t i = 0; i < indices.size(); i++)
      entry.cloud->points[i] = this->prior_map_cloud_->points[indices[i]];
    entry.cloud->width = entry.cloud->points.size();
    entry.cloud->height = 1;
    entry.cloud->is_dense = true;

    // Voxel filter 0.5m
    pcl::VoxelGrid<PointType> vf;
    vf.setLeafSize(0.5f, 0.5f, 0.5f);
    vf.setInputCloud(entry.cloud);
    vf.filter(*entry.cloud);
    entry.cloud->width = entry.cloud->points.size();
    entry.cloud->height = 1;

    this->submap_entries_.push_back(std::move(entry));
  }

  // Init state
  this->submap_loc_state_ = SubmapLocState::STAGE1;

  // Init shadow T_map_odom from current system value
  {
    std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
    this->submap_loc_T_map_odom_shadow_ = this->T_map_odom_;
  }

  // Clear scan buffer
  this->submap_loc_scan_buffer_.clear();

  // Init motion tracker
  this->submap_motion_.current_kf_idx = -1;
  this->submap_motion_.prev_kf_idx = -1;
  this->submap_motion_.T_odom_prev = Eigen::Matrix4f::Identity();
  this->submap_motion_.T_map_body_prev = Eigen::Matrix4f::Identity();
  this->submap_motion_.consecutive_valid = 0;
  this->submap_motion_.total_validated = 0;

  RCLCPP_INFO(this->get_logger(),
              "[submap_loc] Initialized %zu submaps from %d keyframes (group_size=%d)",
              this->submap_entries_.size(), N, group_size);
}

void dlio::OdomNode::submapLocalizeTick()
{
  // Prerequisites
  if (!this->dlio_initialized || !this->relocalized_)
    return;

  {
    std::lock_guard<std::mutex> lock(this->submap_loc_mtx_);
    if (this->submap_entries_.empty())
      return;
    if (this->submap_loc_state_ == SubmapLocState::IDLE)
      return;
  }

  // Get latest scan + T_odom_body
  pcl::PointCloud<PointType>::ConstPtr scan_body;
  Eigen::Matrix4f T_odom_body;
  double scan_time;
  {
    std::lock_guard<std::mutex> lock(this->latest_scan_mtx_);
    scan_body = this->latest_scan_;
    T_odom_body = this->latest_scan_T_;
    scan_time = this->latest_scan_time_;
  }
  if (!scan_body || scan_body->empty())
    return;

  // Skip stale scans
  double now_sec = this->now().seconds();
  double scan_age = now_sec - scan_time;
  if (scan_time > 0.0 && scan_age > 3.0 * this->submap_loc_interval_)
    return;

  // Buffer recent scans for accumulated SC descriptor
  {
    auto cloud_copy = std::make_shared<pcl::PointCloud<PointType>>(*scan_body);
    this->submap_loc_scan_buffer_.push_back({cloud_copy, T_odom_body});
    int max_buf = std::max(1, this->submap_loc_sc_accum_scans_);
    while (static_cast<int>(this->submap_loc_scan_buffer_.size()) > max_buf)
      this->submap_loc_scan_buffer_.pop_front();
  }

  // Dispatch
  std::lock_guard<std::mutex> lock(this->submap_loc_mtx_);
  if (this->submap_loc_state_ == SubmapLocState::STAGE1)
    this->submapLocalizeStage1(scan_body, T_odom_body);
  else if (this->submap_loc_state_ == SubmapLocState::STAGE2)
    this->submapLocalizeStage2(scan_body, T_odom_body);
}

void dlio::OdomNode::submapLocalizeStage1(
    pcl::PointCloud<PointType>::ConstPtr scan_body,
    const Eigen::Matrix4f &T_odom_body)
{
  // Use shadow T_map_odom for orientation estimate (does not touch system variable)
  Eigen::Matrix4f T_map_body_est = this->submap_loc_T_map_odom_shadow_ * T_odom_body;
  Eigen::Matrix4f T_body_lidar = this->extrinsics.baselink2lidar_T;

  float best_fitness = std::numeric_limits<float>::max();
  float best_prob = 0.0f;
  int best_submap_idx = -1;
  Eigen::Matrix4f best_result_T = Eigen::Matrix4f::Identity();

  // SC filter: compute descriptor from accumulated scans to reject dissimilar submaps
  dlio::sc::ScanContextDescriptor query_sc_desc;
  dlio::sc::SectorKey query_sector_key;
  bool sc_filter_enabled = (this->submap_loc_sc_dist_thresh_ > 0.0f &&
                            !this->sc_database_.empty());
  if (sc_filter_enabled)
  {
    // Accumulate recent scans into current body frame
    pcl::PointCloud<PointType>::Ptr accum_cloud;
    if (this->submap_loc_scan_buffer_.size() <= 1)
    {
      accum_cloud = std::make_shared<pcl::PointCloud<PointType>>(*scan_body);
    }
    else
    {
      accum_cloud = std::make_shared<pcl::PointCloud<PointType>>();
      Eigen::Matrix4f T_curr_inv = T_odom_body.inverse();
      for (const auto &entry : this->submap_loc_scan_buffer_)
      {
        // Transform past scan from its odom frame to current body frame
        Eigen::Matrix4f T_curr_past = T_curr_inv * entry.T_odom_body;
        for (const auto &pt : entry.cloud->points)
        {
          Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
          Eigen::Vector4f p_tf = T_curr_past * p;
          PointType pt_tf;
          pt_tf.x = p_tf[0];
          pt_tf.y = p_tf[1];
          pt_tf.z = p_tf[2];
          pt_tf.intensity = pt.intensity;
          accum_cloud->points.push_back(pt_tf);
        }
      }
      accum_cloud->width = accum_cloud->points.size();
      accum_cloud->height = 1;
      accum_cloud->is_dense = true;
    }

    auto sc_scan = dlio::sc::prepareGravityAlignedScan(
        accum_cloud, this->kfdb_gravity_q_, this->extrinsics.baselink2lidar.R,
        this->sc_ground_height_threshold_);
    query_sc_desc = dlio::sc::computeScanContext(sc_scan, this->sc_max_range_);
    query_sector_key = dlio::sc::computeSectorKey(query_sc_desc);
  }

  int sc_passed = 0, sc_rejected = 0;

  for (size_t si = 0; si < this->submap_entries_.size(); si++)
  {
    auto &sm = this->submap_entries_[si];
    if (!sm.cloud || sm.cloud->empty())
      continue;

    // SC filter: check min SC distance against member KFs
    if (sc_filter_enabled)
    {
      float min_sc_dist = std::numeric_limits<float>::max();
      for (int ki : sm.kf_indices)
      {
        if (ki < 0 || ki >= static_cast<int>(this->sc_database_.size()))
          continue;
        auto [dist, shift] = dlio::sc::computeScanContextDistance(
            query_sc_desc, this->sc_database_[ki].descriptor,
            query_sector_key, this->sc_database_[ki].sector_key,
            this->sc_search_window_);
        if (dist < min_sc_dist)
          min_sc_dist = dist;
      }
      if (min_sc_dist > this->submap_loc_sc_dist_thresh_)
      {
        sc_rejected++;
        continue; // skip this submap — too dissimilar
      }
      sc_passed++;
    }

    // Build init_guess: centroid position + current orientation from IMU
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    init_guess.block<3, 3>(0, 0) = T_map_body_est.block<3, 3>(0, 0); // keep orientation
    init_guess.block<3, 1>(0, 3) = sm.centroid;                      // position = submap centroid
    init_guess = init_guess * T_body_lidar;                          // convert to lidar frame

    // GICP/NDT registration
    pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
    bool converged = false;
    float fitness = std::numeric_limits<float>::max();
    Eigen::Matrix4f result_T = Eigen::Matrix4f::Identity();

    if (this->use_gicp_)
    {
      nano_gicp::NanoGICP<PointType, PointType> gicp;
      gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
      gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
      gicp.setMaximumIterations(64);
      gicp.setTransformationEpsilon(0.05);
      gicp.setRotationEpsilon(0.05);

      gicp.setInputSource(scan_body);
      gicp.calculateSourceCovariances();
      gicp.setInputTarget(sm.cloud);
      gicp.calculateTargetCovariances();

      gicp.align(*aligned, init_guess);
      converged = gicp.hasConverged();
      fitness = gicp.getFitnessScore(1.0);
      result_T = gicp.getFinalTransformation();
    }
    else
    {
      pclomp::NormalDistributionsTransform<PointType, PointType> ndt_local;
      ndt_local.setResolution(this->ndt_resolution_);
      ndt_local.setNumThreads(this->ndt_num_threads_);
      ndt_local.setNeighborhoodSearchMethod(pclomp::DIRECT7);
      ndt_local.setMaximumIterations(64);
      ndt_local.setTransformationEpsilon(0.05);

      ndt_local.setInputSource(scan_body);
      ndt_local.setInputTarget(sm.cloud);

      ndt_local.align(*aligned, init_guess);
      converged = ndt_local.hasConverged();
      fitness = ndt_local.getFitnessScore(1.0);
      result_T = ndt_local.getFinalTransformation();
    }

    float prob = converged ? (1.0f / (1.0f + fitness)) : 0.0f;
    sm.probability = prob;

    if (converged && fitness < best_fitness)
    {
      best_fitness = fitness;
      best_prob = prob;
      best_submap_idx = static_cast<int>(si);
      best_result_T = result_T;
    }
  }

  if (this->debug_)
    RCLCPP_INFO(this->get_logger(),
                "[submap_loc] Stage1: best_submap=%d fitness=%.4f prob=%.3f thresh=%.3f sc_pass=%d sc_reject=%d",
                best_submap_idx, best_fitness, best_prob, this->submap_loc_prob_threshold_,
                sc_passed, sc_rejected);

  // Check if best submap passes threshold
  if (best_submap_idx >= 0 &&
      best_prob > this->submap_loc_prob_threshold_ &&
      best_fitness < this->submap_loc_fitness_thresh_)
  {
    // GICP result is in lidar frame, convert to body frame
    Eigen::Matrix4f T_map_lidar_gicp = best_result_T;
    Eigen::Matrix4f T_map_body_gicp = T_map_lidar_gicp * T_body_lidar.inverse();
    Eigen::Vector3f gicp_pos = T_map_body_gicp.block<3, 1>(0, 3);

    // Update shadow T_map_odom from Stage1 GICP result
    // This gives Stage2 an accurate init_guess without touching the system variable
    this->submap_loc_T_map_odom_shadow_ = T_map_body_gicp * T_odom_body.inverse();

    // Find closest KF
    int closest_kf = this->findClosestKF(gicp_pos);

    // Init motion tracker
    this->submap_motion_.current_kf_idx = closest_kf;
    this->submap_motion_.prev_kf_idx = closest_kf;
    this->submap_motion_.T_odom_prev = T_odom_body;
    this->submap_motion_.T_map_body_prev = T_map_body_gicp;
    this->submap_motion_.consecutive_valid = 0;
    this->submap_motion_.total_validated = 0;

    this->submap_loc_state_ = SubmapLocState::STAGE2;

    if (this->debug_)
      RCLCPP_INFO(this->get_logger(),
                  "[submap_loc] Stage1→Stage2: submap=%d kf=%d fitness=%.4f prob=%.3f pos=[%.1f,%.1f,%.1f]",
                  best_submap_idx, closest_kf, best_fitness, best_prob,
                  gicp_pos[0], gicp_pos[1], gicp_pos[2]);
  }
}

void dlio::OdomNode::submapLocalizeStage2(
    pcl::PointCloud<PointType>::ConstPtr scan_body,
    const Eigen::Matrix4f &T_odom_body)
{
  // Use shadow T_map_odom for position estimate (does not touch system variable)
  Eigen::Matrix4f T_map_body_est = this->submap_loc_T_map_odom_shadow_ * T_odom_body;
  Eigen::Matrix4f T_body_lidar = this->extrinsics.baselink2lidar_T;
  Eigen::Vector3f est_pos = T_map_body_est.block<3, 1>(0, 3);
  Eigen::Vector3f gicp_result_pos = Eigen::Vector3f::Zero(); // set after GICP

  // Find nearest submap to estimated position
  int nearest_si = -1;
  float min_dist = std::numeric_limits<float>::max();
  for (size_t si = 0; si < this->submap_entries_.size(); si++)
  {
    float d = (this->submap_entries_[si].centroid - est_pos).norm();
    if (d < min_dist)
    {
      min_dist = d;
      nearest_si = static_cast<int>(si);
    }
  }

  if (nearest_si < 0 || !this->submap_entries_[nearest_si].cloud ||
      this->submap_entries_[nearest_si].cloud->empty())
  {
    RCLCPP_WARN(this->get_logger(), "[submap_loc] Stage2: no valid nearest submap");
    return;
  }

  auto &sm = this->submap_entries_[nearest_si];

  // GICP/NDT against nearest submap (init_guess from shadow estimate)
  Eigen::Matrix4f init_guess = T_map_body_est * T_body_lidar; // lidar frame

  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  bool converged = false;
  float fitness = std::numeric_limits<float>::max();
  Eigen::Matrix4f result_T = Eigen::Matrix4f::Identity();

  if (this->use_gicp_)
  {
    nano_gicp::NanoGICP<PointType, PointType> gicp;
    gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
    gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
    gicp.setMaximumIterations(64);
    gicp.setTransformationEpsilon(0.05);
    gicp.setRotationEpsilon(0.05);

    gicp.setInputSource(scan_body);
    gicp.calculateSourceCovariances();
    gicp.setInputTarget(sm.cloud);
    gicp.calculateTargetCovariances();

    gicp.align(*aligned, init_guess);
    converged = gicp.hasConverged();
    fitness = gicp.getFitnessScore(1.0);
    result_T = gicp.getFinalTransformation();
  }
  else
  {
    pclomp::NormalDistributionsTransform<PointType, PointType> ndt_local;
    ndt_local.setResolution(this->ndt_resolution_);
    ndt_local.setNumThreads(this->ndt_num_threads_);
    ndt_local.setNeighborhoodSearchMethod(pclomp::DIRECT7);
    ndt_local.setMaximumIterations(64);
    ndt_local.setTransformationEpsilon(0.05);

    ndt_local.setInputSource(scan_body);
    ndt_local.setInputTarget(sm.cloud);

    ndt_local.align(*aligned, init_guess);
    converged = ndt_local.hasConverged();
    fitness = ndt_local.getFitnessScore(1.0);
    result_T = ndt_local.getFinalTransformation();
  }

  gicp_result_pos = result_T.block<3, 1>(0, 3);
  float prob = converged ? (1.0f / (1.0f + fitness)) : 0.0f;
  sm.probability = prob;

  // Reject if not converged, prob dropped, or fitness too high
  if (!converged || prob < this->submap_loc_prob_drop_thresh_ ||
      fitness > this->submap_loc_fitness_thresh_)
  {
    RCLCPP_WARN(this->get_logger(),
                "[submap_loc] Stage2: reject — submap=%d fitness=%.4f prob=%.3f converged=%d "
                "est=[%.1f,%.1f,%.1f] gicp=[%.1f,%.1f,%.1f] sm_centroid=[%.1f,%.1f,%.1f] sm_dist=%.1f scan=%zu sm_pts=%zu",
                nearest_si, fitness, prob, converged,
                est_pos[0], est_pos[1], est_pos[2],
                gicp_result_pos[0], gicp_result_pos[1], gicp_result_pos[2],
                sm.centroid[0], sm.centroid[1], sm.centroid[2],
                min_dist, scan_body->size(), sm.cloud->size());

    // Not converged or prob dropped → revert to Stage1
    // Bad fitness alone → skip tick, stay in Stage2
    if (!converged || prob < this->submap_loc_prob_drop_thresh_)
    {
      this->submap_loc_state_ = SubmapLocState::STAGE1;
      this->submap_motion_.consecutive_valid = 0;
    }
    return;
  }

  // Convert GICP result from lidar frame to body frame
  Eigen::Matrix4f T_map_lidar_gicp = result_T;
  Eigen::Matrix4f T_map_body_gicp = T_map_lidar_gicp * T_body_lidar.inverse();

  // Only update shadow from GICP results with good fitness
  this->submap_loc_T_map_odom_shadow_ = T_map_body_gicp * T_odom_body.inverse();

  // Motion prediction validation
  // pTlink = T_odom_prev⁻¹ * T_odom_curr (relative transform from odometry)
  Eigen::Matrix4f pTlink = this->submap_motion_.T_odom_prev.inverse() * T_odom_body;

  // Tlink = T_map_body_prev⁻¹ * T_map_body_gicp (relative transform from GICP)
  Eigen::Matrix4f Tlink = this->submap_motion_.T_map_body_prev.inverse() * T_map_body_gicp;

  // delta = Tlink⁻¹ * pTlink
  Eigen::Matrix4f delta = Tlink.inverse() * pTlink;

  float pos_error = delta.block<3, 1>(0, 3).norm();
  Eigen::Quaternionf q_delta(delta.block<3, 3>(0, 0));
  float rot_error = 2.0f * std::acos(std::min(std::abs(q_delta.w()), 1.0f)) * 180.0f / M_PI;

  bool motion_valid = (pos_error < this->submap_loc_motion_error_thresh_) &&
                      (rot_error < this->submap_loc_motion_rot_thresh_);

  if (motion_valid)
  {
    this->submap_motion_.consecutive_valid++;
    this->submap_motion_.total_validated++;
  }
  else
  {
    this->submap_motion_.consecutive_valid = 0;
  }

  // Update motion tracker state for next tick
  this->submap_motion_.T_odom_prev = T_odom_body;
  this->submap_motion_.T_map_body_prev = T_map_body_gicp;

  // Find closest KF to GICP result
  Eigen::Vector3f gicp_pos = T_map_body_gicp.block<3, 1>(0, 3);
  this->submap_motion_.prev_kf_idx = this->submap_motion_.current_kf_idx;
  this->submap_motion_.current_kf_idx = this->findClosestKF(gicp_pos);

  if (this->debug_)
    RCLCPP_INFO(this->get_logger(),
                "[submap_loc] Stage2: submap=%d fitness=%.4f prob=%.3f pos_err=%.3fm rot_err=%.1fdeg "
                "valid=%d consec=%d/%d total=%d",
                nearest_si, fitness, prob, pos_error, rot_error,
                motion_valid, this->submap_motion_.consecutive_valid,
                this->submap_loc_min_motion_valid_, this->submap_motion_.total_validated);

  // TF Correction (when consecutive_valid >= min_motion_valid)
  if (this->submap_motion_.consecutive_valid >= this->submap_loc_min_motion_valid_)
  {
    Eigen::Matrix4f T_map_odom_new = T_map_body_gicp * T_odom_body.inverse();

    // Compare against system T_map_odom_ for correction distance
    Eigen::Matrix4f T_map_odom_sys;
    {
      std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
      T_map_odom_sys = this->T_map_odom_;
    }
    Eigen::Matrix4f correction_delta = T_map_odom_new * T_map_odom_sys.inverse();
    float correction_dist = correction_delta.block<3, 1>(0, 3).norm();

    if (correction_dist > this->submap_loc_max_correction_)
    {
      RCLCPP_WARN(this->get_logger(),
                  "[submap_loc] Correction too large: %.3fm > %.1fm, rejected",
                  correction_dist, this->submap_loc_max_correction_);
      return;
    }

    // Apply correction to system variable (validated)
    if (!this->enable_global_correction_.load())
    {
      if (this->debug_)
        RCLCPP_INFO(this->get_logger(), "[submap_loc] global correction disabled, skipping TF update");
      return;
    }
    {
      std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
      this->T_map_odom_ = T_map_odom_new;
    }

    // Publish confidence
    float fitness_conf = std::max(0.0f, 1.0f - fitness / static_cast<float>(this->submap_loc_fitness_thresh_));
    float confidence = fitness_conf * prob;
    this->last_confidence_ = confidence;
    if (this->confidence_pub_)
    {
      std_msgs::msg::Float32 msg;
      msg.data = confidence;
      this->confidence_pub_->publish(msg);
    }

    if (this->debug_)
      RCLCPP_INFO(this->get_logger(),
                  "[submap_loc] TF CORRECTED: dist=%.3fm fitness=%.4f conf=%.2f pos=[%.1f,%.1f,%.1f]",
                  correction_dist, fitness, confidence,
                  gicp_pos[0], gicp_pos[1], gicp_pos[2]);

    // Reset consecutive count after applying correction
    this->submap_motion_.consecutive_valid = 0;
  }
}

int dlio::OdomNode::findClosestKF(const Eigen::Vector3f &pos) const
{
  int closest = 0;
  float min_dist_sq = std::numeric_limits<float>::max();
  for (int i = 0; i < static_cast<int>(this->sc_database_.size()); i++)
  {
    float d = (this->sc_database_[i].position - pos).squaredNorm();
    if (d < min_dist_sq)
    {
      min_dist_sq = d;
      closest = i;
    }
  }
  return closest;
}
