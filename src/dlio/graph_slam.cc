/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 *          Azzam Wildan M (SCLC extensions)               *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include "dlio/graph_slam.h"
#include "dlio/utils.h"

#include <filesystem>

dlio::GraphSlamNode::GraphSlamNode(const rclcpp::NodeOptions &options)
    : Node("dlio_graph_slam_node", options)
{

  this->getParams();

  // Subscriber
  this->keyframe_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto keyframe_sub_opt = rclcpp::SubscriptionOptions();
  keyframe_sub_opt.callback_group = this->keyframe_cb_group;
  this->keyframe_sub = this->create_subscription<direct_lidar_inertial_odometry::msg::KeyframeStamped>(
      "keyframe_stamped", 100,
      std::bind(&dlio::GraphSlamNode::callbackKeyframe, this, std::placeholders::_1),
      keyframe_sub_opt);

  // Deskewed scan subscriber (dense scans for SC descriptor computation)
  this->deskewed_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto deskewed_sub_opt = rclcpp::SubscriptionOptions();
  deskewed_sub_opt.callback_group = this->deskewed_cb_group_;
  this->deskewed_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "deskewed", 10,
      std::bind(&dlio::GraphSlamNode::callbackDeskewed, this, std::placeholders::_1),
      deskewed_sub_opt);

  // GPS subscriber
  if (this->gps_enabled_)
  {
    this->gps_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    auto gps_sub_opt = rclcpp::SubscriptionOptions();
    gps_sub_opt.callback_group = this->gps_cb_group_;
    this->gps_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
        this->gps_topic_, 100,
        std::bind(&dlio::GraphSlamNode::callbackGPS, this, std::placeholders::_1),
        gps_sub_opt);
    RCLCPP_INFO(this->get_logger(), "[graph] GPS enabled on topic: %s (radius=%.1fm)",
                this->gps_topic_.c_str(), this->gps_search_radius_);
  }

  // Publishers
  this->corrected_path_pub = this->create_publisher<nav_msgs::msg::Path>("corrected_path", 1);
  this->corrected_map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("corrected_map", 1);
  this->corrected_kf_pose_pub = this->create_publisher<geometry_msgs::msg::PoseArray>("corrected_kf_poses", 10);
  this->loop_closure_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("loop_closures", 1);

  // Service
  this->save_pcd_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->save_pcd_srv = this->create_service<direct_lidar_inertial_odometry::srv::SavePCD>(
      "save_corrected_pcd",
      std::bind(&dlio::GraphSlamNode::savePCD, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default,
      this->save_pcd_cb_group);

  // Timer for loop closure detection
  this->loop_timer = this->create_wall_timer(
      std::chrono::milliseconds(this->loop_detection_period_ms_),
      std::bind(&dlio::GraphSlamNode::searchLoopClosure, this));

  this->optimization_done = false;
  this->last_loop_checked_idx = 0;
  this->shutdown_saved_ = false;

  // Auto-save timer — only in mapping mode
  if (this->map_mode_ == "mapping" && !this->map_path_.empty() && this->auto_save_interval_ > 0.)
  {
    this->auto_save_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(this->auto_save_interval_),
        std::bind(&dlio::GraphSlamNode::autoSave, this));
  }

  RCLCPP_INFO(this->get_logger(),
              "[graph] Initialized (min_gap=%d, sc_thresh=%.2f, sc_range=%.0f, fitness_thresh=%.2f, debug=%s)",
              this->min_keyframe_gap_, this->sc_distance_threshold_, this->sc_max_range_,
              this->threshold_loop_closure_score_, this->debug_ ? "true" : "false");
}

dlio::GraphSlamNode::~GraphSlamNode()
{
  this->saveOnShutdown();
}

void dlio::GraphSlamNode::getParams()
{

  dlio::declare_param(this, "debug/graph_slam", this->debug_, false);
  dlio::declare_param(this, "frames/odom", this->odom_frame, std::string("odom"));
  dlio::declare_param(this, "frames/map", this->map_frame_, std::string("map"));

  dlio::declare_param(this, "loop_detection_period_ms", this->loop_detection_period_ms_, 2000);
  dlio::declare_param(this, "voxel_leaf_size", this->voxel_leaf_size_, 0.2);
  dlio::declare_param(this, "search_submap_num", this->search_submap_num_, 3);

  dlio::declare_param(this, "loop_closure/range", this->range_of_searching_loop_, 15.0);
  dlio::declare_param(this, "loop_closure/min_keyframe_gap", this->min_keyframe_gap_, 50);
  dlio::declare_param(this, "loop_closure/fitness_score_threshold", this->threshold_loop_closure_score_, 0.3);

  std::string reg_method_str;
  dlio::declare_param(this, "registration_method", reg_method_str, std::string("gicp"));
  this->use_gicp_ = (reg_method_str == "gicp");
  dlio::declare_param(this, "ndt/resolution", this->ndt_resolution_, 2.0);
  dlio::declare_param(this, "ndt/num_threads", this->ndt_num_threads_, 4);

  dlio::declare_param(this, "gicp/k_correspondences", this->lc_gicp_k_correspondences_, 16);
  dlio::declare_param(this, "gicp/max_correspondence_distance", this->lc_gicp_max_corr_dist_, 1.0);
  dlio::declare_param(this, "gicp/max_iterations", this->lc_gicp_max_iter_, 64);
  dlio::declare_param(this, "gicp/transformation_epsilon", this->lc_gicp_transformation_ep_, 0.01);
  dlio::declare_param(this, "gicp/rotation_epsilon", this->lc_gicp_rotation_ep_, 0.01);

  dlio::declare_param(this, "pose_graph/num_adjacent_constraints", this->num_adjacent_constraints_, 5);
  dlio::declare_param(this, "pose_graph/optimization_iterations", this->optimization_iterations_, 20);
  dlio::declare_param(this, "pose_graph/odom_edge_info_scale", this->odom_edge_info_scale_, 10.0);
  dlio::declare_param(this, "pose_graph/loop_edge_info_scale", this->loop_edge_info_scale_, 20.0);

  // Map save params (shared with map node)
  dlio::declare_param(this, "map/mode", this->map_mode_, std::string("localization"));
  dlio::declare_param(this, "map/path", this->map_path_, std::string(""));
  if (this->map_path_.empty())
  {
    const char *home = std::getenv("HOME");
    if (home)
      this->map_path_ = std::string(home) + "/.ros/dlio_map.pcd";
  }
  dlio::declare_param(this, "map/voxel_size", this->map_voxel_size_, 0.25);
  dlio::declare_param(this, "map/auto_save_interval", this->auto_save_interval_, 30.0);
  dlio::declare_param(this, "map/publish_interval", this->publish_interval_, 5.0);

  // SC++ loop closure detection params
  double sc_mr = 20.0;
  dlio::declare_param(this, "sc/max_range", sc_mr, 20.0);
  sc_max_range_ = static_cast<float>(sc_mr);
  double sc_dt = 0.5;
  dlio::declare_param(this, "sc/distance_threshold", sc_dt, 0.5);
  sc_distance_threshold_ = static_cast<float>(sc_dt);
  double sc_ght = 0.0;
  dlio::declare_param(this, "sc/ground_height_threshold", sc_ght, 0.0);
  sc_ground_height_threshold_ = static_cast<float>(sc_ght);
  dlio::declare_param(this, "sc/search_window", sc_search_window_, 7);

  // GPS-assisted loop closure params
  dlio::declare_param(this, "gps/enabled", gps_enabled_, false);
  dlio::declare_param(this, "gps/topic", gps_topic_, std::string("/gps/fix"));
  double gps_sr = 50.0;
  dlio::declare_param(this, "gps/search_radius", gps_sr, 50.0);
  gps_search_radius_ = static_cast<float>(gps_sr);
  double gps_ma = 5.0;
  dlio::declare_param(this, "gps/min_accuracy", gps_ma, 5.0);
  gps_min_accuracy_ = static_cast<float>(gps_ma);
}

void dlio::GraphSlamNode::callbackKeyframe(
    const direct_lidar_inertial_odometry::msg::KeyframeStamped::SharedPtr msg)
{

  Keyframe kf;
  kf.id = msg->id;
  kf.timestamp = msg->header.stamp;

  // Extract pose as Eigen::Isometry3d
  Eigen::Quaterniond q(msg->pose.orientation.w, msg->pose.orientation.x,
                       msg->pose.orientation.y, msg->pose.orientation.z);
  Eigen::Vector3d t(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  kf.pose = Eigen::Isometry3d::Identity();
  kf.pose.linear() = q.toRotationMatrix();
  kf.pose.translation() = t;

  // DLIO transforms scans to world frame during deskewing,
  // so the cloud from OdomNode is ALREADY in world frame.
  // Use it directly as cloud_world (do NOT apply pose transform again).
  kf.cloud_world = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(msg->cloud, *kf.cloud_world);

  // Compute approximate body-frame cloud by inverse-transforming.
  // Needed for re-transformation after pose graph optimization.
  kf.cloud_local = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::transformPointCloud(*kf.cloud_world, *kf.cloud_local,
                           kf.pose.inverse().matrix().cast<float>());

  // Compute SC++ descriptor using dense deskewed scan (not sparse keyframe cloud).
  // The deskewed scan has 10-50x more points, giving much better SC fill rate.
  {
    pcl::PointCloud<PointType>::Ptr sc_cloud;
    bool used_deskewed = false;

    // Find closest deskewed scan by timestamp
    {
      std::lock_guard<std::mutex> lock(this->deskewed_buffer_mtx_);
      if (!this->deskewed_buffer_.empty())
      {
        double kf_sec = rclcpp::Time(msg->header.stamp).seconds();
        double best_dt = 1e9;
        int best_idx = -1;
        for (int i = 0; i < static_cast<int>(this->deskewed_buffer_.size()); i++)
        {
          double dt = std::abs(this->deskewed_buffer_[i].timestamp.seconds() - kf_sec);
          if (dt < best_dt)
          {
            best_dt = dt;
            best_idx = i;
          }
        }
        if (best_idx >= 0 && best_dt < 1.0)
        {
          // Deskewed cloud is in odom (world) frame — center at sensor position
          sc_cloud = std::make_shared<pcl::PointCloud<PointType>>(
              *this->deskewed_buffer_[best_idx].cloud);
          used_deskewed = true;
        }
      }
    }

    // Fallback to keyframe cloud if no deskewed scan available
    if (!sc_cloud)
      sc_cloud = std::make_shared<pcl::PointCloud<PointType>>(*kf.cloud_world);

    // Center at sensor position (SC needs range/angle from sensor origin)
    Eigen::Vector3f pos = kf.pose.translation().cast<float>();
    for (auto &pt : sc_cloud->points)
    {
      pt.x -= pos.x();
      pt.y -= pos.y();
      pt.z -= pos.z();
    }

    // Normalize Z to ground level
    float z_min = std::numeric_limits<float>::max();
    for (const auto &pt : sc_cloud->points)
      if (pt.z < z_min)
        z_min = pt.z;
    for (auto &pt : sc_cloud->points)
      pt.z -= z_min;

    // Ground removal for more discriminative descriptors
    if (sc_ground_height_threshold_ > 0.0f)
    {
      auto filtered = std::make_shared<pcl::PointCloud<PointType>>();
      filtered->points.reserve(sc_cloud->points.size());
      for (const auto &pt : sc_cloud->points)
        if (pt.z >= sc_ground_height_threshold_)
          filtered->points.push_back(pt);
      sc_cloud = filtered;
    }

    kf.sc_descriptor = dlio::sc::computeScanContext(sc_cloud, sc_max_range_);
    kf.sector_key = dlio::sc::computeSectorKey(kf.sc_descriptor);

    if (this->debug_)
    {
      int nonzero = 0;
      for (int r = 0; r < dlio::sc::SC_NR; r++)
        for (int s = 0; s < dlio::sc::SC_NS; s++)
          if (kf.sc_descriptor(r, s) != 0.f)
            nonzero++;
      RCLCPP_INFO(this->get_logger(),
                  "[graph] SC kf%u: %s, %zu pts, nonzero=%d/%d (%.1f%%)",
                  kf.id, used_deskewed ? "deskewed" : "keyframe",
                  sc_cloud->points.size(), nonzero,
                  dlio::sc::SC_NR * dlio::sc::SC_NS,
                  100.f * nonzero / (dlio::sc::SC_NR * dlio::sc::SC_NS));
    }
  }

  // Attach GPS if available
  if (this->gps_enabled_)
  {
    double kf_time = rclcpp::Time(msg->header.stamp).seconds();
    GPSMeasurement gps;
    if (this->getGPSAtTime(kf_time, gps))
    {
      kf.gps_valid = this->gpsToLocal(gps.latitude, gps.longitude, gps.altitude,
                                      kf.gps_x, kf.gps_y, kf.gps_z);
    }
  }

  {
    std::lock_guard<std::mutex> lock(this->keyframes_mutex);
    this->keyframes.push_back(kf);
  }

  if (this->debug_)
  {
    RCLCPP_INFO(this->get_logger(), "[graph] Keyframe %u: pos=[%.2f, %.2f, %.2f], %zu pts",
                kf.id, t.x(), t.y(), t.z(), kf.cloud_world->points.size());
  }
}

void dlio::GraphSlamNode::callbackDeskewed(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  DeskewedScan scan;
  scan.timestamp = msg->header.stamp;
  scan.cloud = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*msg, *scan.cloud);

  if (this->debug_)
  {
    static int deskewed_count = 0;
    if (deskewed_count++ % 50 == 0)
    {
      RCLCPP_INFO(this->get_logger(), "[graph] Deskewed raw scan #%d: %zu pts",
                  deskewed_count, scan.cloud->points.size());
    }
  }

  std::lock_guard<std::mutex> lock(this->deskewed_buffer_mtx_);
  if (this->deskewed_buffer_.size() >= DESKEWED_BUFFER_MAX)
    this->deskewed_buffer_.pop_front();
  this->deskewed_buffer_.push_back(std::move(scan));
}

void dlio::GraphSlamNode::callbackGPS(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
  // Reject invalid fix
  if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX)
    return;

  // Check horizontal accuracy from covariance
  float h_acc = std::sqrt(static_cast<float>(msg->position_covariance[0]));
  if (h_acc > this->gps_min_accuracy_ && msg->position_covariance_type != 0)
    return;

  // Initialize origin from first valid fix
  if (!this->gps_origin_set_)
  {
    this->gps_converter_ = std::make_unique<GeographicLib::LocalCartesian>(
        msg->latitude, msg->longitude, msg->altitude);
    this->gps_origin_set_ = true;
    RCLCPP_INFO(this->get_logger(),
                "[graph] GPS origin set: lat=%.8f lon=%.8f alt=%.2f",
                msg->latitude, msg->longitude, msg->altitude);
  }

  double ts = this->now().seconds();
  {
    std::lock_guard<std::mutex> lock(this->gps_buffer_mtx_);
    if (this->gps_buffer_.size() >= GPS_BUFFER_MAX)
      this->gps_buffer_.pop_front();
    this->gps_buffer_.push_back({msg->latitude, msg->longitude, msg->altitude,
                                 ts, h_acc});
  }
}

bool dlio::GraphSlamNode::getGPSAtTime(double timestamp, GPSMeasurement &out)
{
  std::lock_guard<std::mutex> lock(this->gps_buffer_mtx_);
  if (this->gps_buffer_.empty())
    return false;

  double best_dt = 1e9;
  int best_idx = -1;
  for (int i = 0; i < static_cast<int>(this->gps_buffer_.size()); i++)
  {
    double dt = std::abs(this->gps_buffer_[i].timestamp - timestamp);
    if (dt < best_dt)
    {
      best_dt = dt;
      best_idx = i;
    }
  }
  if (best_idx < 0 || best_dt > 1.0)
    return false;
  out = this->gps_buffer_[best_idx];
  return true;
}

bool dlio::GraphSlamNode::gpsToLocal(double lat, double lon, double alt,
                                     float &x, float &y, float &z)
{
  if (!this->gps_origin_set_ || !this->gps_converter_)
    return false;
  double dx, dy, dz;
  this->gps_converter_->Forward(lat, lon, alt, dx, dy, dz);
  x = static_cast<float>(dx);
  y = static_cast<float>(dy);
  z = static_cast<float>(dz);
  return true;
}

void dlio::GraphSlamNode::searchLoopClosure()
{

  // Skip in localization mode — prior map is already optimized
  if (this->map_mode_ != "mapping")
    return;

  // Copy keyframes snapshot to avoid holding mutex during GICP
  std::vector<Keyframe> kf_snapshot;
  int num_keyframes;
  {
    std::lock_guard<std::mutex> lock(this->keyframes_mutex);
    num_keyframes = static_cast<int>(this->keyframes.size());
    if (num_keyframes < this->min_keyframe_gap_ + 1)
      return;
    kf_snapshot = this->keyframes;
  }

  int start_idx = std::max(this->last_loop_checked_idx, this->min_keyframe_gap_);
  if (start_idx >= num_keyframes)
  {
    // Nothing new to check
    return;
  }
  if (this->debug_)
  {
    RCLCPP_INFO(this->get_logger(),
                "[graph] Loop search: checking kf %d-%d (total=%d, gap=%d)",
                start_idx, num_keyframes - 1, num_keyframes, this->min_keyframe_gap_);
  }

  bool loop_found = false;

  for (int current_idx = std::max(this->last_loop_checked_idx, this->min_keyframe_gap_);
       current_idx < num_keyframes; ++current_idx)
  {

    int candidate_idx = -1;
    double candidate_dist = 0.0;
    int sc_shift = 0;
    if (this->detectLoopCandidate(kf_snapshot, current_idx, candidate_idx, candidate_dist, sc_shift))
    {
      RCLCPP_INFO(this->get_logger(),
                  "[graph] Loop candidate: kf %d <-> kf %d (sc_dist=%.4f, shift=%d)",
                  current_idx, candidate_idx, candidate_dist, sc_shift);

      Eigen::Isometry3d relative_pose;
      double fitness_score;
      if (this->performLoopRegistration(kf_snapshot, current_idx, candidate_idx,
                                        sc_shift, relative_pose, fitness_score))
      {
        RCLCPP_INFO(this->get_logger(),
                    "[graph] Loop closure CONFIRMED: %d <-> %d, fitness=%.4f",
                    candidate_idx, current_idx, fitness_score);

        LoopEdge edge;
        edge.from_idx = candidate_idx;
        edge.to_idx = current_idx;
        edge.relative_pose = relative_pose;
        edge.information = Eigen::Matrix<double, 6, 6>::Identity() * this->loop_edge_info_scale_;
        this->loop_edges.push_back(edge);
        loop_found = true;
      }
      else if (this->debug_)
      {
        RCLCPP_INFO(this->get_logger(),
                    "[graph] Loop candidate REJECTED: %d <-> %d, fitness=%.4f (thresh=%.2f)",
                    candidate_idx, current_idx, fitness_score,
                    this->threshold_loop_closure_score_);
      }
    }
  }

  this->last_loop_checked_idx = num_keyframes;

  if (loop_found)
  {
    // Brief lock: snapshot keyframes, then run heavy work outside
    std::vector<Keyframe> kf_opt;
    {
      std::lock_guard<std::mutex> lock(this->keyframes_mutex);
      kf_opt = this->keyframes;
    }
    this->optimizePoseGraph(kf_opt);
    this->publishLoopClosureMarkers(kf_opt);
  }

  // Always publish path/map/poses (uses corrected poses if available, raw otherwise)
  this->publishCorrectedData(kf_snapshot);
}

bool dlio::GraphSlamNode::detectLoopCandidate(
    const std::vector<Keyframe> &kfs, int current_idx,
    int &candidate_idx, double &candidate_dist, int &sc_shift)
{

  candidate_idx = -1;
  candidate_dist = std::numeric_limits<double>::max();
  sc_shift = 0;

  const auto &current_kf = kfs[current_idx];
  bool current_has_gps = this->gps_enabled_ && current_kf.gps_valid;

  // ---- Strategy 1: GPS proximity search (like continuous localization) ----
  if (current_has_gps)
  {
    float best_gps_dist = std::numeric_limits<float>::max();
    int gps_candidate = -1;
    int n_gps_candidates = 0;

    for (int i = 0; i < current_idx - this->min_keyframe_gap_; ++i)
    {
      if (!kfs[i].gps_valid)
        continue;

      float dx = current_kf.gps_x - kfs[i].gps_x;
      float dy = current_kf.gps_y - kfs[i].gps_y;
      float gps_dist = std::sqrt(dx * dx + dy * dy);

      if (gps_dist < this->gps_search_radius_ && gps_dist < best_gps_dist)
      {
        best_gps_dist = gps_dist;
        gps_candidate = i;
        n_gps_candidates++;
      }
    }

    if (gps_candidate >= 0)
    {
      candidate_idx = gps_candidate;
      candidate_dist = static_cast<double>(best_gps_dist);
      sc_shift = 0; // no SC shift needed for GPS match

      RCLCPP_INFO(this->get_logger(),
                  "[graph] GPS loop candidate: kf%d <-> kf%d (gps_dist=%.2fm, %d within radius)",
                  current_idx, gps_candidate, best_gps_dist, n_gps_candidates);
      return true;
    }

    // GPS had no candidates within radius — fall through to SC++
    if (this->debug_)
    {
      RCLCPP_INFO(this->get_logger(),
                  "[graph] GPS kf%d: no candidates within %.1fm radius, falling back to SC++",
                  current_idx, this->gps_search_radius_);
    }
  }

  // ---- Strategy 2: SC++ descriptor matching across ALL keyframes (fallback) ----
  const auto &current_desc = current_kf.sc_descriptor;

  float best_dist_overall = std::numeric_limits<float>::max();
  int best_idx_overall = -1;
  int n_compared = 0;

  for (int i = 0; i < current_idx - this->min_keyframe_gap_; ++i)
  {
    auto [dist, shift] = dlio::sc::computeScanContextDistance(
        current_desc, kfs[i].sc_descriptor);

    n_compared++;
    if (dist < best_dist_overall)
    {
      best_dist_overall = dist;
      best_idx_overall = i;
    }

    if (dist < this->sc_distance_threshold_ && dist < candidate_dist)
    {
      candidate_dist = dist;
      candidate_idx = i;
      sc_shift = shift;
    }
  }

  if (candidate_idx < 0 && this->debug_)
  {
    RCLCPP_INFO(this->get_logger(),
                "[graph] SC kf%d: no match (best_dist=%.4f vs thresh=%.2f, best_kf=%d, compared=%d)",
                current_idx, best_dist_overall, this->sc_distance_threshold_,
                best_idx_overall, n_compared);
  }

  return (candidate_idx >= 0);
}

bool dlio::GraphSlamNode::performLoopRegistration(
    const std::vector<Keyframe> &kfs,
    int current_idx, int candidate_idx, int sc_shift,
    Eigen::Isometry3d &relative_pose, double &fitness_score)
{

  // Standard loop closure ICP:
  //   Target = submap around candidate, each cloud transformed to world by its OWN pose
  //   Source = current keyframe body-frame cloud (untransformed)
  //   Init guess = candidate's pose (SC/GPS says current is near candidate)
  //   ICP output = corrected world pose of current keyframe
  //   relative_pose = candidate.pose.inverse() * corrected_pose

  // --- Build target submap in world frame ---
  pcl::PointCloud<PointType>::Ptr target_cloud = std::make_shared<pcl::PointCloud<PointType>>();
  for (int j = -this->search_submap_num_; j <= this->search_submap_num_; ++j)
  {
    int idx = candidate_idx + j;
    if (idx < 0 || idx >= static_cast<int>(kfs.size()))
      continue;
    pcl::PointCloud<PointType> tmp;
    pcl::transformPointCloud(*kfs[idx].cloud_local, tmp, kfs[idx].pose.matrix().cast<float>());
    *target_cloud += tmp;
  }

  // Voxel filter target
  pcl::VoxelGrid<PointType> voxel;
  voxel.setLeafSize(this->voxel_leaf_size_, this->voxel_leaf_size_, this->voxel_leaf_size_);
  voxel.setInputCloud(target_cloud);
  pcl::PointCloud<PointType>::Ptr filtered_target = std::make_shared<pcl::PointCloud<PointType>>();
  voxel.filter(*filtered_target);

  if (filtered_target->empty() || kfs[current_idx].cloud_local->empty())
  {
    fitness_score = std::numeric_limits<double>::max();
    return false;
  }

  // --- Source: body-frame cloud (untransformed) ---
  pcl::PointCloud<PointType>::Ptr source_cloud = kfs[current_idx].cloud_local;

  // --- Initial guess: candidate's pose + SC yaw correction ---
  // SC/GPS detected that current is at a similar place as candidate.
  // Use candidate's pose as init guess so ICP starts near the correct position.
  Eigen::Matrix4f init_guess = kfs[candidate_idx].pose.matrix().cast<float>();

  // Apply SC shift as yaw correction: each sector = 360/NS degrees
  if (sc_shift != 0)
  {
    float yaw_offset = static_cast<float>(sc_shift) * 2.0f * static_cast<float>(M_PI) / static_cast<float>(dlio::sc::SC_NS);
    Eigen::AngleAxisf yaw_rot(yaw_offset, Eigen::Vector3f::UnitZ());
    init_guess.block<3, 3>(0, 0) = init_guess.block<3, 3>(0, 0) * yaw_rot.toRotationMatrix();
  }

  if (this->debug_)
  {
    RCLCPP_INFO(this->get_logger(),
                "[graph] ICP: source=%zu pts (body), target=%zu pts (world), sc_shift=%d",
                source_cloud->points.size(), filtered_target->points.size(), sc_shift);
  }

  // --- ICP: align body-frame source to world-frame target ---
  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  bool converged = false;
  Eigen::Matrix4f final_T;

  if (this->use_gicp_)
  {
    nano_gicp::NanoGICP<PointType, PointType> gicp;
    gicp.setCorrespondenceRandomness(this->lc_gicp_k_correspondences_);
    gicp.setMaxCorrespondenceDistance(this->lc_gicp_max_corr_dist_);
    gicp.setMaximumIterations(this->lc_gicp_max_iter_);
    gicp.setTransformationEpsilon(this->lc_gicp_transformation_ep_);
    gicp.setRotationEpsilon(this->lc_gicp_rotation_ep_);

    gicp.setInputSource(source_cloud);
    gicp.setInputTarget(filtered_target);
    gicp.align(*aligned, init_guess);

    converged = gicp.hasConverged();
    fitness_score = gicp.getFitnessScore();
    final_T = gicp.getFinalTransformation();
  }
  else
  {
    pclomp::NormalDistributionsTransform<PointType, PointType> ndt_lc;
    ndt_lc.setResolution(this->ndt_resolution_);
    ndt_lc.setNumThreads(this->ndt_num_threads_);
    ndt_lc.setNeighborhoodSearchMethod(pclomp::DIRECT7);
    ndt_lc.setMaximumIterations(this->lc_gicp_max_iter_);
    ndt_lc.setTransformationEpsilon(this->lc_gicp_transformation_ep_);

    ndt_lc.setInputSource(source_cloud);
    ndt_lc.setInputTarget(filtered_target);
    ndt_lc.align(*aligned, init_guess);

    converged = ndt_lc.hasConverged();
    fitness_score = ndt_lc.getFitnessScore();
    final_T = ndt_lc.getFinalTransformation();
  }

  if (!converged || fitness_score > this->threshold_loop_closure_score_)
  {
    return false;
  }

  // final_T = corrected world pose of current keyframe (body → world)
  // relative_pose = between-factor for g2o edge (candidate → current)
  relative_pose = kfs[candidate_idx].pose.inverse() * Eigen::Isometry3d(final_T.cast<double>());

  return true;
}

void dlio::GraphSlamNode::optimizePoseGraph(const std::vector<Keyframe> &kf_snap)
{

  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);

  auto linear_solver = std::make_unique<
      g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
  auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
  auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
  optimizer.setAlgorithm(solver);

  int num_kf = static_cast<int>(kf_snap.size());
  Eigen::Matrix<double, 6, 6> odom_info =
      Eigen::Matrix<double, 6, 6>::Identity() * this->odom_edge_info_scale_;

  // Add vertices
  for (int i = 0; i < num_kf; ++i)
  {
    g2o::VertexSE3 *vertex = new g2o::VertexSE3();
    vertex->setId(i);
    vertex->setEstimate(kf_snap[i].pose);
    if (i == 0)
      vertex->setFixed(true);
    optimizer.addVertex(vertex);
  }

  // Add sequential (odometry) edges
  for (int i = 1; i < num_kf; ++i)
  {
    int start = std::max(0, i - this->num_adjacent_constraints_);
    for (int j = start; j < i; ++j)
    {
      Eigen::Isometry3d relative = kf_snap[j].pose.inverse() * kf_snap[i].pose;

      g2o::EdgeSE3 *edge = new g2o::EdgeSE3();
      edge->setMeasurement(relative);
      edge->setInformation(odom_info);
      edge->vertices()[0] = optimizer.vertex(j);
      edge->vertices()[1] = optimizer.vertex(i);
      optimizer.addEdge(edge);
    }
  }

  // Add loop closure edges
  for (const auto &loop : this->loop_edges)
  {
    g2o::EdgeSE3 *edge = new g2o::EdgeSE3();
    edge->setMeasurement(loop.relative_pose);
    edge->setInformation(loop.information);
    edge->vertices()[0] = optimizer.vertex(loop.from_idx);
    edge->vertices()[1] = optimizer.vertex(loop.to_idx);
    optimizer.addEdge(edge);
  }

  // Optimize
  optimizer.initializeOptimization();
  optimizer.optimize(this->optimization_iterations_);

  // Brief lock: store corrected poses
  {
    std::lock_guard<std::mutex> lock(this->corrected_mutex);
    this->corrected_poses.resize(num_kf);
    for (int i = 0; i < num_kf; ++i)
    {
      g2o::VertexSE3 *v = static_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
      this->corrected_poses[i] = v->estimate();
    }
    this->optimization_done = true;
  }

  RCLCPP_INFO(this->get_logger(), "[graph] Pose graph optimized: %d vertices, %d loop edges",
              num_kf, static_cast<int>(this->loop_edges.size()));
}

void dlio::GraphSlamNode::publishCorrectedData(const std::vector<Keyframe> &kf_snap)
{

  // Use corrected poses if available, otherwise fall back to raw keyframe poses
  IsometryVec poses_snap;
  {
    std::lock_guard<std::mutex> lock(this->corrected_mutex);
    if (this->optimization_done)
      poses_snap = this->corrected_poses;
  }

  int num_stored = static_cast<int>(kf_snap.size());
  if (poses_snap.empty())
  {
    // No optimization yet — use raw odom poses
    poses_snap.reserve(num_stored);
    for (int i = 0; i < num_stored; ++i)
      poses_snap.push_back(kf_snap[i].pose);
  }

  int num_kf = static_cast<int>(poses_snap.size());

  // Use map frame for corrected data (graph-optimized poses are in map frame)
  std::string frame = this->map_frame_;

  // Corrected path
  nav_msgs::msg::Path path;
  path.header.stamp = this->now();
  path.header.frame_id = frame;

  // Corrected keyframe poses
  geometry_msgs::msg::PoseArray kf_poses;
  kf_poses.header = path.header;

  for (int i = 0; i < num_kf && i < num_stored; ++i)
  {
    const Eigen::Isometry3d &pose = poses_snap[i];

    // Path
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = kf_snap[i].timestamp;
    ps.header.frame_id = frame;
    Eigen::Quaterniond q(pose.rotation());
    ps.pose.position.x = pose.translation().x();
    ps.pose.position.y = pose.translation().y();
    ps.pose.position.z = pose.translation().z();
    ps.pose.orientation.w = q.w();
    ps.pose.orientation.x = q.x();
    ps.pose.orientation.y = q.y();
    ps.pose.orientation.z = q.z();
    path.poses.push_back(ps);
    kf_poses.poses.push_back(ps.pose);
  }

  this->corrected_path_pub->publish(path);
  this->corrected_kf_pose_pub->publish(kf_poses);

  // Throttle corrected map publish (expensive: transforms all keyframe clouds)
  static rclcpp::Time last_map_pub_time(0, 0, RCL_ROS_TIME);
  rclcpp::Time now = this->now();
  if ((now - last_map_pub_time).seconds() >= this->publish_interval_)
  {
    pcl::PointCloud<PointType>::Ptr corrected_map = std::make_shared<pcl::PointCloud<PointType>>();
    for (int i = 0; i < num_kf && i < num_stored; ++i)
    {
      pcl::PointCloud<PointType>::Ptr transformed = std::make_shared<pcl::PointCloud<PointType>>();
      pcl::transformPointCloud(*kf_snap[i].cloud_local,
                               *transformed, poses_snap[i].matrix().cast<float>());
      *corrected_map += *transformed;
    }

    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(this->voxel_leaf_size_, this->voxel_leaf_size_, this->voxel_leaf_size_);
    voxel.setInputCloud(corrected_map);
    voxel.filter(*corrected_map);

    sensor_msgs::msg::PointCloud2 map_ros;
    pcl::toROSMsg(*corrected_map, map_ros);
    map_ros.header.stamp = now;
    map_ros.header.frame_id = frame;
    this->corrected_map_pub->publish(map_ros);

    last_map_pub_time = now;
  }
}

void dlio::GraphSlamNode::publishLoopClosureMarkers(const std::vector<Keyframe> &kf_snap)
{

  visualization_msgs::msg::MarkerArray markers;

  for (size_t i = 0; i < this->loop_edges.size(); ++i)
  {
    const auto &edge = this->loop_edges[i];

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = this->map_frame_;
    marker.header.stamp = this->now();
    marker.ns = "loop_closures";
    marker.id = static_cast<int>(i);
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.1;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    geometry_msgs::msg::Point p1, p2;
    const auto &pose_from = kf_snap[edge.from_idx].pose;
    const auto &pose_to = kf_snap[edge.to_idx].pose;
    p1.x = pose_from.translation().x();
    p1.y = pose_from.translation().y();
    p1.z = pose_from.translation().z();
    p2.x = pose_to.translation().x();
    p2.y = pose_to.translation().y();
    p2.z = pose_to.translation().z();
    marker.points.push_back(p1);
    marker.points.push_back(p2);

    markers.markers.push_back(marker);
  }

  this->loop_closure_pub->publish(markers);
}

void dlio::GraphSlamNode::saveGraphMaps(const std::string &save_dir, float leaf_size,
                                        const std::vector<Keyframe> &kf_snap,
                                        const IsometryVec &poses_snap, bool opt_done)
{

  int num_kf = static_cast<int>(kf_snap.size());
  if (num_kf == 0)
    return;

  std::filesystem::create_directories(save_dir);

  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);

  // Process in chunks to limit peak memory usage.
  constexpr int CHUNK_SIZE = 50;
  pcl::PointCloud<PointType>::Ptr result = std::make_shared<pcl::PointCloud<PointType>>();

  for (int start = 0; start < num_kf; start += CHUNK_SIZE)
  {
    int end = std::min(start + CHUNK_SIZE, num_kf);

    size_t chunk_pts = 0;
    for (int i = start; i < end; ++i)
      chunk_pts += kf_snap[i].cloud_local->points.size();

    pcl::PointCloud<PointType>::Ptr chunk = std::make_shared<pcl::PointCloud<PointType>>();
    chunk->points.reserve(chunk_pts);

    for (int i = start; i < end; ++i)
    {
      Eigen::Matrix4f T;
      if (opt_done && i < static_cast<int>(poses_snap.size()))
        T = poses_snap[i].matrix().cast<float>();
      else
        T = kf_snap[i].pose.matrix().cast<float>();

      pcl::PointCloud<PointType> tmp;
      pcl::transformPointCloud(*kf_snap[i].cloud_local, tmp, T);
      chunk->points.insert(chunk->points.end(), tmp.points.begin(), tmp.points.end());
    }
    chunk->width = chunk->points.size();
    chunk->height = 1;

    vg.setInputCloud(chunk);
    vg.filter(*chunk);

    result->points.insert(result->points.end(), chunk->points.begin(), chunk->points.end());
  }

  result->width = result->points.size();
  result->height = 1;

  vg.setInputCloud(result);
  vg.filter(*result);

  if (result->points.empty())
  {
    std::cout << "[graph] saveGraphMaps: corrected cloud is empty, skipping" << std::endl;
    return;
  }

  std::filesystem::path map_fp(this->map_path_);
  std::string stem = map_fp.stem().string();
  if (stem.empty())
    stem = "dlio_map";
  std::string corr_file = save_dir + "/" + stem + "_corrected.pcd";
  int ret = pcl::io::savePCDFileBinary(corr_file, *result);
  if (ret == 0)
  {
    std::cout << "[graph] Saved corrected map: " << result->points.size() << " pts -> " << corr_file
              << (opt_done ? " (graph-optimized)" : " (no loop closures, same as raw)")
              << std::endl;
  }
  else
  {
    std::cerr << "[graph] FAILED to save corrected map: " << corr_file << std::endl;
  }
}

void dlio::GraphSlamNode::autoSave()
{
  // Brief lock: copy keyframes
  std::vector<Keyframe> kf_snap;
  {
    std::lock_guard<std::mutex> lock(this->keyframes_mutex);
    if (this->keyframes.empty())
      return;
    kf_snap = this->keyframes;
  }

  // Brief lock: copy corrected poses
  IsometryVec poses_snap;
  bool opt_done;
  {
    std::lock_guard<std::mutex> lock(this->corrected_mutex);
    poses_snap = this->corrected_poses;
    opt_done = this->optimization_done;
  }

  std::filesystem::path map_fp(this->map_path_);
  std::string save_dir = map_fp.parent_path().string();
  if (save_dir.empty())
    save_dir = ".";

  // Heavy save work runs on local copies — no locks held
  this->saveGraphMaps(save_dir, static_cast<float>(this->map_voxel_size_),
                      kf_snap, poses_snap, opt_done);
}

void dlio::GraphSlamNode::saveOnShutdown()
{
  if (this->map_mode_ != "mapping")
    return;
  if (this->map_path_.empty())
    return;
  if (this->shutdown_saved_.exchange(true))
    return; // already saved

  // Brief lock: copy keyframes
  std::vector<Keyframe> kf_snap;
  {
    std::lock_guard<std::mutex> lock(this->keyframes_mutex);
    if (this->keyframes.empty())
      return;
    kf_snap = this->keyframes;
  }

  // Brief lock: copy corrected poses
  IsometryVec poses_snap;
  bool opt_done;
  {
    std::lock_guard<std::mutex> lock(this->corrected_mutex);
    poses_snap = this->corrected_poses;
    opt_done = this->optimization_done;
  }

  std::filesystem::path map_fp(this->map_path_);
  std::string save_dir = map_fp.parent_path().string();
  if (save_dir.empty())
    save_dir = ".";

  this->saveGraphMaps(save_dir, static_cast<float>(this->map_voxel_size_),
                      kf_snap, poses_snap, opt_done);
}

void dlio::GraphSlamNode::savePCD(
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res)
{

  // Brief lock: copy keyframes
  std::vector<Keyframe> kf_snap;
  {
    std::lock_guard<std::mutex> lock(this->keyframes_mutex);
    if (this->keyframes.empty())
    {
      RCLCPP_WARN(this->get_logger(), "[graph] No keyframes to save");
      res->success = false;
      return;
    }
    kf_snap = this->keyframes;
  }

  // Brief lock: copy corrected poses
  IsometryVec poses_snap;
  bool opt_done;
  {
    std::lock_guard<std::mutex> lock(this->corrected_mutex);
    poses_snap = this->corrected_poses;
    opt_done = this->optimization_done;
  }

  // Heavy save work runs on local copies
  this->saveGraphMaps(req->save_path, req->leaf_size,
                      kf_snap, poses_snap, opt_done);
  res->success = true;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(dlio::GraphSlamNode)
