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

dlio::GraphSlamNode::GraphSlamNode(const rclcpp::NodeOptions& options)
: Node("dlio_graph_slam_node", options) {

  this->getParams();

  // Subscriber
  this->keyframe_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto keyframe_sub_opt = rclcpp::SubscriptionOptions();
  keyframe_sub_opt.callback_group = this->keyframe_cb_group;
  this->keyframe_sub = this->create_subscription<direct_lidar_inertial_odometry::msg::KeyframeStamped>(
    "keyframe_stamped", 100,
    std::bind(&dlio::GraphSlamNode::callbackKeyframe, this, std::placeholders::_1),
    keyframe_sub_opt);

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
  if (this->map_mode_ == "mapping" && !this->map_path_.empty() && this->auto_save_interval_ > 0.) {
    this->auto_save_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(this->auto_save_interval_),
        std::bind(&dlio::GraphSlamNode::autoSave, this));
  }

  RCLCPP_INFO(this->get_logger(), "[graph] Initialized (min_gap=%d, range=%.1f, fitness_thresh=%.2f, debug=%s)",
              this->min_keyframe_gap_, this->range_of_searching_loop_, this->threshold_loop_closure_score_,
              this->debug_ ? "true" : "false");
}

dlio::GraphSlamNode::~GraphSlamNode() {
  this->saveOnShutdown();
}

void dlio::GraphSlamNode::getParams() {

  dlio::declare_param(this, "debug/graph_slam", this->debug_, false);
  dlio::declare_param(this, "frames/odom", this->odom_frame, std::string("odom"));

  dlio::declare_param(this, "loop_detection_period_ms", this->loop_detection_period_ms_, 2000);
  dlio::declare_param(this, "voxel_leaf_size", this->voxel_leaf_size_, 0.2);
  dlio::declare_param(this, "search_submap_num", this->search_submap_num_, 3);

  dlio::declare_param(this, "loop_closure/range", this->range_of_searching_loop_, 15.0);
  dlio::declare_param(this, "loop_closure/min_keyframe_gap", this->min_keyframe_gap_, 50);
  dlio::declare_param(this, "loop_closure/fitness_score_threshold", this->threshold_loop_closure_score_, 0.3);

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
  if (this->map_path_.empty()) {
    const char* home = std::getenv("HOME");
    if (home)
      this->map_path_ = std::string(home) + "/.ros/dlio_map.pcd";
  }
  dlio::declare_param(this, "map/voxel_size", this->map_voxel_size_, 0.25);
  dlio::declare_param(this, "map/auto_save_interval", this->auto_save_interval_, 30.0);
}

void dlio::GraphSlamNode::callbackKeyframe(
    const direct_lidar_inertial_odometry::msg::KeyframeStamped::SharedPtr msg) {

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

  {
    std::lock_guard<std::mutex> lock(this->keyframes_mutex);
    this->keyframes.push_back(kf);
  }

  if (this->debug_) {
    RCLCPP_INFO(this->get_logger(), "[graph] Keyframe %u: pos=[%.2f, %.2f, %.2f], %zu pts",
                kf.id, t.x(), t.y(), t.z(), kf.cloud_world->points.size());
  }
}

void dlio::GraphSlamNode::searchLoopClosure() {

  // Copy keyframes snapshot to avoid holding mutex during GICP
  std::vector<Keyframe> kf_snapshot;
  int num_keyframes;
  {
    std::lock_guard<std::mutex> lock(this->keyframes_mutex);
    num_keyframes = static_cast<int>(this->keyframes.size());
    if (num_keyframes < this->min_keyframe_gap_ + 1) return;
    kf_snapshot = this->keyframes;
  }

  bool loop_found = false;

  for (int current_idx = std::max(this->last_loop_checked_idx, this->min_keyframe_gap_);
       current_idx < num_keyframes; ++current_idx) {

    int candidate_idx = -1;
    double candidate_dist = 0.0;
    if (this->detectLoopCandidate(kf_snapshot, current_idx, candidate_idx, candidate_dist)) {
      if (this->debug_) {
        RCLCPP_INFO(this->get_logger(),
          "[graph] Loop candidate: kf %d <-> kf %d (dist=%.2fm)", current_idx, candidate_idx, candidate_dist);
      }

      Eigen::Isometry3d relative_pose;
      double fitness_score;
      if (this->performLoopRegistration(kf_snapshot, current_idx, candidate_idx,
                                         relative_pose, fitness_score)) {
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
      } else {
        if (this->debug_) {
          RCLCPP_INFO(this->get_logger(),
            "[graph] Loop candidate REJECTED: %d <-> %d, fitness=%.4f (thresh=%.2f)",
            candidate_idx, current_idx, fitness_score,
            this->threshold_loop_closure_score_);
        }
      }
    }
  }

  this->last_loop_checked_idx = num_keyframes;

  if (loop_found) {
    // Re-lock to read latest keyframes for optimization
    std::lock_guard<std::mutex> lock(this->keyframes_mutex);
    this->optimizePoseGraph();
    this->publishCorrectedData();
    this->publishLoopClosureMarkers();
  }
}

bool dlio::GraphSlamNode::detectLoopCandidate(
    const std::vector<Keyframe>& kfs, int current_idx, int& candidate_idx, double& candidate_dist) {

  const Eigen::Vector3d& current_pos = kfs[current_idx].pose.translation();
  double min_dist = std::numeric_limits<double>::max();
  candidate_idx = -1;

  for (int i = 0; i < current_idx - this->min_keyframe_gap_; ++i) {
    double dist = (current_pos - kfs[i].pose.translation()).norm();

    if (dist < this->range_of_searching_loop_ && dist < min_dist) {
      min_dist = dist;
      candidate_idx = i;
    }
  }

  candidate_dist = min_dist;
  return (candidate_idx >= 0);
}

bool dlio::GraphSlamNode::performLoopRegistration(
    const std::vector<Keyframe>& kfs,
    int current_idx, int candidate_idx,
    Eigen::Isometry3d& relative_pose, double& fitness_score) {

  // Build target submap from candidate and neighbors (world frame)
  pcl::PointCloud<PointType>::Ptr target_cloud = std::make_shared<pcl::PointCloud<PointType>>();

  for (int j = -this->search_submap_num_; j <= this->search_submap_num_; ++j) {
    int idx = candidate_idx + j;
    if (idx < 0 || idx >= static_cast<int>(kfs.size())) continue;
    *target_cloud += *(kfs[idx].cloud_world);
  }

  // Voxel filter target
  pcl::VoxelGrid<PointType> voxel;
  voxel.setLeafSize(this->voxel_leaf_size_, this->voxel_leaf_size_, this->voxel_leaf_size_);
  voxel.setInputCloud(target_cloud);
  pcl::PointCloud<PointType>::Ptr filtered_target = std::make_shared<pcl::PointCloud<PointType>>();
  voxel.filter(*filtered_target);

  // Source: current keyframe cloud in world frame
  // DLIO scans are already in world frame, so both source and target are world-frame.
  pcl::PointCloud<PointType>::Ptr source_cloud = kfs[current_idx].cloud_world;

  if (filtered_target->empty() || source_cloud->empty()) {
    fitness_score = std::numeric_limits<double>::max();
    return false;
  }

  if (this->debug_) {
    RCLCPP_INFO(this->get_logger(), "[graph] GICP: source=%zu pts, target=%zu pts (both world frame)",
                source_cloud->points.size(), filtered_target->points.size());
  }

  // Create fresh GICP instance for this registration
  nano_gicp::NanoGICP<PointType, PointType> gicp;
  gicp.setCorrespondenceRandomness(this->lc_gicp_k_correspondences_);
  gicp.setMaxCorrespondenceDistance(this->lc_gicp_max_corr_dist_);
  gicp.setMaximumIterations(this->lc_gicp_max_iter_);
  gicp.setTransformationEpsilon(this->lc_gicp_transformation_ep_);
  gicp.setRotationEpsilon(this->lc_gicp_rotation_ep_);

  gicp.setInputSource(source_cloud);
  gicp.setInputTarget(filtered_target);

  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  gicp.align(*aligned);

  fitness_score = gicp.getFitnessScore();

  if (!gicp.hasConverged() || fitness_score > this->threshold_loop_closure_score_) {
    return false;
  }

  // GICP gives T_correction that aligns source_world to target_world.
  // T_correction * current_world_cloud ≈ candidate_world_cloud
  // The relative pose between candidate and current in their respective body frames:
  // relative = candidate_pose^-1 * T_correction * current_pose
  Eigen::Isometry3d T_correction = Eigen::Isometry3d(gicp.getFinalTransformation().cast<double>());
  Eigen::Isometry3d current_pose = kfs[current_idx].pose;
  Eigen::Isometry3d candidate_pose = kfs[candidate_idx].pose;

  relative_pose = candidate_pose.inverse() * T_correction * current_pose;

  return true;
}

void dlio::GraphSlamNode::optimizePoseGraph() {

  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);

  auto linear_solver = std::make_unique<
      g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
  auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
  auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
  optimizer.setAlgorithm(solver);

  int num_kf = static_cast<int>(this->keyframes.size());
  Eigen::Matrix<double, 6, 6> odom_info =
      Eigen::Matrix<double, 6, 6>::Identity() * this->odom_edge_info_scale_;

  // Add vertices
  for (int i = 0; i < num_kf; ++i) {
    g2o::VertexSE3* vertex = new g2o::VertexSE3();
    vertex->setId(i);
    vertex->setEstimate(this->keyframes[i].pose);
    if (i == 0) vertex->setFixed(true);
    optimizer.addVertex(vertex);
  }

  // Add sequential (odometry) edges
  for (int i = 1; i < num_kf; ++i) {
    int start = std::max(0, i - this->num_adjacent_constraints_);
    for (int j = start; j < i; ++j) {
      Eigen::Isometry3d relative = this->keyframes[j].pose.inverse() * this->keyframes[i].pose;

      g2o::EdgeSE3* edge = new g2o::EdgeSE3();
      edge->setMeasurement(relative);
      edge->setInformation(odom_info);
      edge->vertices()[0] = optimizer.vertex(j);
      edge->vertices()[1] = optimizer.vertex(i);
      optimizer.addEdge(edge);
    }
  }

  // Add loop closure edges
  for (const auto& loop : this->loop_edges) {
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    edge->setMeasurement(loop.relative_pose);
    edge->setInformation(loop.information);
    edge->vertices()[0] = optimizer.vertex(loop.from_idx);
    edge->vertices()[1] = optimizer.vertex(loop.to_idx);
    optimizer.addEdge(edge);
  }

  // Optimize
  optimizer.initializeOptimization();
  optimizer.optimize(this->optimization_iterations_);

  // Extract corrected poses
  std::lock_guard<std::mutex> lock(this->corrected_mutex);
  this->corrected_poses.resize(num_kf);
  for (int i = 0; i < num_kf; ++i) {
    g2o::VertexSE3* v = static_cast<g2o::VertexSE3*>(optimizer.vertex(i));
    this->corrected_poses[i] = v->estimate();
  }
  this->optimization_done = true;

  RCLCPP_INFO(this->get_logger(), "[graph] Pose graph optimized: %d vertices, %d loop edges",
              num_kf, static_cast<int>(this->loop_edges.size()));
}

void dlio::GraphSlamNode::publishCorrectedData() {

  std::lock_guard<std::mutex> lock(this->corrected_mutex);
  if (!this->optimization_done) return;

  int num_kf = static_cast<int>(this->corrected_poses.size());
  int num_stored = static_cast<int>(this->keyframes.size());

  // Corrected path
  nav_msgs::msg::Path path;
  path.header.stamp = this->now();
  path.header.frame_id = this->odom_frame;

  // Corrected keyframe poses
  geometry_msgs::msg::PoseArray kf_poses;
  kf_poses.header = path.header;

  // Corrected map
  pcl::PointCloud<PointType>::Ptr corrected_map = std::make_shared<pcl::PointCloud<PointType>>();

  for (int i = 0; i < num_kf && i < num_stored; ++i) {
    const Eigen::Isometry3d& pose = this->corrected_poses[i];

    // Path
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = this->keyframes[i].timestamp;
    ps.header.frame_id = this->odom_frame;
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

    // Transform local cloud with corrected pose for visualization only
    pcl::PointCloud<PointType>::Ptr transformed = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud(*this->keyframes[i].cloud_local,
                              *transformed, pose.matrix().cast<float>());
    *corrected_map += *transformed;
  }

  this->corrected_path_pub->publish(path);
  this->corrected_kf_pose_pub->publish(kf_poses);

  // Voxel filter the corrected map before publishing
  pcl::VoxelGrid<PointType> voxel;
  voxel.setLeafSize(this->voxel_leaf_size_, this->voxel_leaf_size_, this->voxel_leaf_size_);
  voxel.setInputCloud(corrected_map);
  voxel.filter(*corrected_map);

  sensor_msgs::msg::PointCloud2 map_ros;
  pcl::toROSMsg(*corrected_map, map_ros);
  map_ros.header.stamp = this->now();
  map_ros.header.frame_id = this->odom_frame;
  this->corrected_map_pub->publish(map_ros);
}

void dlio::GraphSlamNode::publishLoopClosureMarkers() {

  visualization_msgs::msg::MarkerArray markers;

  for (size_t i = 0; i < this->loop_edges.size(); ++i) {
    const auto& edge = this->loop_edges[i];

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = this->odom_frame;
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
    const auto& pose_from = this->keyframes[edge.from_idx].pose;
    const auto& pose_to = this->keyframes[edge.to_idx].pose;
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

void dlio::GraphSlamNode::saveGraphMaps(const std::string& save_dir, float leaf_size) {

  int num_kf = static_cast<int>(this->keyframes.size());
  if (num_kf == 0) return;

  std::filesystem::create_directories(save_dir);

  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);

  // Process in chunks to limit peak memory usage.
  // Accumulate N keyframes, voxel filter, repeat, then merge filtered chunks.
  constexpr int CHUNK_SIZE = 50;
  pcl::PointCloud<PointType>::Ptr result = std::make_shared<pcl::PointCloud<PointType>>();

  for (int start = 0; start < num_kf; start += CHUNK_SIZE) {
    int end = std::min(start + CHUNK_SIZE, num_kf);

    // Pre-count points in this chunk for reserve
    size_t chunk_pts = 0;
    for (int i = start; i < end; ++i)
      chunk_pts += this->keyframes[i].cloud_local->points.size();

    pcl::PointCloud<PointType>::Ptr chunk = std::make_shared<pcl::PointCloud<PointType>>();
    chunk->points.reserve(chunk_pts);

    for (int i = start; i < end; ++i) {
      Eigen::Matrix4f T;
      if (this->optimization_done && i < static_cast<int>(this->corrected_poses.size()))
        T = this->corrected_poses[i].matrix().cast<float>();
      else
        T = this->keyframes[i].pose.matrix().cast<float>();

      pcl::PointCloud<PointType> tmp;
      pcl::transformPointCloud(*this->keyframes[i].cloud_local, tmp, T);
      chunk->points.insert(chunk->points.end(), tmp.points.begin(), tmp.points.end());
    }
    chunk->width = chunk->points.size();
    chunk->height = 1;

    // Voxel filter chunk before merging into result
    vg.setInputCloud(chunk);
    vg.filter(*chunk);

    result->points.insert(result->points.end(), chunk->points.begin(), chunk->points.end());
  }

  result->width = result->points.size();
  result->height = 1;

  // Final voxel pass to merge overlapping chunk boundaries
  vg.setInputCloud(result);
  vg.filter(*result);

  if (result->points.empty()) {
    std::cout << "[graph] saveGraphMaps: corrected cloud is empty, skipping" << std::endl;
    return;
  }

  // Derive filename from map_path_ stem (e.g. "test_gs.pcd" -> "test_gs_corrected.pcd")
  std::filesystem::path map_fp(this->map_path_);
  std::string stem = map_fp.stem().string();
  if (stem.empty()) stem = "dlio_map";
  std::string corr_file = save_dir + "/" + stem + "_corrected.pcd";
  int ret = pcl::io::savePCDFileBinary(corr_file, *result);
  if (ret == 0) {
    std::cout << "[graph] Saved corrected map: " << result->points.size() << " pts -> " << corr_file
              << (this->optimization_done ? " (graph-optimized)" : " (no loop closures, same as raw)")
              << std::endl;
  } else {
    std::cerr << "[graph] FAILED to save corrected map: " << corr_file << std::endl;
  }
}

void dlio::GraphSlamNode::autoSave() {
  std::lock_guard<std::mutex> lock1(this->corrected_mutex);
  std::lock_guard<std::mutex> lock2(this->keyframes_mutex);
  if (this->keyframes.empty()) return;

  std::filesystem::path map_fp(this->map_path_);
  std::string save_dir = map_fp.parent_path().string();
  if (save_dir.empty()) save_dir = ".";

  this->saveGraphMaps(save_dir, static_cast<float>(this->map_voxel_size_));
}

void dlio::GraphSlamNode::saveOnShutdown() {
  if (this->map_mode_ != "mapping") return;
  if (this->map_path_.empty()) return;
  if (this->shutdown_saved_.exchange(true)) return;  // already saved

  std::lock_guard<std::mutex> lock1(this->corrected_mutex);
  std::lock_guard<std::mutex> lock2(this->keyframes_mutex);
  if (this->keyframes.empty()) return;

  std::filesystem::path map_fp(this->map_path_);
  std::string save_dir = map_fp.parent_path().string();
  if (save_dir.empty()) save_dir = ".";

  this->saveGraphMaps(save_dir, static_cast<float>(this->map_voxel_size_));
}

void dlio::GraphSlamNode::savePCD(
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res) {

  std::lock_guard<std::mutex> lock1(this->corrected_mutex);
  std::lock_guard<std::mutex> lock2(this->keyframes_mutex);

  if (this->keyframes.empty()) {
    RCLCPP_WARN(this->get_logger(), "[graph] No keyframes to save");
    res->success = false;
    return;
  }

  this->saveGraphMaps(req->save_path, req->leaf_size);
  res->success = true;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(dlio::GraphSlamNode)
