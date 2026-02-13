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

#ifndef DLIO_GRAPH_SLAM_H
#define DLIO_GRAPH_SLAM_H

#include "dlio/dlio.h"

// ROS
#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <direct_lidar_inertial_odometry/msg/keyframe_stamped.hpp>
#include <direct_lidar_inertial_odometry/srv/save_pcd.hpp>

// PCL
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

// nano_gicp
#include <nano_gicp/nano_gicp.h>

// g2o
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"

#include <mutex>
#include <atomic>

class dlio::GraphSlamNode : public rclcpp::Node {

public:

  explicit GraphSlamNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~GraphSlamNode();

private:

  // --- Keyframe storage ---
  struct Keyframe {
    uint32_t id;
    rclcpp::Time timestamp;
    Eigen::Isometry3d pose;
    pcl::PointCloud<PointType>::Ptr cloud_local;
    pcl::PointCloud<PointType>::Ptr cloud_world;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  // --- Loop closure edge ---
  struct LoopEdge {
    int from_idx;
    int to_idx;
    Eigen::Isometry3d relative_pose;
    Eigen::Matrix<double, 6, 6> information;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  // --- Parameters ---
  void getParams();

  // --- Callbacks ---
  void callbackKeyframe(const direct_lidar_inertial_odometry::msg::KeyframeStamped::SharedPtr msg);

  // --- Loop closure ---
  void searchLoopClosure();
  bool detectLoopCandidate(const std::vector<Keyframe>& kfs, int current_idx,
                            int& candidate_idx, double& candidate_dist);
  bool performLoopRegistration(const std::vector<Keyframe>& kfs, int current_idx,
                                int candidate_idx, Eigen::Isometry3d& relative_pose,
                                double& fitness_score);

  // --- Pose graph optimization ---
  void optimizePoseGraph();

  // --- Publishing ---
  void publishCorrectedData();
  void publishLoopClosureMarkers();

  // --- Save PCD service ---
  void savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
               std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res);

  // --- ROS ---
  rclcpp::Subscription<direct_lidar_inertial_odometry::msg::KeyframeStamped>::SharedPtr keyframe_sub;
  rclcpp::CallbackGroup::SharedPtr keyframe_cb_group;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr corrected_path_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr corrected_map_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr corrected_kf_pose_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr loop_closure_pub;

  rclcpp::Service<direct_lidar_inertial_odometry::srv::SavePCD>::SharedPtr save_pcd_srv;
  rclcpp::CallbackGroup::SharedPtr save_pcd_cb_group;

  rclcpp::TimerBase::SharedPtr loop_timer;

  // --- Data ---
  std::vector<Keyframe> keyframes;
  std::mutex keyframes_mutex;

  std::vector<LoopEdge> loop_edges;

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> corrected_poses;
  std::mutex corrected_mutex;

  std::atomic<bool> optimization_done;
  int last_loop_checked_idx;

  // --- Frames ---
  std::string odom_frame;

  // --- Parameters ---
  int loop_detection_period_ms_;
  double voxel_leaf_size_;
  int search_submap_num_;

  // loop closure detection
  double range_of_searching_loop_;
  int min_keyframe_gap_;
  double threshold_loop_closure_score_;

  // loop closure registration (nano_gicp)
  int lc_gicp_k_correspondences_;
  double lc_gicp_max_corr_dist_;
  int lc_gicp_max_iter_;
  double lc_gicp_transformation_ep_;
  double lc_gicp_rotation_ep_;

  // pose graph
  int num_adjacent_constraints_;
  int optimization_iterations_;
  double odom_edge_info_scale_;
  double loop_edge_info_scale_;

  bool debug_;

};

#endif // DLIO_GRAPH_SLAM_H
