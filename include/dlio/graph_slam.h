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
#include "dlio/scan_context.h"

// ROS
#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <direct_lidar_inertial_odometry/msg/keyframe_stamped.hpp>
#include <direct_lidar_inertial_odometry/srv/save_pcd.hpp>

// PCL
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

// nano_gicp
#include <nano_gicp/nano_gicp.h>

// NDT-OMP
#include <pclomp/voxel_grid_covariance_omp.h>
#include <pclomp/voxel_grid_covariance_omp_impl.hpp>
#include <pclomp/ndt_omp.h>
#include <pclomp/ndt_omp_impl.hpp>

// g2o
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"

#include <GeographicLib/LocalCartesian.hpp>

#include <mutex>
#include <atomic>
#include <deque>

class dlio::GraphSlamNode : public rclcpp::Node
{

public:
  explicit GraphSlamNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~GraphSlamNode();

  void saveOnShutdown();

private:
  // --- Keyframe storage ---
  struct Keyframe
  {
    uint32_t id;
    rclcpp::Time timestamp;
    Eigen::Isometry3d pose;
    pcl::PointCloud<PointType>::Ptr cloud_local;
    pcl::PointCloud<PointType>::Ptr cloud_world;
    dlio::sc::ScanContextDescriptor sc_descriptor;
    dlio::sc::SectorKey sector_key;
    // GPS
    float gps_x = 0.f, gps_y = 0.f, gps_z = 0.f;
    bool gps_valid = false;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  // --- Loop closure edge ---
  struct LoopEdge
  {
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
  void callbackDeskewed(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void callbackGPS(const sensor_msgs::msg::NavSatFix::SharedPtr msg);

  // --- GPS helpers ---
  struct GPSMeasurement
  {
    double latitude, longitude, altitude, timestamp;
    float horizontal_accuracy;
  };
  bool getGPSAtTime(double timestamp, GPSMeasurement &out);
  bool gpsToLocal(double lat, double lon, double alt, float &x, float &y, float &z);

  // --- Loop closure ---
  void searchLoopClosure();
  bool detectLoopCandidate(const std::vector<Keyframe> &kfs, int current_idx,
                           int &candidate_idx, double &candidate_dist, int &sc_shift);
  bool performLoopRegistration(const std::vector<Keyframe> &kfs, int current_idx,
                               int candidate_idx, int sc_shift,
                               Eigen::Isometry3d &relative_pose, double &fitness_score);

  // --- Pose graph optimization ---
  void optimizePoseGraph(const std::vector<Keyframe> &kf_snap);

  // --- Publishing ---
  void publishCorrectedData(const std::vector<Keyframe> &kf_snap);
  void publishLoopClosureMarkers(const std::vector<Keyframe> &kf_snap);

  // --- Map save ---
  using IsometryVec = std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>;
  void saveGraphMaps(const std::string &save_dir, float leaf_size,
                     const std::vector<Keyframe> &kf_snap,
                     const IsometryVec &poses_snap, bool opt_done);
  void autoSave();
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
  rclcpp::TimerBase::SharedPtr auto_save_timer_;

  // Deskewed scan buffer (dense scans for SC computation)
  struct DeskewedScan
  {
    rclcpp::Time timestamp;
    pcl::PointCloud<PointType>::Ptr cloud; // in odom (world) frame
  };
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_sub_;
  rclcpp::CallbackGroup::SharedPtr deskewed_cb_group_;
  std::deque<DeskewedScan> deskewed_buffer_;
  std::mutex deskewed_buffer_mtx_;
  static constexpr size_t DESKEWED_BUFFER_MAX = 30;

  // GPS
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
  rclcpp::CallbackGroup::SharedPtr gps_cb_group_;
  std::deque<GPSMeasurement> gps_buffer_;
  std::mutex gps_buffer_mtx_;
  static constexpr size_t GPS_BUFFER_MAX = 200;
  std::unique_ptr<GeographicLib::LocalCartesian> gps_converter_;
  bool gps_origin_set_ = false;

  // --- Data ---
  std::vector<Keyframe> keyframes;
  std::mutex keyframes_mutex;

  std::vector<LoopEdge> loop_edges;

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> corrected_poses;
  std::mutex corrected_mutex;

  std::atomic<bool> optimization_done;
  std::atomic<bool> shutdown_saved_;
  int last_loop_checked_idx;

  // --- Frames ---
  std::string odom_frame;
  std::string map_frame_;

  // --- Parameters ---
  int loop_detection_period_ms_;
  double voxel_leaf_size_;
  int search_submap_num_;

  // loop closure detection
  double range_of_searching_loop_;
  int min_keyframe_gap_;
  double threshold_loop_closure_score_;

  // loop closure registration
  bool use_gicp_;
  int lc_gicp_k_correspondences_;
  double lc_gicp_max_corr_dist_;
  int lc_gicp_max_iter_;
  double lc_gicp_transformation_ep_;
  double lc_gicp_rotation_ep_;

  // NDT params (loop closure)
  double ndt_resolution_;
  int ndt_num_threads_;

  // pose graph
  int num_adjacent_constraints_;
  int optimization_iterations_;
  double odom_edge_info_scale_;
  double loop_edge_info_scale_;

  // SC++ loop closure detection
  float sc_max_range_;
  float sc_distance_threshold_;
  float sc_ground_height_threshold_;
  int sc_search_window_;

  // GPS-assisted loop closure
  bool gps_enabled_;
  std::string gps_topic_;
  float gps_search_radius_;
  float gps_min_accuracy_;

  bool debug_;

  // map save
  std::string map_mode_;
  std::string map_path_;
  double map_voxel_size_;
  double auto_save_interval_;
  double publish_interval_;
};

#endif // DLIO_GRAPH_SLAM_H
