/***********************************************************
 *                                                         *
 * LIO-SAM-style Map Optimization for DLIO                 *
 *                                                         *
 * GTSAM iSAM2-based incremental pose graph optimization   *
 * with loop closure detection and GPS factor integration.  *
 *                                                         *
 * Author: Azzam Wildan M                                  *
 *                                                         *
 ***********************************************************/

#ifndef DLIO_LIO_SAM_MAP_OPTIMIZATION_H
#define DLIO_LIO_SAM_MAP_OPTIMIZATION_H

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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>

// nano_gicp
#include <nano_gicp/nano_gicp.h>

// NDT-OMP
#include <pclomp/voxel_grid_covariance_omp.h>
#include <pclomp/voxel_grid_covariance_omp_impl.hpp>
#include <pclomp/ndt_omp.h>
#include <pclomp/ndt_omp_impl.hpp>

// GTSAM
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/inference/Symbol.h>

// TF2
#include <tf2_ros/transform_broadcaster.h>

// GeographicLib
#include <GeographicLib/LocalCartesian.hpp>

#include <mutex>
#include <atomic>
#include <deque>

class dlio::LioSamMapOptimizationNode : public rclcpp::Node
{

public:
  explicit LioSamMapOptimizationNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~LioSamMapOptimizationNode();

  void saveOnShutdown();

private:
  // --- Keyframe storage ---
  struct Keyframe
  {
    uint32_t id;
    rclcpp::Time timestamp;
    Eigen::Isometry3d pose;                      // original odom pose
    pcl::PointCloud<PointType>::Ptr cloud_local; // body-frame cloud
    pcl::PointCloud<PointType>::Ptr cloud_world; // world-frame cloud
    dlio::sc::ScanContextDescriptor sc_descriptor;
    dlio::sc::SectorKey sector_key;
    // GPS local coords
    float gps_x = 0.f, gps_y = 0.f, gps_z = 0.f;
    bool gps_valid = false;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  // --- Loop closure edge ---
  struct LoopConstraint
  {
    int from_idx;
    int to_idx;
    gtsam::Pose3 relative_pose;
    gtsam::noiseModel::Diagonal::shared_ptr noise;
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

  // --- Core GTSAM pipeline (called per keyframe) ---
  void addKeyframeToGraph(const Keyframe &kf);
  void addOdomFactor(int idx, const gtsam::Pose3 &pose_from, const gtsam::Pose3 &pose_to);
  void addGPSFactor(int idx, const Keyframe &kf);
  void addLoopFactors();
  void updateISAM();
  void correctPoses();

  // --- Loop closure (separate timer thread) ---
  void loopClosureThread();
  bool detectLoopClosureDistance(int &loop_cur, int &loop_pre);
  bool detectLoopClosureSC(int &loop_cur, int &loop_pre, int &sc_shift);
  void performLoopClosure();

  // --- Publishing ---
  struct PublishCorrectedData_PoseSnapshot
  {
    Eigen::Isometry3d pose;
    rclcpp::Time timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
  void publishCorrectedData();
  void publishLoopClosureMarkers();
  void publishGlobalMap(const std::vector<PublishCorrectedData_PoseSnapshot> &pose_snap,
                        const std::vector<pcl::PointCloud<PointType>::Ptr> &cloud_snap);

  // --- Map save ---
  using IsometryVec = std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>;
  void saveGraphMaps(const std::string &save_dir, float leaf_size);
  void autoSave();
  void savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
               std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res);

  // --- TF ---
  void publishMapToOdomTF();

  // --- Utility ---
  gtsam::Pose3 isometryToGtsamPose(const Eigen::Isometry3d &iso);
  Eigen::Isometry3d gtsamPoseToIsometry(const gtsam::Pose3 &pose);

  // ==================== GTSAM ====================
  gtsam::NonlinearFactorGraph gtsam_graph_;
  gtsam::Values initial_estimate_;
  gtsam::ISAM2 *isam_;
  gtsam::Values isam_current_estimate_;
  Eigen::MatrixXd pose_covariance_;

  // ==================== ROS ====================
  // Subscribers
  rclcpp::Subscription<direct_lidar_inertial_odometry::msg::KeyframeStamped>::SharedPtr keyframe_sub_;
  rclcpp::CallbackGroup::SharedPtr keyframe_cb_group_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_sub_;
  rclcpp::CallbackGroup::SharedPtr deskewed_cb_group_;

  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
  rclcpp::CallbackGroup::SharedPtr gps_cb_group_;

  // Publishers
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr corrected_path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr corrected_map_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr corrected_kf_pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr loop_closure_pub_;

  // Service
  rclcpp::Service<direct_lidar_inertial_odometry::srv::SavePCD>::SharedPtr save_pcd_srv_;
  rclcpp::CallbackGroup::SharedPtr save_pcd_cb_group_;

  // TF
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  Eigen::Isometry3d T_map_odom_cached_ = Eigen::Isometry3d::Identity();
  std::mutex tf_map_odom_mtx_;

  // Timers
  rclcpp::TimerBase::SharedPtr loop_timer_;
  rclcpp::TimerBase::SharedPtr auto_save_timer_;
  rclcpp::TimerBase::SharedPtr vis_timer_;
  rclcpp::TimerBase::SharedPtr tf_timer_;

  // ==================== Data ====================
  // Keyframes
  std::vector<Keyframe> keyframes_;
  std::mutex keyframes_mtx_;

  // KD-tree for spatial loop closure search (3D poses)
  pcl::PointCloud<pcl::PointXYZ>::Ptr keyframe_poses_3d_;
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_history_keyposes_;

  // Optimized poses (updated after iSAM2)
  std::vector<gtsam::Pose3> optimized_poses_;

  // Loop closure queues
  std::vector<LoopConstraint> loop_queue_;
  std::mutex loop_queue_mtx_;
  std::map<int, int> loop_index_container_; // prevents duplicate loop closures

  // Flags
  std::atomic<bool> a_loop_is_closed_{false};
  std::atomic<bool> shutdown_saved_{false};

  // Deskewed scan buffer
  struct DeskewedScan
  {
    rclcpp::Time timestamp;
    pcl::PointCloud<PointType>::Ptr cloud;
  };
  std::deque<DeskewedScan> deskewed_buffer_;
  std::mutex deskewed_buffer_mtx_;
  static constexpr size_t DESKEWED_BUFFER_MAX = 30;

  // GPS buffer
  std::deque<GPSMeasurement> gps_buffer_;
  std::mutex gps_buffer_mtx_;
  static constexpr size_t GPS_BUFFER_MAX = 200;
  std::unique_ptr<GeographicLib::LocalCartesian> gps_converter_;
  bool gps_origin_set_ = false;

  // ==================== Parameters ====================
  bool debug_;
  std::string odom_frame_;
  std::string map_frame_;

  // Loop closure detection
  int loop_detection_period_ms_;
  double history_keyframe_search_radius_;
  double history_keyframe_search_time_diff_;
  int history_keyframe_search_num_;
  double history_keyframe_fitness_score_;
  int min_keyframe_gap_;

  // SC++ fallback
  bool sc_enabled_;
  float sc_max_range_;
  float sc_distance_threshold_;
  float sc_ground_height_threshold_;
  int sc_search_window_;

  // Registration (ICP / GICP / NDT for loop closure)
  std::string lc_registration_method_; // "icp", "gicp", or "ndt"
  int lc_max_iterations_;
  double lc_max_corr_dist_;
  double lc_transformation_ep_;
  double lc_rotation_ep_;
  int lc_gicp_k_correspondences_;
  double ndt_resolution_;
  int ndt_num_threads_;

  // Voxel filter
  double voxel_leaf_size_;

  // GPS
  bool gps_enabled_;
  std::string gps_topic_;
  double gps_cov_threshold_;
  double pose_cov_threshold_;
  bool use_gps_elevation_;
  float gps_min_accuracy_;

  // iSAM2
  double isam_relinearize_threshold_;
  int isam_relinearize_skip_;

  // Map save / TF
  std::string map_mode_;
  std::string tf_map_odom_source_; // "odom" or "lio_sam_opt"
  std::string map_path_;
  double map_voxel_size_;
  double auto_save_interval_;
  double publish_interval_;
  double global_map_vis_radius_;
};

#endif // DLIO_LIO_SAM_MAP_OPTIMIZATION_H
