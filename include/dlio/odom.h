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

#include "dlio/dlio.h"
#include "dlio/scan_context.h"
#include "dlio/occupancy_grid.h"
#include "dlio/robust_icp.h"
#include "dlio/voxel_hash_map.h"

// g2o
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"

// ROS
#include "rclcpp/rclcpp.hpp"
#include <direct_lidar_inertial_odometry/msg/keyframe_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <GeographicLib/LocalCartesian.hpp>
#include <GeographicLib/Geocentric.hpp>
#include <direct_lidar_inertial_odometry/srv/set_mode.hpp>
#include <direct_lidar_inertial_odometry/srv/relocalize.hpp>
#include <direct_lidar_inertial_odometry/srv/set_pose.hpp>
#include <direct_lidar_inertial_odometry/srv/get_state.hpp>
#include <direct_lidar_inertial_odometry/srv/new_map.hpp>
#include <direct_lidar_inertial_odometry/srv/new_map_w_zero.hpp>
#include <direct_lidar_inertial_odometry/srv/save_pcd.hpp>

// STL (explicit for GPS buffer)
#include <deque>

// BOOST
#include <boost/format.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/adjacent_filtered.hpp>

// NDT-OMP
#include <pclomp/voxel_grid_covariance_omp.h>
#include <pclomp/voxel_grid_covariance_omp_impl.hpp>
#include <pclomp/ndt_omp.h>
#include <pclomp/ndt_omp_impl.hpp>

// PCL
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl_conversions/pcl_conversions.h>

class dlio::OdomNode : public rclcpp::Node
{

public:
  explicit OdomNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~OdomNode();

  void start();
  bool saveKeyframeDatabase();          // public for atexit handler
  bool saveCorrectedKeyframeDatabase(); // public for atexit handler

private:
  struct State;
  struct ImuMeas;

  void getParams();

  void callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc);
  void callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu);

  void publishPose();

  void publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud,
                    pcl::PointCloud<PointType>::ConstPtr raw_deskewed_cloud,
                    Eigen::Matrix4f T_cloud);
  void publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud,
                    pcl::PointCloud<PointType>::ConstPtr raw_deskewed_cloud,
                    Eigen::Matrix4f T_cloud);
  void publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                                 pcl::PointCloud<PointType>::ConstPtr>
                           kf,
                       rclcpp::Time timestamp,
                       pcl::PointCloud<PointType>::ConstPtr local_cloud);

  void getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr &pc);
  void preprocessPoints();
  void deskewPointcloud();
  void initializeInputTarget();
  void setInputSource();

  void initializeDLIO();

  void getNextPose();
  bool imuMeasFromTimeRange(double start_time, double end_time,
                            boost::circular_buffer<ImuMeas>::reverse_iterator &begin_imu_it,
                            boost::circular_buffer<ImuMeas>::reverse_iterator &end_imu_it);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
  integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
               const std::vector<double> &sorted_timestamps);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
  integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                       const std::vector<double> &sorted_timestamps,
                       boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                       boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it);
  void propagateGICP();

  void propagateState();
  void updateState();

  void setAdaptiveParams();

  void computeMetrics(pcl::PointCloud<PointType>::ConstPtr scan);
  void computeSpaciousness(pcl::PointCloud<PointType>::ConstPtr scan);
  void computeDensity();

  sensor_msgs::msg::Imu::SharedPtr transformImu(const sensor_msgs::msg::Imu::SharedPtr &imu);

  void updateKeyframes();
  void computeConvexHull();
  void computeConcaveHull();
  void pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames);
  void buildSubmap(State vehicle_state);
  void buildKeyframesAndSubmap(State vehicle_state);
  void pauseSubmapBuildIfNeeded();

  void loadPriorMap();

  // Scan Context Relocalization (types in dlio/scan_context.h)
  void buildScanContextDatabase();
  bool runRelocalization(pcl::PointCloud<PointType>::ConstPtr scan);

  // Keyframe Database (KFDB) — save real SC descriptors during mapping
  std::string getKfdbPath() const;
  void computeAndStoreKeyframeSC();
  bool loadKeyframeDatabase();

  // Runtime control services
  void srvSetMode(std::shared_ptr<direct_lidar_inertial_odometry::srv::SetMode::Request> req,
                  std::shared_ptr<direct_lidar_inertial_odometry::srv::SetMode::Response> res);
  void srvRelocalize(std::shared_ptr<direct_lidar_inertial_odometry::srv::Relocalize::Request> req,
                     std::shared_ptr<direct_lidar_inertial_odometry::srv::Relocalize::Response> res);
  void srvSetPose(std::shared_ptr<direct_lidar_inertial_odometry::srv::SetPose::Request> req,
                  std::shared_ptr<direct_lidar_inertial_odometry::srv::SetPose::Response> res);
  void srvGetState(std::shared_ptr<direct_lidar_inertial_odometry::srv::GetState::Request> req,
                   std::shared_ptr<direct_lidar_inertial_odometry::srv::GetState::Response> res);
  void srvNewMap(std::shared_ptr<direct_lidar_inertial_odometry::srv::NewMap::Request> req,
                 std::shared_ptr<direct_lidar_inertial_odometry::srv::NewMap::Response> res);
  void srvNewMapWZero(std::shared_ptr<direct_lidar_inertial_odometry::srv::NewMapWZero::Request> req,
                      std::shared_ptr<direct_lidar_inertial_odometry::srv::NewMapWZero::Response> res);
  void reloadPriorMapForRelocalization();
  void clearAllMapData();
  bool callSavePCD();
  bool callSaveCorrectedPCD();
  // Continuous localization (map→odom TF correction)
  void continuousLocalize();
  bool verifyLoopWithG2O(int loop_kf_idx,
                         const Eigen::Matrix4f &T_map_body_gicp,
                         const Eigen::Matrix4f &T_odom_body,
                         const Eigen::Matrix4f &T_map_odom_current,
                         const std::vector<dlio::sc::ScanContextEntry> &sc_snap,
                         double &out_chi2, int &out_num_anchors);

  // Submap-based relocalization (GPS-denied)
  void initSubmapLocalization();
  void submapLocalizeTick();
  void submapLocalizeStage1(pcl::PointCloud<PointType>::ConstPtr scan_body,
                            const Eigen::Matrix4f &T_odom_body);
  void submapLocalizeStage2(pcl::PointCloud<PointType>::ConstPtr scan_body,
                            const Eigen::Matrix4f &T_odom_body);
  int findClosestKF(const Eigen::Vector3f &pos) const;

  void debug();

  rclcpp::TimerBase::SharedPtr publish_timer;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::CallbackGroup::SharedPtr lidar_cb_group, imu_cb_group;

  // Services
  rclcpp::Service<direct_lidar_inertial_odometry::srv::SetMode>::SharedPtr set_mode_srv_;
  rclcpp::Service<direct_lidar_inertial_odometry::srv::Relocalize>::SharedPtr relocalize_srv_;
  rclcpp::Service<direct_lidar_inertial_odometry::srv::SetPose>::SharedPtr set_pose_srv_;
  rclcpp::Service<direct_lidar_inertial_odometry::srv::GetState>::SharedPtr get_state_srv_;
  rclcpp::Service<direct_lidar_inertial_odometry::srv::NewMap>::SharedPtr new_map_srv_;
  rclcpp::Service<direct_lidar_inertial_odometry::srv::NewMapWZero>::SharedPtr new_map_w_zero_srv_;
  rclcpp::Client<direct_lidar_inertial_odometry::srv::SavePCD>::SharedPtr save_pcd_client_;
  rclcpp::Client<direct_lidar_inertial_odometry::srv::SavePCD>::SharedPtr save_corrected_pcd_client_;
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;
  std::mutex state_mtx_;

  // Publishers
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr kf_pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kf_cloud_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_raw_pub;
  rclcpp::Publisher<direct_lidar_inertial_odometry::msg::KeyframeStamped>::SharedPtr kf_stamped_pub;

  // TF
  std::shared_ptr<tf2_ros::TransformBroadcaster> br;

  // ROS Msgs
  nav_msgs::msg::Odometry odom_ros;
  geometry_msgs::msg::PoseStamped pose_ros;
  nav_msgs::msg::Path path_ros;
  geometry_msgs::msg::PoseArray kf_pose_ros;

  // Flags
  std::atomic<bool> dlio_initialized;
  std::atomic<bool> first_valid_scan;
  std::atomic<bool> first_imu_received;
  std::atomic<bool> imu_calibrated;
  std::atomic<bool> submap_hasChanged;
  std::atomic<bool> gicp_hasConverged;
  double last_fitness_ = 0.0;
  std::atomic<bool> deskew_status;
  std::atomic<int> deskew_size;

  // Threads
  std::thread publish_thread;
  std::thread publish_keyframe_thread;
  std::thread metrics_thread;
  std::thread debug_thread;

  // Mutex for detached publish/metrics threads to prevent concurrent push_back
  std::mutex publish_mtx_;
  std::mutex kf_publish_mtx_;
  std::mutex metrics_mtx_;

  // Trajectory
  std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>> trajectory;
  double length_traversed;

  // Keyframes
  std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                        pcl::PointCloud<PointType>::ConstPtr>>
      keyframes;
  std::vector<rclcpp::Time> keyframe_timestamps;
  std::vector<std::shared_ptr<const nano_gicp::CovarianceList>> keyframe_normals;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_transformations;
  std::mutex keyframes_mutex;

  // Sensor Type
  dlio::SensorType sensor;

  // Frames
  std::string odom_frame;
  std::string baselink_frame;
  std::string lidar_frame;
  std::string imu_frame;

  // Preprocessing
  pcl::CropBox<PointType> crop;
  pcl::VoxelGrid<PointType> voxel;

  // Point Clouds
  pcl::PointCloud<PointType>::ConstPtr original_scan;
  pcl::PointCloud<PointType>::ConstPtr deskewed_scan;
  pcl::PointCloud<PointType>::ConstPtr current_scan;

  // Keyframes
  pcl::PointCloud<PointType>::ConstPtr keyframe_cloud;
  int num_processed_keyframes;

  pcl::ConvexHull<PointType> convex_hull;
  pcl::ConcaveHull<PointType> concave_hull;
  std::vector<int> keyframe_convex;
  std::vector<int> keyframe_concave;

  // Submap
  std::string submap_method_;
  pcl::PointCloud<PointType>::ConstPtr submap_cloud;
  std::shared_ptr<const nano_gicp::CovarianceList> submap_normals;
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> submap_kdtree;
  std::unique_ptr<dlio::VoxelHashMap> voxel_map_;
  int voxel_map_last_kf_idx_ = 0;

  std::vector<int> submap_kf_idx_curr;
  std::vector<int> submap_kf_idx_prev;

  bool new_submap_is_ready;
  std::future<void> submap_future;
  std::condition_variable submap_build_cv;
  bool main_loop_running;
  std::mutex main_loop_running_mutex;

  // Timestamps
  rclcpp::Time scan_header_stamp;
  double scan_stamp;
  double prev_scan_stamp;
  double scan_dt;
  std::vector<double> comp_times;
  std::vector<double> imu_rates;
  std::vector<double> lidar_rates;

  double first_scan_stamp;
  double elapsed_time;

  // Registration (GICP, NDT, or Robust ICP)
  std::string registration_method_;
  bool use_gicp_;
  nano_gicp::NanoGICP<PointType, PointType> gicp;
  nano_gicp::NanoGICP<PointType, PointType> gicp_temp;
  pclomp::NormalDistributionsTransform<PointType, PointType> ndt;
  pclomp::NormalDistributionsTransform<PointType, PointType> ndt_temp;
  dlio::RobustICP robust_icp_;
  dlio::RobustICP robust_icp_temp_;

  // Transformations
  Eigen::Matrix4f T, T_prior, T_corr;
  Eigen::Quaternionf q_final;

  Eigen::Vector3f origin;

  struct Extrinsics
  {
    struct SE3
    {
      Eigen::Vector3f t;
      Eigen::Matrix3f R;
    };
    SE3 baselink2imu;
    SE3 baselink2lidar;
    Eigen::Matrix4f baselink2imu_T;
    Eigen::Matrix4f baselink2lidar_T;
  };
  Extrinsics extrinsics;

  // IMU
  rclcpp::Time imu_stamp;
  double first_imu_stamp;
  double prev_imu_stamp;
  double imu_dp, imu_dq_deg;
  double imu_transform_prev_stamp_ = 0.0;
  Eigen::Vector3f imu_transform_ang_vel_prev_ = Eigen::Vector3f::Zero();

  struct ImuMeas
  {
    double stamp;
    double dt; // defined as the difference between the current and the previous measurement
    Eigen::Vector3f ang_vel;
    Eigen::Vector3f lin_accel;
  };
  ImuMeas imu_meas;

  boost::circular_buffer<ImuMeas> imu_buffer;
  std::mutex mtx_imu;
  std::condition_variable cv_imu_stamp;

  static bool comparatorImu(ImuMeas m1, ImuMeas m2)
  {
    return (m1.stamp < m2.stamp);
  };

  // Geometric Observer
  struct Geo
  {
    bool first_opt_done;
    std::mutex mtx;
    double dp;
    double dq_deg;
    Eigen::Vector3f prev_p;
    Eigen::Quaternionf prev_q;
    Eigen::Vector3f prev_vel;
  };
  Geo geo;

  // State Vector
  struct ImuBias
  {
    Eigen::Vector3f gyro;
    Eigen::Vector3f accel;
  };

  struct Frames
  {
    Eigen::Vector3f b;
    Eigen::Vector3f w;
  };

  struct Velocity
  {
    Frames lin;
    Frames ang;
  };

  struct State
  {
    Eigen::Vector3f p;    // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
    Velocity v;
    ImuBias b; // imu biases in body frame
  };
  State state;

  struct Pose
  {
    Eigen::Vector3f p;    // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
  };
  Pose lidarPose;
  Pose imuPose;

  // Submap-based relocalization types
  struct SubmapEntry
  {
    int submap_id;
    std::vector<int> kf_indices;           // indices into sc_database_
    Eigen::Vector3f centroid;              // centroid of member KF positions
    pcl::PointCloud<PointType>::Ptr cloud; // pre-built submap cloud
    float probability;                     // current probability robot is here
  };

  enum class SubmapLocState
  {
    IDLE,
    STAGE1,
    STAGE2
  };

  struct SubmapMotionTracker
  {
    int current_kf_idx;
    int prev_kf_idx;
    Eigen::Matrix4f T_odom_prev;
    Eigen::Matrix4f T_map_body_prev;
    int consecutive_valid;
    int total_validated;
  };

  // Metrics
  struct Metrics
  {
    std::vector<float> spaciousness;
    std::vector<float> density;
  };
  Metrics metrics;
  float spaciousness_median_prev_ = 0.f;
  float density_prev_ = 0.f;
  uint8_t debug_print_counter_ = 0;

  std::string cpu_type;
  std::vector<double> cpu_percents;
  clock_t lastCPU, lastSysCPU, lastUserCPU;
  int numProcessors;

  // Parameters
  std::string version_;
  int num_threads_;

  bool deskew_;

  double gravity_;

  bool time_offset_;

  bool adaptive_params_;

  double obs_submap_thresh_;
  double obs_keyframe_thresh_;
  double obs_keyframe_lag_;

  double keyframe_thresh_dist_;
  double keyframe_thresh_rot_;

  int submap_knn_;
  int submap_kcv_;
  int submap_kcc_;
  double submap_concave_alpha_;

  bool densemap_filtered_;
  bool wait_until_move_;

  double crop_size_;

  bool vf_use_;
  double vf_res_;

  bool imu_calibrate_;
  bool calibrate_gyro_;
  bool calibrate_accel_;
  bool gravity_align_;
  double imu_calib_time_;
  int imu_buffer_size_;
  Eigen::Matrix3f imu_accel_sm_;

  int gicp_min_num_points_;
  int gicp_k_correspondences_;
  double gicp_max_corr_dist_;
  int gicp_max_iter_;
  double gicp_transformation_ep_;
  double gicp_rotation_ep_;
  double gicp_init_lambda_factor_;

  // NDT params
  double ndt_resolution_;
  int ndt_num_threads_;
  std::string ndt_search_method_;
  double ndt_step_size_;

  double geo_Kp_;
  double geo_Kv_;
  double geo_Kq_;
  double geo_Kab_;
  double geo_Kgb_;
  double geo_abias_max_;
  double geo_gbias_max_;

  bool debug_;
  bool debug_print_;
  bool deep_debug_ = false;
  bool use_2d_imu_ = false;
  bool use_imu_ = true;
  Eigen::Matrix4f T_prev_ = Eigen::Matrix4f::Identity();

  // Map load/save
  std::string map_mode_;
  std::string tf_map_odom_source_; // "odom" or "lio_sam_opt"
  std::string map_path_;
  bool use_corrected_;
  double map_voxel_size_;
  double map_chunk_size_;
  bool use_prior_map_;
  Eigen::Vector3f initial_position_;
  float initial_yaw_;
  bool prior_map_pose_set_;
  int num_prior_keyframes_;

  // Scan Context Relocalization
  bool relocalize_;
  bool relocalized_;
  float sc_max_range_;
  int sc_num_candidates_;
  double sc_distance_threshold_;
  int sc_max_attempts_;
  int sc_attempt_count_;
  float sc_ground_height_threshold_; // ground removal height for SC (0 = disabled)
  int sc_search_window_;             // SC++ local search window (default 7)
  double last_reloc_fitness_;
  std::vector<dlio::sc::ScanContextEntry> sc_database_;
  pcl::PointCloud<PointType>::Ptr prior_map_cloud_;
  std::shared_ptr<nanoflann::KdTreeFLANN<PointType>> prior_map_kdtree_;

  // KFDB entries accumulated during mapping
  std::vector<dlio::sc::ScanContextEntry> kfdb_entries_;
  std::mutex kfdb_mutex_;
  Eigen::Quaternionf kfdb_gravity_q_{1.f, 0.f, 0.f, 0.f}; // gravity quaternion for KFDB SC frame

  // Corrected keyframe poses from graph_slam (for corrected KFDB)
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr corrected_kf_poses_sub_;
  std::vector<geometry_msgs::msg::Pose> corrected_kf_poses_;
  std::mutex corrected_kf_poses_mutex_;

  // Continuous localization — Bayesian filter (RTAB-Map style)
  bool continuous_localize_;
  bool continuous_localize_on_mapping_; // also run during mapping mode (drift correction vs prior map)
  double continuous_localize_interval_;
  double continuous_localize_fitness_thresh_;
  double continuous_localize_max_correction_;
  std::string map_frame_;

  // Bayesian state
  float bayes_virtual_place_prior_;    // P(new place) prior, default 0.9
  float bayes_loop_threshold_;         // posterior threshold to accept, default 0.5
  int bayes_min_consecutive_;          // required consecutive accepts, default 2
  float bayes_sc_dist_threshold_;      // max SC distance for candidate (0 = no filter)
  int bayes_sc_top_k_;                 // only keep top-K SC matches (0 = no filter)
  int bayes_consecutive_accepts_;      // current consecutive count
  std::vector<float> bayes_posterior_; // [0]=virtual place, [1..N]=keyframes

  // g2o pose graph verification
  bool g2o_verification_enabled_;
  double g2o_chi2_threshold_;
  int g2o_iterations_;

  Eigen::Matrix4f T_map_odom_; // TF: map→odom correction
  std::mutex continuous_localize_mtx_;
  rclcpp::TimerBase::SharedPtr continuous_localize_timer_;
  pcl::PointCloud<PointType>::ConstPtr latest_scan_; // body/sensor frame
  Eigen::Matrix4f latest_scan_T_;                    // T at time of scan (body→odom)
  double latest_scan_time_;                          // wall time when scan was stored
  std::mutex latest_scan_mtx_;

  // Confidence publisher + global correction control
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr confidence_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr global_correction_sub_;
  std::atomic<bool> enable_global_correction_{true};
  float last_confidence_{0.0f};

  // Occupancy grid
  bool occupancy_grid_enabled_ = false;
  std::unique_ptr<dlio::OccupancyGridGenerator> occupancy_grid_gen_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub_;
  rclcpp::TimerBase::SharedPtr occupancy_grid_timer_;
  double og_last_scan_time_ = 0.0;
  void publishOccupancyGrid();

  // GPS subscriber + callback
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
  rclcpp::CallbackGroup::SharedPtr gps_cb_group_;
  void callbackGPS(const sensor_msgs::msg::NavSatFix::SharedPtr msg);

  // GPS state
  struct GPSMeasurement
  {
    double latitude, longitude, altitude;
    double timestamp;
    float horizontal_accuracy;
    uint8_t status;
  };
  std::deque<GPSMeasurement> gps_buffer_;
  std::mutex gps_buffer_mtx_;
  static constexpr size_t GPS_BUFFER_MAX = 200;

  // GPS coordinate converter
  std::unique_ptr<GeographicLib::LocalCartesian> gps_converter_;
  bool gps_origin_set_ = false;
  double gps_origin_lat_ = 0.0;
  double gps_origin_lon_ = 0.0;
  double gps_origin_alt_ = 0.0;

  // GPS params
  bool gps_enabled_ = false;
  std::string gps_topic_;
  double gps_origin_param_lat_ = 0.0;
  double gps_origin_param_lon_ = 0.0;
  double gps_origin_param_alt_ = 0.0;
  float gps_search_radius_ = 30.0f;
  float gps_min_accuracy_ = 5.0f;
  bool gps_publish_earth_tf_ = true;
  bool gps_trust_all_ = false; // bypass status + accuracy filters (for sim)

  // GPS correction tracking (g2o only after first successful correction)
  bool gps_corrected_once_ = false;

  // earth→map TF (static, published once when origin is set)
  bool earth_tf_published_ = false;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
  void publishEarthToMapTF();

  // Helper: get interpolated GPS at a given timestamp
  bool getGPSAtTime(double timestamp, GPSMeasurement &out);

  // Helper: convert lat/lon to local ENU
  bool gpsToLocal(double lat, double lon, double alt,
                  float &x, float &y, float &z);

  // Submap-based relocalization
  bool submap_loc_enabled_;
  double submap_loc_interval_;            // timer interval (s), default 2.0
  int submap_loc_group_size_;             // KFs per submap, default 10
  double submap_loc_search_radius_;       // radius for cloud extraction (m), default 50.0
  double submap_loc_fitness_thresh_;      // GICP fitness threshold, default 0.15
  double submap_loc_prob_threshold_;      // prob to enter Stage 2, default 0.6
  double submap_loc_motion_error_thresh_; // max pos error Tlink vs pTlink (m), default 1.0
  double submap_loc_motion_rot_thresh_;   // max rot error (deg), default 5.0
  int submap_loc_min_motion_valid_;       // consecutive valid steps needed, default 3
  double submap_loc_prob_drop_thresh_;    // prob to revert to Stage 1, default 0.3
  double submap_loc_max_correction_;      // max TF correction (m), default 5.0
  float submap_loc_sc_dist_thresh_;       // SC distance filter for Stage1 (0=disabled), default 0.5
  int submap_loc_sc_accum_scans_;         // number of recent scans to accumulate for SC descriptor, default 5

  SubmapLocState submap_loc_state_;
  std::vector<SubmapEntry> submap_entries_;
  SubmapMotionTracker submap_motion_;
  rclcpp::TimerBase::SharedPtr submap_loc_timer_;
  std::mutex submap_loc_mtx_;

  // Ring buffer of recent scans for accumulated SC descriptor
  struct ScanStamped
  {
    pcl::PointCloud<PointType>::Ptr cloud;
    Eigen::Matrix4f T_odom_body;
  };
  std::deque<ScanStamped> submap_loc_scan_buffer_;

  // Shadow T_map_odom for submap localization — does NOT touch the system
  // T_map_odom_ until correction is validated.  Updated from Stage1 GICP
  // result so that Stage2 init_guess is accurate.
  Eigen::Matrix4f submap_loc_T_map_odom_shadow_;
};
