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

// ROS
#include "rclcpp/rclcpp.hpp"
#include <direct_lidar_inertial_odometry/msg/keyframe_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <direct_lidar_inertial_odometry/srv/set_mode.hpp>
#include <direct_lidar_inertial_odometry/srv/relocalize.hpp>
#include <direct_lidar_inertial_odometry/srv/set_pose.hpp>
#include <direct_lidar_inertial_odometry/srv/get_state.hpp>
#include <direct_lidar_inertial_odometry/srv/new_map.hpp>
#include <direct_lidar_inertial_odometry/srv/new_map_w_zero.hpp>
#include <direct_lidar_inertial_odometry/srv/save_pcd.hpp>

// BOOST
#include <boost/format.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/adjacent_filtered.hpp>

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
  bool saveKeyframeDatabase(); // public for atexit handler

private:
  struct State;
  struct ImuMeas;

  void getParams();

  void callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc);
  void callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu);

  void publishPose();

  void publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
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
  void setKeyframeCloud();

  void computeMetrics();
  void computeSpaciousness();
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

  // Scan Context Relocalization
  static constexpr int SC_NR = 20;               // number of rings
  static constexpr int SC_NS = 60;               // number of sectors
  using ScanContextDescriptor = Eigen::MatrixXf; // NR x NS
  using RingKey = Eigen::VectorXf;               // NR

  struct ScanContextEntry
  {
    ScanContextDescriptor descriptor;
    RingKey ring_key;
    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;
  };

  ScanContextDescriptor computeScanContext(pcl::PointCloud<PointType>::ConstPtr cloud, float max_range);
  RingKey computeRingKey(const ScanContextDescriptor &desc);
  void buildScanContextDatabase();
  std::pair<float, int> computeScanContextDistance(const ScanContextDescriptor &a, const ScanContextDescriptor &b);
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

  // Continuous localization (map→odom TF correction)
  void continuousLocalize();

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
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;
  std::mutex state_mtx_;

  // Publishers
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr kf_pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kf_cloud_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_pub;
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
  std::atomic<bool> deskew_status;
  std::atomic<int> deskew_size;

  // Threads
  std::thread publish_thread;
  std::thread publish_keyframe_thread;
  std::thread metrics_thread;
  std::thread debug_thread;

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
  pcl::PointCloud<PointType>::ConstPtr submap_cloud;
  std::shared_ptr<const nano_gicp::CovarianceList> submap_normals;
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> submap_kdtree;

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

  // GICP
  nano_gicp::NanoGICP<PointType, PointType> gicp;
  nano_gicp::NanoGICP<PointType, PointType> gicp_temp;

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

  // Metrics
  struct Metrics
  {
    std::vector<float> spaciousness;
    std::vector<float> density;
  };
  Metrics metrics;

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

  double geo_Kp_;
  double geo_Kv_;
  double geo_Kq_;
  double geo_Kab_;
  double geo_Kgb_;
  double geo_abias_max_;
  double geo_gbias_max_;

  bool debug_;
  bool debug_print_;

  // Map load/save
  std::string map_mode_;
  std::string map_path_;
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
  double last_reloc_fitness_;
  std::vector<ScanContextEntry> sc_database_;
  pcl::PointCloud<PointType>::Ptr prior_map_cloud_;
  std::shared_ptr<nanoflann::KdTreeFLANN<PointType>> prior_map_kdtree_;

  // KFDB entries accumulated during mapping
  std::vector<ScanContextEntry> kfdb_entries_;
  std::mutex kfdb_mutex_;
  Eigen::Quaternionf kfdb_gravity_q_{1.f, 0.f, 0.f, 0.f}; // gravity quaternion for KFDB SC frame

  // Continuous localization — Bayesian filter (RTAB-Map style)
  bool continuous_localize_;
  double continuous_localize_interval_;
  double continuous_localize_fitness_thresh_;
  double continuous_localize_max_correction_;
  std::string map_frame_;

  // Bayesian state
  float bayes_virtual_place_prior_;   // P(new place) prior, default 0.9
  float bayes_loop_threshold_;        // posterior threshold to accept, default 0.5
  int bayes_min_consecutive_;         // required consecutive accepts, default 2
  int bayes_consecutive_accepts_;     // current consecutive count
  std::vector<float> bayes_posterior_; // [0]=virtual place, [1..N]=keyframes

  Eigen::Matrix4f T_map_odom_;     // TF: map→odom correction
  std::mutex continuous_localize_mtx_;
  rclcpp::TimerBase::SharedPtr continuous_localize_timer_;
  pcl::PointCloud<PointType>::ConstPtr latest_scan_;  // body/sensor frame
  Eigen::Matrix4f latest_scan_T_;                      // T at time of scan (body→odom)
  double latest_scan_time_;                             // wall time when scan was stored
  std::mutex latest_scan_mtx_;
};
