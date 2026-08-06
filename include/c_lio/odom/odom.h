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

#include "c_lio/c_lio.h"
#include "c_lio/engines/appearance_engine.h"
#include "c_lio/odom/occupancy_grid.h"
#include "c_lio/algorithms/robust_icp.h"
#include "c_lio/algorithms/voxel_hash_map.h"
#include "c_lio/algorithms/error_state_ekf.h"
#include "c_lio/algorithms/classic_mode_kf.h"
#include "c_lio/engines/registration_engine.h"
#include "c_lio/engines/prefilter_engine.h"
// registration_helper.h removed — using registration_engine.h

namespace c_lio
{
  enum class FusionMethod
  {
    GEO,
    KF,
    EKF
  };
  FusionMethod parseFusionMethod(const std::string &s);

  enum class MotionModelType
  {
    NONE,
    ACKERMANN,
    DIFF_DRIVE,
    HOLONOMIC
  };
  MotionModelType parseMotionModelType(const std::string &s);
}

// GTSAM
#include <gtsam/navigation/PreintegratedRotation.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/geometry/Pose3.h>

// g2o
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"

// ROS
#include "rclcpp/rclcpp.hpp"
#include <c_lio/msg/keyframe_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <GeographicLib/LocalCartesian.hpp>
#include <GeographicLib/Geocentric.hpp>
#include <c_lio/srv/set_mode.hpp>
#include <c_lio/srv/relocalize.hpp>
#include <c_lio/srv/set_pose.hpp>
#include <c_lio/srv/get_state.hpp>
#include <c_lio/srv/new_map.hpp>
#include <c_lio/srv/new_map_w_zero.hpp>
#include <c_lio/srv/save_pcd.hpp>

// STL (explicit for GPS buffer)
#include <deque>
#include <tuple>

// BOOST
#include <boost/format.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/adjacent_filtered.hpp>

// NDT-OMP and NDT-CUDA included via registration_engine.h

// PCL
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl_conversions/pcl_conversions.h>

class c_lio::OdomNode : public rclcpp::Node
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

  // Alpha-Beta-Gamma tracker for lidarPose prediction-based gating
  struct LidarPoseTracker
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Position ABG state
    Eigen::Vector3f p = Eigen::Vector3f::Zero();
    Eigen::Vector3f v = Eigen::Vector3f::Zero();
    Eigen::Vector3f a = Eigen::Vector3f::Zero();

    // Rotation ABG state
    Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
    Eigen::Vector3f omega = Eigen::Vector3f::Zero();
    Eigen::Vector3f alpha_rot = Eigen::Vector3f::Zero();

    // ABG gains (position)
    float alpha_p = 0.5f;
    float beta_p = 0.4f;
    float gamma_p = 0.1f;

    // ABG gains (rotation)
    float alpha_q = 0.5f;
    float beta_q = 0.3f;
    float gamma_q = 0.05f;

    // Gate thresholds on residual
    float max_pos_residual = 2.0f;
    float max_rot_residual_deg = 30.0f;

    double prev_stamp = 0.0;
    bool initialized = false;

    void initialize(const Eigen::Vector3f &z_p, const Eigen::Quaternionf &z_q, double stamp);
    std::pair<float, float> predictAndGate(const Eigen::Vector3f &z_p, const Eigen::Quaternionf &z_q, double stamp);
    void update(const Eigen::Vector3f &z_p, const Eigen::Quaternionf &z_q, double stamp);
  };

  void getParams();

  void callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc);
  void callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu);
  void callbackExternalOdom(const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg);

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

  void initializeC_LIO();

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
  void updateStateGeo();
  void updateStateKF();
  void updateStateEKF();
  void propagateStateKF(double dt);
  bool evaluatePoseGate();
  void applyMotionModelConstraint();
  void applyMotionModelConstraintImu(Eigen::Vector3f &v, const Eigen::Quaternionf &q);

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
  // Localization: build registration target as an ROI of the frozen prior map
  // around the robot (prior_map_kdtree_ radiusSearch + sliced covariances).
  bool buildPriorMapRoiSubmap(const State &vehicle_state);
  // Localization memory bound: free clouds/covariances of old live keyframes.
  void pruneLocalizationKeyframes();

  // Classic localization sub-mode (odom/submap/method == "classic"). Runs on
  // its own timer (classic_localization_timer_), independent of the scan
  // callback — see src/c_lio/odom/odom_classic_mode.cc.
  void callbackUnlocalizedOdom(const nav_msgs::msg::Odometry::SharedPtr msg);
  // Returns (position, orientation, source_sequence_number). The sequence
  // number lets classic_localization_routine() skip ticks where the
  // underlying source (latest_scan_seq_ for Case A, classic_unlocalized_
  // odom_seq_ for Case B) hasn't actually produced new data yet —
  // classic_localization_timer_ runs at its own configured rate, decoupled
  // from the scan/topic rate, so without this check a faster timer would
  // repeatedly read the same stale value (zero delta) between real updates,
  // then jump — a stair-step pattern that reads as jitter. Deliberately a
  // plain counter, NOT this->now()-derived — see latest_scan_seq_'s comment.
  std::tuple<Eigen::Vector3f, Eigen::Quaternionf, uint64_t> update_unlocalized_odom();
  bool registration_to_prior_map(Eigen::Vector3f &meas_p, Eigen::Quaternionf &meas_q,
                                 double &score, bool &converged);
  // Classic-dedicated clone of buildPriorMapRoiSubmap() — writes into
  // classic_submap_*/classic_roi_initialized_/classic_last_roi_center_ and
  // uses classic_engine_, so it never touches the buffers the concurrently-
  // running local keyframe pipeline's own async submap builder may be
  // writing into.
  bool buildClassicPriorMapRoiSubmap(const State &vehicle_state);
  double calc_scan_score();
  bool apply_corr_gate(const Eigen::Vector3f &meas_p, const Eigen::Quaternionf &meas_q,
                       double score, bool converged);
  void fuse_odom(const Eigen::Vector3f &meas_p, const Eigen::Quaternionf &meas_q);
  // (Re-)seed classic_kf_ from classic_seed_p_ + classic_seed_q_ — used both
  // on first bootstrap and after a fresh /initialpose click. Deliberately
  // NOT initial_position_/state.q — see classic_seed_p_/classic_seed_q_'s
  // comments in odom.h.
  void seedClassicKF();
  void classic_localization_routine();
  // Clone of publishToROS()/publishCloud() sourced from classic_kf_ instead
  // of lidarPose/T. Deliberately takes NO cloud parameters — published_cloud/
  // deskewed_scan from the scan callback are already in the LOCAL keyframe
  // pipeline's own frame (via T_prior), which is unrelated to classic_kf_'s
  // pose; using them here would silently double-transform (the same frame-
  // mismatch bug an earlier draft hit). Grabs latest_scan_ (raw sensor-frame)
  // itself instead. See src/c_lio/odom/odom_services.cc.
  void publishClassicToROS();

  void loadPriorMap();

  // Scan Context Relocalization (types in c_lio/scan_context.h)
  void buildScanContextDatabase();
  bool runRelocalization(pcl::PointCloud<PointType>::ConstPtr scan);

  // Keyframe Database (KFDB) — save real SC descriptors during mapping
  std::string getKfdbPath() const;
  void computeAndStoreKeyframeSC();
  bool loadKeyframeDatabase();

  // Runtime control services
  void srvSetMode(std::shared_ptr<c_lio::srv::SetMode::Request> req,
                  std::shared_ptr<c_lio::srv::SetMode::Response> res);
  void srvRelocalize(std::shared_ptr<c_lio::srv::Relocalize::Request> req,
                     std::shared_ptr<c_lio::srv::Relocalize::Response> res);
  void srvSetPose(std::shared_ptr<c_lio::srv::SetPose::Request> req,
                  std::shared_ptr<c_lio::srv::SetPose::Response> res);
  void srvGetState(std::shared_ptr<c_lio::srv::GetState::Request> req,
                   std::shared_ptr<c_lio::srv::GetState::Response> res);
  void srvNewMap(std::shared_ptr<c_lio::srv::NewMap::Request> req,
                 std::shared_ptr<c_lio::srv::NewMap::Response> res);
  void srvNewMapWZero(std::shared_ptr<c_lio::srv::NewMapWZero::Request> req,
                      std::shared_ptr<c_lio::srv::NewMapWZero::Response> res);
  void reloadPriorMapForRelocalization();
  void clearAllMapData();
  bool callSavePCD();
  bool callSaveCorrectedPCD();

  // Runtime mapping/localization helpers
  void applyModeTransition(const std::string &new_mode); // toggle timers/keyframing/target
  void publishMode();                                    // broadcast current mode (latched)
  // RViz "2D Pose Estimate": seed clicked pose as a relocalization guess; the
  // existing SC+GICP path refines it against the prior map and applies it.
  void callbackInitialPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  // Re-seed GTSAM preintegration nav state from the current state (pose+vel) so
  // the IMU thread predicts from the relocalized pose instead of a stale one.
  void resetGtsamNavState();
  // Continuous localization (map→odom TF correction)
  void continuousLocalize();
  bool verifyLoopWithG2O(int loop_kf_idx,
                         const Eigen::Matrix4f &T_map_body_gicp,
                         const Eigen::Matrix4f &T_odom_body,
                         const Eigen::Matrix4f &T_map_odom_current,
                         const std::vector<c_lio::AppearanceEntry> &sc_snap,
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
  rclcpp::Service<c_lio::srv::SetMode>::SharedPtr set_mode_srv_;
  rclcpp::Service<c_lio::srv::Relocalize>::SharedPtr relocalize_srv_;
  rclcpp::Service<c_lio::srv::SetPose>::SharedPtr set_pose_srv_;
  rclcpp::Service<c_lio::srv::GetState>::SharedPtr get_state_srv_;
  rclcpp::Service<c_lio::srv::NewMap>::SharedPtr new_map_srv_;
  rclcpp::Service<c_lio::srv::NewMapWZero>::SharedPtr new_map_w_zero_srv_;
  rclcpp::Client<c_lio::srv::SavePCD>::SharedPtr save_pcd_client_;
  rclcpp::Client<c_lio::srv::SavePCD>::SharedPtr save_corrected_pcd_client_;
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;
  std::mutex state_mtx_;

  // Publishers
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_2d_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr kf_pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kf_cloud_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_raw_pub;
  rclcpp::Publisher<c_lio::msg::KeyframeStamped>::SharedPtr kf_stamped_pub;

  // TF
  std::shared_ptr<tf2_ros::TransformBroadcaster> br;

  // ROS Msgs
  nav_msgs::msg::Odometry odom_ros;
  geometry_msgs::msg::PoseStamped pose_ros;
  nav_msgs::msg::Path path_ros;
  geometry_msgs::msg::PoseArray kf_pose_ros;

  // Flags
  std::atomic<bool> c_lio_initialized;
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
  c_lio::SensorType sensor;

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
  std::string submap_method_;        // effective method (may be forced to "prior_map" in localization)
  std::string submap_method_param_;  // configured value from params (restored when back to mapping)
  pcl::PointCloud<PointType>::ConstPtr submap_cloud;
  std::shared_ptr<const nano_gicp::CovarianceList> submap_normals;
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> submap_kdtree;
  std::unique_ptr<c_lio::VoxelHashMap> voxel_map_;
  int voxel_map_last_kf_idx_ = 0;

  // Prior-map ROI submap (localization): registration target = region of the
  // frozen map around the robot, extracted via prior_map_kdtree_ radiusSearch.
  // Rebuilt only when the robot moves > loc_roi_refresh_dist_ from last center.
  Eigen::Vector3f last_roi_center_ = Eigen::Vector3f::Zero();
  bool roi_initialized_ = false;
  double loc_roi_radius_ = 60.0;       // ROI radius around robot (m)
  double loc_roi_refresh_dist_ = 5.0;  // rebuild ROI after this much movement (m)

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
  // Registration engine (replaces gicp, ndt, ndt_cuda_, robust_icp_ + flags)
  c_lio::RegistrationEngine engine_;
  c_lio::RegistrationEngine engine_temp_; // for async submap building
  c_lio::PrefilterEngine prefilter_;
  c_lio::AppearanceEngine appearance_;           // loop closure descriptor (SC++, STD, etc.)
  c_lio::RegistrationEngine loc_registration_;   // continuous/submap localization
  c_lio::RegistrationEngine reloc_registration_; // SC relocalization (wider params)

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
  bool imu_gravity_removed_; // true if IMU driver already removed gravity from accel
  bool imu_accel_in_g_;      // true if IMU publishes linear_acceleration in g (will be scaled to m/s^2)

  bool adaptive_params_;

  double obs_submap_thresh_;
  double obs_keyframe_thresh_;
  double obs_keyframe_lag_;

  double keyframe_thresh_dist_;
  double keyframe_thresh_rot_;

  int submap_knn_;
  int submap_kcv_;
  int submap_kcc_;
  int submap_recent_n_;
  double submap_concave_alpha_;

  bool densemap_filtered_;
  bool wait_until_move_;

  double crop_size_;

  bool vf_use_;
  double vf_res_;
  bool gpu_preprocess_; // GPU deskewing + voxel filter

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

  // Fusion strategy
  c_lio::FusionMethod fusion_method_ = c_lio::FusionMethod::GEO;

  // Pose safety gate
  bool gate_enabled_ = false;
  double gate_fitness_threshold_ = 1.0;
  double gate_min_spaciousness_ = 0.5;
  double gate_max_translation_ = 5.0;
  double gate_max_rotation_deg_ = 45.0;
  int gate_max_consecutive_rejects_ = 10;
  int consecutive_gate_rejects_ = 0;
  bool last_gate_passed_ = true;
  float last_good_forward_speed_ = 0.0f;                   // last known forward speed when GICP was good
  Eigen::Vector3f prev_state_p_ = Eigen::Vector3f::Zero(); // previous state position for motion model pose filter
  Pose prev_lidarPose_;                                    // previous scan matching result for gate comparison
  LidarPoseTracker lidar_tracker_;                         // ABG prediction-based gate

  // KF state (15x15 covariance)
  Eigen::Matrix<float, 15, 15> kf_P_;
  bool kf_initialized_ = false;
  double kf_sigma_accel_ = 0.1;
  double kf_sigma_gyro_ = 0.01;
  double kf_sigma_accel_bias_ = 0.001;
  double kf_sigma_gyro_bias_ = 0.0001;
  double kf_sigma_pos_meas_ = 0.1;
  double kf_sigma_rot_meas_ = 0.02;

  // Error-State EKF
  c_lio::ErrorStateEkf ekf_;
  c_lio::EkfParams ekf_params_;

  // GTSAM IMU preintegration
  boost::shared_ptr<gtsam::PreintegrationParams> gtsam_imu_params_;
  boost::shared_ptr<gtsam::PreintegratedImuMeasurements> imu_preintegration_;
  gtsam::imuBias::ConstantBias gtsam_bias_;
  gtsam::NavState gtsam_nav_state_;
  bool gtsam_imu_initialized_ = false;

  // Motion model constraint
  c_lio::MotionModelType motion_model_type_ = c_lio::MotionModelType::NONE;
  struct MotionModelParams
  {
    // Ackermann
    float ack_max_fwd_vel = 30.f, ack_max_rev_vel = 5.f;
    float ack_max_lat_vel = 0.5f, ack_max_vert_vel = 1.f;
    float ack_max_fwd_accel = 8.f, ack_max_lat_accel = 3.f;
    float ack_max_yaw_rate = 1.5f, ack_max_roll_rate = 0.3f, ack_max_pitch_rate = 0.3f;
    // Diff drive
    float dd_max_fwd_vel = 5.f, dd_max_rev_vel = 2.f;
    float dd_max_lat_vel = 0.1f, dd_max_vert_vel = 0.5f;
    float dd_max_fwd_accel = 5.f, dd_max_lat_accel = 1.f;
    float dd_max_yaw_rate = 3.f, dd_max_roll_rate = 0.2f, dd_max_pitch_rate = 0.2f;
    // Holonomic
    float holo_max_horiz_vel = 5.f, holo_max_vert_vel = 0.5f;
    float holo_max_horiz_accel = 5.f;
    float holo_max_yaw_rate = 3.f, holo_max_roll_rate = 0.2f, holo_max_pitch_rate = 0.2f;
  };
  MotionModelParams mm_params_;

  // Debug: pure IMU velocity (no fusion correction)
  Eigen::Vector3f imu_only_vel_w_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f imu_only_vel_b_ = Eigen::Vector3f::Zero();

  bool debug_;
  bool debug_print_;
  bool deep_debug_ = false;
  bool use_2d_imu_ = false;
  bool use_imu_ = true;

  // IMU differential orientation mode
  bool imu_differential_orientation_ = false;
  std::string imu_preintegration_mode_ = "simple";
  Eigen::Quaternionf imu_prev_orientation_ = Eigen::Quaternionf::Identity();
  double imu_prev_orientation_stamp_ = 0.0;
  bool imu_prev_orientation_valid_ = false;
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
  // Orientation seed for the non-classic reloc path, set by callbackInitialPose()
  // from the /initialpose click's own orientation. Deliberately NOT written into
  // state.q directly: state.q is also written by propagateState() (200Hz, from
  // callbackImu()) under geo.mtx, a different mutex than the one guarding this
  // click handler (state_mtx_) — with no exclusion between the two, the IMU
  // thread's continuing dead-reckoning clobbers the click's orientation within
  // milliseconds (this is the exact bug already found and fixed for classic mode
  // via classic_seed_q_; see its comment below). runRelocalization() and the
  // exhausted-retries fallback in callbackPointCloud() both read this instead of
  // state.q so the click's orientation survives regardless of that race.
  // Protected by state_mtx_.
  Eigen::Quaternionf reloc_seed_q_ = Eigen::Quaternionf::Identity();
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
  std::vector<c_lio::AppearanceEntry> sc_database_;
  pcl::PointCloud<PointType>::Ptr prior_map_cloud_;
  std::shared_ptr<nanoflann::KdTreeFLANN<PointType>> prior_map_kdtree_;
  // Per-point covariances of prior_map_cloud_ (indices aligned), computed once
  // in loadPriorMap(). Sliced by radiusSearch indices to feed GICP ROI target.
  std::shared_ptr<const nano_gicp::CovarianceList> prior_map_covariances_;

  // Debug publish of the loaded frozen map (so RViz can show it for 2D Pose
  // Estimate). interval <= 0 disables. Published in the map frame.
  // The map is static, so we downsample ONCE to prior_map_debug_cloud_ and
  // republish that cheap cached cloud (avoids re-filtering a huge map + keeps
  // RViz light). prior_map_pub_leaf_ controls the debug resolution.
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr prior_map_pub_;
  rclcpp::TimerBase::SharedPtr prior_map_pub_timer_;
  int prior_map_pub_interval_ms_ = -1;
  double prior_map_pub_leaf_ = 1.0;
  pcl::PointCloud<PointType>::Ptr prior_map_debug_cloud_;
  void publishPriorMap();

  // Runtime mapping/localization switching: when false (localization), no new
  // keyframes are created (prevents unbounded memory growth).
  bool keyframing_enabled_ = true;

  // Localization memory bound: keep clouds+covariances only for the most recent
  // N live keyframes; free older ones (poses kept). The prior-map virtual
  // keyframes + recent live keyframes cover the KNN submap, so this caps memory
  // without changing odometry behaviour. <=0 disables (unbounded, mapping-style).
  int loc_keyframe_window_ = 100;

  // When true, relocalization refines ONLY around initial_position_ (an explicit
  // /initialpose click) and skips the global SC candidate search — so the user's
  // clicked pose is authoritative instead of being outvoted by SC matches.
  bool reloc_guess_only_ = false;

  // Set by /initialpose; consumed on the odometry thread to do a FRESH start:
  // drop live keyframes + zero velocity/observer, then relocalize around the
  // click. Prevents stale float-keyframes/submap from "flinging" the pose.
  std::atomic<bool> request_fresh_reloc_{false};

  // Dedicated callback groups so heavy/periodic timers run concurrently instead
  // of serializing in the node's default MutuallyExclusive group — that made
  // continuous-localize (and its match-score publish) fire far slower than its
  // 2s period because the 100Hz publish + submap-localize GICP + prior-map
  // publish all queued ahead of it. Shared state stays protected by its mutexes.
  rclcpp::CallbackGroup::SharedPtr continuous_localize_cb_group_;
  rclcpp::CallbackGroup::SharedPtr submap_loc_cb_group_;
  rclcpp::CallbackGroup::SharedPtr fast_pub_cb_group_;
  rclcpp::CallbackGroup::SharedPtr aux_timer_cb_group_;

  // KFDB entries accumulated during mapping
  std::vector<c_lio::AppearanceEntry> kfdb_entries_;
  std::mutex kfdb_mutex_;
  Eigen::Quaternionf kfdb_gravity_q_{1.f, 0.f, 0.f, 0.f}; // gravity quaternion for KFDB SC frame

  // Periodic KFDB auto-save (mapping mode) — so the .kfdb survives a hard kill,
  // not only a clean Ctrl+C shutdown. Writes are atomic (temp + rename).
  rclcpp::TimerBase::SharedPtr kfdb_autosave_timer_;
  double kfdb_auto_save_interval_ = 30.0; // seconds (0 = disabled)
  std::mutex kfdb_file_mutex_;            // serializes KFDB disk writes (timer / service / shutdown)

  // Corrected keyframe poses from lio_sam_opt (for corrected KFDB)
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
  // classic mode's registration input. Deliberately NOT latest_scan_
  // (=original_scan): that is the RAW, un-deskewed sweep — a single point
  // cloud with no per-point motion compensation. The default pipeline never
  // registers that directly; it registers current_scan, which
  // deskewPointcloud() builds by transforming each point with its own
  // per-point-time-interpolated IMU pose (frames[i]) before matching against
  // the submap. Feeding classic's GICP the raw sweep instead introduces a
  // velocity-dependent shear each scan — worse when moving faster — which
  // GICP "corrects" for inconsistently tick to tick, and fuse_odom() then
  // blends that noise in (this is why disabling registration/fuse made the
  // jitter disappear entirely: the distorted-scan path was simply never
  // exercised).
  //
  // Computed as T_prior.inverse() * current_scan: current_scan is already
  // per-point deskewed into WORLD frame via (frames[i] * baselink2lidar_T)
  // per point, and T_prior == frames[median_pt_index] by construction (see
  // deskewPointcloud(), odom_callbacks.cc) — so this bulk un-transform
  // exactly recovers a body-frame cloud with the intra-sweep motion
  // compensation already baked in, WITHOUT reusing current_scan directly
  // (which would still carry the LOCAL pipeline's own T_prior baked into its
  // absolute frame — the same class of frame-mismatch bug this project hit
  // once already). classic then applies its OWN T_predicted on top, exactly
  // like it already does for latest_scan_.
  pcl::PointCloud<PointType>::ConstPtr latest_scan_deskewed_body_;
  // classic mode Case A's predict source. Deliberately NOT latest_scan_T_:
  // this->T (odom_registration.cc: T = T_corr * T_prior) is the RAW
  // scan-to-local-submap GICP correction, captured BEFORE updateState()'s
  // KF/EKF fuses it into state.p/state.q — i.e. before the same smoothing
  // that "keyframe" mode's own published pose (state.p/q, see
  // publishPose()) actually benefits from. Predicting off latest_scan_T_
  // instead would feed classic_kf_ a noisier signal than what "keyframe"
  // mode publishes for the exact same input, producing visible extra jitter
  // that has nothing to do with classic's own registration/fuse layer.
  // Captured under geo.mtx (state's own convention), then stored here
  // alongside latest_scan_T_ under latest_scan_mtx_ — see odom_callbacks.cc.
  Eigen::Vector3f latest_scan_state_p_;
  Eigen::Quaternionf latest_scan_state_q_;
  double latest_scan_time_;                          // wall time when scan was stored
  // Plain monotonic counter, incremented alongside latest_scan_T_/
  // latest_scan_time_ above — used by classic mode's staleness check instead
  // of latest_scan_time_, since that relies on this->now() (ROS/wall clock),
  // which was observed frozen in at least one deployment environment
  // (use_sim_time without a working /clock source), silently breaking any
  // clock-based comparison (including RCLCPP_*_THROTTLE macros, which use
  // the same clock internally).
  uint64_t latest_scan_seq_ = 0;
  std::mutex latest_scan_mtx_;

  // Confidence publisher + global correction control
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr confidence_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr imu_debug_markers_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr global_correction_sub_;
  std::atomic<bool> enable_global_correction_{true};
  float last_confidence_{0.0f};

  // Per-scan scan↔map registration score (raw ICP/GICP fitness; lower = better
  // match, grows when the scan stops matching the map). One value per scan.
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr scan_match_pub_;

  // Runtime mode broadcast (latched) — lio_sam_opt subscribes to mirror mode.
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mode_pub_;
  // RViz "2D Pose Estimate" → guess refined against the prior map, then applied.
  // Dedicated callback group so it is never starved by the heavy localization
  // timers (continuous/submap localize) that share the default group.
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub_;
  rclcpp::CallbackGroup::SharedPtr initialpose_cb_group_;

  // 2D odometry publish
  bool publish_2d_odom_enabled_ = false;
  double odom_2d_pose_cov_xy_  = 1e-2;
  double odom_2d_pose_cov_yaw_ = 1e-2;
  double odom_2d_twist_cov_lin_ = 1e-2;
  double odom_2d_twist_cov_ang_ = 1e-2;

  // Occupancy grid
  bool occupancy_grid_enabled_ = false;
  std::unique_ptr<c_lio::OccupancyGridGenerator> occupancy_grid_gen_;
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

  // GPS yaw calibration (align odom heading to GPS heading)
  bool gps_yaw_calibrated_ = false;
  bool gps_yaw_first_fix_stored_ = false;
  Eigen::Vector3f gps_yaw_first_pos_ = Eigen::Vector3f::Zero();  // ENU position
  Eigen::Vector3f gps_yaw_first_odom_ = Eigen::Vector3f::Zero(); // odom position at that time
  double gps_yaw_min_dist_ = 10.0;                               // min travel distance for reliable heading

  // GPS coordinate converter
  std::unique_ptr<GeographicLib::LocalCartesian> gps_converter_;
  bool gps_origin_set_ = false;
  double gps_origin_lat_ = 0.0;
  double gps_origin_lon_ = 0.0;
  double gps_origin_alt_ = 0.0;

  // External velocity source (wheel encoder / INS twist)
  bool ext_odom_enabled_ = false;
  std::string ext_odom_topic_;
  Eigen::Vector3f ext_odom_scale_ = Eigen::Vector3f::Ones();
  struct
  {
    Eigen::Vector3f t = Eigen::Vector3f::Zero();
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  } ext_baselink2odom_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr ext_odom_sub_;
  rclcpp::CallbackGroup::SharedPtr ext_odom_cb_group_;
  Eigen::Vector3f ext_odom_vel_body_ = Eigen::Vector3f::Zero();
  std::mutex ext_odom_mtx_;
  std::atomic<bool> ext_odom_received_{false};
  double ext_odom_stamp_ = 0.0;

  // Classic localization sub-mode (odom/submap/method == "classic")
  rclcpp::TimerBase::SharedPtr classic_localization_timer_;
  rclcpp::CallbackGroup::SharedPtr classic_cb_group_;
  double classic_routine_rate_hz_ = 10.0;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr unlocalized_odom_sub_;
  rclcpp::CallbackGroup::SharedPtr unlocalized_odom_cb_group_;
  std::string classic_unlocalized_odom_topic_ = "";
  std::mutex classic_unlocalized_odom_mtx_;
  nav_msgs::msg::Odometry::SharedPtr classic_latest_unlocalized_odom_;
  // Plain monotonic counter, incremented in callbackUnlocalizedOdom() each
  // time a new message arrives — see latest_scan_seq_'s comment for why this
  // is used instead of a clock-derived timestamp (Case B equivalent).
  uint64_t classic_unlocalized_odom_seq_ = 0;

  // "cls_unlocalized_odom" / prev — raw odom input (Case A: local keyframe
  // pipeline's this->T via latest_scan_T_; Case B: unlocalized_odom topic),
  // BEFORE fusion. Differenced each tick to drive classic_kf_.predict().
  Eigen::Vector3f cls_prev_unlocalized_odom_p_ = Eigen::Vector3f::Zero();
  Eigen::Quaternionf cls_prev_unlocalized_odom_q_ = Eigen::Quaternionf::Identity();
  // Last source sequence number actually consumed — classic_localization_
  // routine() skips a tick entirely (no predict, no register) if
  // update_unlocalized_odom()'s source hasn't produced anything newer than
  // this, since classic_localization_timer_ runs decoupled from the scan/
  // topic rate. A plain counter (latest_scan_seq_/classic_unlocalized_odom_
  // seq_), NOT a clock-derived timestamp — this->now()/get_clock()->now()
  // were observed frozen in at least one deployment environment
  // (use_sim_time without a working /clock source), which would otherwise
  // make this check (and RCLCPP_*_THROTTLE logging) silently never fire.
  uint64_t classic_last_source_seq_used_ = 0;

  // "cls_odom_filtered" == classic_kf_.position()/orientation() — the final,
  // fused output. Protected by classic_state_mtx_ since classic_localization_
  // timer_ (writer) and publishPose()/publishClassicToROS() (readers, called
  // from different callback groups) can run concurrently.
  std::mutex classic_state_mtx_;
  c_lio::ClassicModeKF classic_kf_;
  c_lio::ClassicModeKFParams classic_kf_params_;
  bool classic_kf_initialized_ = false;
  // Orientation to seed classic_kf_ from — set to AngleAxisf(initial_yaw_,
  // UnitZ()) once at param-load time (getParams()), and OVERWRITTEN by
  // callbackInitialPose() with the /initialpose click's own orientation.
  // Deliberately NOT read from state.q: callbackInitialPose() writes
  // state.q under state_mtx_, but propagateState() (200Hz, from
  // callbackImu()) ALSO writes state.q continuously under geo.mtx — a
  // different mutex that provides no exclusion between the two. Without a
  // dedicated variable, propagateState()'s continuing dead-reckoning
  // overwrites the click's orientation within milliseconds, before
  // classic_localization_routine()'s next tick ever reads it (this is what
  // caused the click's position to apply but not its orientation).
  // Protected by classic_state_mtx_.
  Eigen::Quaternionf classic_seed_q_ = Eigen::Quaternionf::Identity();
  // Position to seed classic_kf_ from — same rationale as classic_seed_q_,
  // but for position: deliberately NOT initial_position_. initial_position_
  // is shared with the OLD keyframe pipeline (state.p bootstrap, relocalize
  // recovery, etc in odom_callbacks.cc/odom_relocalization.cc) — writing it
  // from an /initialpose click while classic mode is active would perturb
  // the old pipeline's own internal state, which Case A (no external
  // unlocalized_odom_topic) relies on as its map-agnostic local odom source.
  // Set to initial_position_'s startup value once at param-load time
  // (getParams()), then OVERWRITTEN only by callbackInitialPose() — never by
  // anything belonging to the old pipeline. Protected by classic_state_mtx_.
  Eigen::Vector3f classic_seed_p_ = Eigen::Vector3f::Zero();
  // Incremented as the very FIRST line of classic_localization_routine(),
  // before any guard/early-return — an unconditional tick counter to verify
  // the timer is actually firing repeatedly, independent of whether any
  // throttled log inside the routine itself is visible/being seen.
  std::atomic<uint64_t> classic_tick_count_{0};

  // Gate thresholds (mirror gate_fitness_threshold_/gate_max_translation_/
  // gate_max_rotation_deg_ — see apply_corr_gate())
  double classic_max_corr_scan_score_threshold_ = 1.0;
  double classic_max_corr_translation_ = 5.0;
  double classic_max_corr_rotation_deg_ = 45.0;

  // Dedicated registration resources for classic's OWN frozen-map ROI
  // registration — MUST be separate from engine_/engine_temp_/submap_*/
  // roi_initialized_/last_roi_center_, which the concurrently-active local
  // keyframe pipeline (Case A) and its async submap-build thread also use.
  c_lio::RegistrationEngine classic_engine_;
  pcl::PointCloud<PointType>::ConstPtr classic_submap_cloud_;
  std::shared_ptr<const nano_gicp::CovarianceList> classic_submap_normals_;
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> classic_submap_kdtree_;
  std::atomic<bool> classic_submap_hasChanged_{false};
  Eigen::Vector3f classic_last_roi_center_ = Eigen::Vector3f::Zero();
  bool classic_roi_initialized_ = false;
  // Dedicated voxel filter for registration_to_prior_map()'s scan downsample.
  // MUST NOT reuse the shared `voxel` member: that one is mutated
  // (setInputCloud+filter, no mutex) by preprocessPoints() on the scan-
  // callback thread (lidar_cb_group) every scan, while
  // registration_to_prior_map() runs on classic_localization_timer_'s own
  // thread (classic_cb_group_ — a separate MutuallyExclusive group, so it
  // genuinely runs concurrently with the scan callback under the
  // multi-threaded container). Two threads calling setInputCloud()/filter()
  // on the SAME pcl::VoxelGrid instance is a data race — pcl::VoxelGrid is
  // not reentrant/thread-safe — and was silently corrupting the scan fed to
  // classic's GICP, producing inconsistent per-tick corrections (visible as
  // jitter whenever the correction/fuse step was enabled, gone when it was
  // disabled since the race was then never triggered).
  pcl::VoxelGrid<PointType> classic_voxel_;

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
