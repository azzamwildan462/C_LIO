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

#include "dlio/odom/odom.h"
#include "dlio/odom/utils.h"
#include "dlio/odom/kfdb_io.h"

#include <queue>
#include <filesystem>

#include "rclcpp/qos.hpp"

// Global pointer for KFDB atexit handler
std::atomic<dlio::OdomNode *> g_odom_node{nullptr};

static void odomAtexitSave()
{
  auto *node = g_odom_node.exchange(nullptr);
  if (node)
  {
    node->saveKeyframeDatabase();
    node->saveCorrectedKeyframeDatabase();
  }
}

dlio::OdomNode::OdomNode(const rclcpp::NodeOptions &options)
    : Node("dlio_odom_node", options)
{

  this->getParams();

  RCLCPP_INFO(this->get_logger(), "[odom] Constructor started, map_path='%s', relocalize=%d, map_mode='%s'",
              this->map_path_.c_str(), static_cast<int>(this->relocalize_), this->map_mode_.c_str());

  this->num_threads_ = omp_get_max_threads();

  this->dlio_initialized = false;
  this->first_valid_scan = false;
  this->first_imu_received = false;
  if (this->imu_calibrate_)
  {
    this->imu_calibrated = false;
  }
  else
  {
    this->imu_calibrated = true;
  }
  this->deskew_status = false;
  this->deskew_size = 0;

  this->lidar_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto lidar_sub_opt = rclcpp::SubscriptionOptions();
  lidar_sub_opt.callback_group = this->lidar_cb_group;
  this->lidar_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("pointcloud", rclcpp::SensorDataQoS(),
                                                                             std::bind(&dlio::OdomNode::callbackPointCloud, this, std::placeholders::_1), lidar_sub_opt);

  this->imu_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto imu_sub_opt = rclcpp::SubscriptionOptions();
  imu_sub_opt.callback_group = this->imu_cb_group;
  this->imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", rclcpp::SensorDataQoS(),
                                                                   std::bind(&dlio::OdomNode::callbackImu, this, std::placeholders::_1), imu_sub_opt);

  this->odom_pub = this->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  this->pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 1);
  this->path_pub = this->create_publisher<nav_msgs::msg::Path>("path", 1);
  this->kf_pose_pub = this->create_publisher<geometry_msgs::msg::PoseArray>("kf_pose", 1);
  this->kf_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("kf_cloud", 10);
  this->deskewed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed", 1);
  this->deskewed_raw_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed_raw", 1);
  this->kf_stamped_pub = this->create_publisher<direct_lidar_inertial_odometry::msg::KeyframeStamped>("kf_stamped", 10);

  this->br = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

  this->publish_timer = this->create_wall_timer(std::chrono::duration<double>(0.01),
                                                std::bind(&dlio::OdomNode::publishPose, this));

  // Occupancy grid publisher + timer
  if (this->occupancy_grid_enabled_)
  {
    this->occupancy_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
        "occupancy_grid", rclcpp::QoS(5));

    double og_rate = 5.0;
    dlio::declare_param(this, "occupancy_grid/update_rate", og_rate, 5.0);

    this->occupancy_grid_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / og_rate),
        std::bind(&dlio::OdomNode::publishOccupancyGrid, this));

    RCLCPP_INFO(this->get_logger(), "Occupancy grid enabled: rate=%.1fHz", og_rate);
  }

  // GPS subscriber
  if (this->gps_enabled_)
  {
    this->gps_cb_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    auto gps_sub_opt = rclcpp::SubscriptionOptions();
    gps_sub_opt.callback_group = this->gps_cb_group_;
    this->gps_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
        this->gps_topic_, rclcpp::SensorDataQoS(),
        std::bind(&dlio::OdomNode::callbackGPS, this, std::placeholders::_1),
        gps_sub_opt);

    this->static_tf_broadcaster_ =
        std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    // If origin param is set, initialize converter immediately
    if (this->gps_origin_param_lat_ != 0.0 || this->gps_origin_param_lon_ != 0.0)
    {
      this->gps_origin_lat_ = this->gps_origin_param_lat_;
      this->gps_origin_lon_ = this->gps_origin_param_lon_;
      this->gps_origin_alt_ = this->gps_origin_param_alt_;
      this->gps_converter_ = std::make_unique<GeographicLib::LocalCartesian>(
          this->gps_origin_lat_, this->gps_origin_lon_, this->gps_origin_alt_);
      this->gps_origin_set_ = true;
    }

    RCLCPP_INFO(this->get_logger(), "GPS enabled: topic='%s', origin=[%.6f,%.6f,%.1f]%s",
                this->gps_topic_.c_str(), this->gps_origin_param_lat_,
                this->gps_origin_param_lon_, this->gps_origin_param_alt_,
                this->gps_origin_set_ ? " (from param)" : " (will use first fix)");
  }

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();

  this->origin = Eigen::Vector3f(0., 0., 0.);
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.;
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.;
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_);
  this->first_imu_stamp = 0.;
  this->prev_imu_stamp = 0.;

  this->original_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->current_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->submap_cloud = std::make_shared<const pcl::PointCloud<PointType>>();

  this->num_processed_keyframes = 0;

  this->submap_hasChanged = true;
  this->submap_kf_idx_prev.clear();
  this->new_submap_is_ready = false;
  this->main_loop_running = false;

  this->first_scan_stamp = 0.;
  this->elapsed_time = 0.;
  this->length_traversed;

  this->convex_hull.setDimension(3);
  this->concave_hull.setDimension(3);
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
  this->concave_hull.setKeepInformation(true);

  // Initialize registration engine
  {
    double kernel_scale = 0.0;
    if (this->registration_method_ == "robust_icp" || this->registration_method_ == "robust_icp_cuda")
      dlio::declare_param(this, "odom/robust_icp/kernel_scale", kernel_scale, 0.0);

    dlio::RegistrationParams rp;
    rp.gicp_k_correspondences = this->gicp_k_correspondences_;
    rp.gicp_max_corr_dist = this->gicp_max_corr_dist_;
    rp.gicp_max_iter = this->gicp_max_iter_;
    rp.gicp_transformation_ep = this->gicp_transformation_ep_;
    rp.gicp_rotation_ep = this->gicp_rotation_ep_;
    rp.gicp_init_lambda_factor = this->gicp_init_lambda_factor_;
    rp.gicp_gpu_voxel_size = this->vf_res_ > 0.0 ? static_cast<float>(this->vf_res_) : 0.5f;
    rp.ndt_resolution = static_cast<float>(this->ndt_resolution_);
    rp.ndt_num_threads = this->ndt_num_threads_;
    rp.ndt_search_method = this->ndt_search_method_;
    rp.ndt_step_size = this->ndt_step_size_;
    rp.robust_icp_kernel_scale = kernel_scale;

    auto method = dlio::parseRegistrationMethod(this->registration_method_);
    this->engine_.init(method, rp, this->get_logger());
    this->engine_temp_.init(method, rp, this->get_logger());

    RCLCPP_INFO(this->get_logger(), "[odom] Registration method: %s",
                dlio::registrationMethodToString(method).c_str());
  }

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
  this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

  this->voxel.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);

  // Reserve capacity for grow-only vectors to prevent reallocation during concurrent access
  // (these are pushed from callbacks and read from other threads)
  this->comp_times.reserve(100000);
  this->lidar_rates.reserve(100000);
  this->imu_rates.reserve(500000);
  this->cpu_percents.reserve(100000);
  this->metrics.spaciousness.reserve(100000);
  this->metrics.density.reserve(100000);
  this->trajectory.reserve(100000);
  this->path_ros.poses.reserve(100000);
  this->kf_pose_ros.poses.reserve(100000);

  this->metrics.spaciousness.push_back(0.);
  this->metrics.density.push_back(this->gicp_max_corr_dist_);

  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

#ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0, 0, 0, 0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i)
  {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
#endif

  FILE *file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while (fgets(line, 128, file) != nullptr)
  {
    if (strncmp(line, "processor", 9) == 0)
      this->numProcessors++;
  }
  fclose(file);

  // Runtime control services
  this->service_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->set_mode_srv_ = this->create_service<direct_lidar_inertial_odometry::srv::SetMode>(
      "dlio_odom/set_mode",
      std::bind(&dlio::OdomNode::srvSetMode, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, this->service_cb_group_);
  this->relocalize_srv_ = this->create_service<direct_lidar_inertial_odometry::srv::Relocalize>(
      "dlio_odom/relocalize",
      std::bind(&dlio::OdomNode::srvRelocalize, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, this->service_cb_group_);
  this->set_pose_srv_ = this->create_service<direct_lidar_inertial_odometry::srv::SetPose>(
      "dlio_odom/set_pose",
      std::bind(&dlio::OdomNode::srvSetPose, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, this->service_cb_group_);
  this->get_state_srv_ = this->create_service<direct_lidar_inertial_odometry::srv::GetState>(
      "dlio_odom/get_state",
      std::bind(&dlio::OdomNode::srvGetState, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, this->service_cb_group_);
  this->new_map_srv_ = this->create_service<direct_lidar_inertial_odometry::srv::NewMap>(
      "dlio_odom/new_map",
      std::bind(&dlio::OdomNode::srvNewMap, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, this->service_cb_group_);
  this->new_map_w_zero_srv_ = this->create_service<direct_lidar_inertial_odometry::srv::NewMapWZero>(
      "dlio_odom/new_map_w_zero",
      std::bind(&dlio::OdomNode::srvNewMapWZero, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, this->service_cb_group_);

  // SavePCD clients (calls MapNode + GraphSlam save services)
  this->save_pcd_client_ = this->create_client<direct_lidar_inertial_odometry::srv::SavePCD>(
      "save_pcd_map", rmw_qos_profile_services_default, this->service_cb_group_);
  this->save_corrected_pcd_client_ = this->create_client<direct_lidar_inertial_odometry::srv::SavePCD>(
      "save_corrected_pcd", rmw_qos_profile_services_default, this->service_cb_group_);

  // Subscribe to corrected keyframe poses from lio_sam_opt
  this->corrected_kf_poses_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
      "corrected_kf_poses", 10,
      [this](const geometry_msgs::msg::PoseArray::SharedPtr msg)
      {
        std::lock_guard<std::mutex> lock(this->corrected_kf_poses_mutex_);
        this->corrected_kf_poses_ = msg->poses;
      });

  // Map load
  this->prior_map_pose_set_ = false;
  this->num_prior_keyframes_ = 0;
  this->use_prior_map_ = false;
  this->relocalized_ = false;
  this->sc_attempt_count_ = 0;
  this->last_reloc_fitness_ = -1.0;

  if (!this->map_path_.empty())
  {
    // Check if file exists
    std::ifstream f(this->map_path_);
    if (f.good())
    {
      f.close();
      this->use_prior_map_ = true;
      this->loadPriorMap();
    }
    else
    {
      RCLCPP_INFO(this->get_logger(), "No existing map at '%s', starting fresh",
                  this->map_path_.c_str());
    }
  }

  // Continuous localization init (Bayesian)
  this->T_map_odom_ = Eigen::Matrix4f::Identity();
  this->latest_scan_T_ = Eigen::Matrix4f::Identity();
  this->latest_scan_time_ = 0.0;
  this->bayes_consecutive_accepts_ = 0;
  this->bayes_posterior_.clear();

  RCLCPP_INFO(this->get_logger(), "[dlio_odom] map/tf_source='%s', map_mode='%s'",
              this->tf_map_odom_source_.c_str(), this->map_mode_.c_str());

  bool cl_enabled = this->continuous_localize_ && this->use_prior_map_ &&
                    (this->map_mode_ == "localization" ||
                     (this->map_mode_ == "mapping" && this->continuous_localize_on_mapping_));
  if (cl_enabled)
  {
    // In mapping mode with prior map, mark position as known so continuousLocalize() runs
    if (this->map_mode_ == "mapping")
      this->relocalized_ = true;

    this->continuous_localize_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(this->continuous_localize_interval_),
        std::bind(&dlio::OdomNode::continuousLocalize, this));

    // Publish confidence every tick so other nodes know alignment quality
    this->confidence_pub_ = this->create_publisher<std_msgs::msg::Float32>("localization_confidence", 10);

    // Subscriber to enable/disable global (Stage 2) correction
    this->global_correction_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "enable_global_correction", 10,
        [this](const std_msgs::msg::Bool::SharedPtr msg)
        {
          bool prev = this->enable_global_correction_.load();
          this->enable_global_correction_.store(msg->data);
          if (prev != msg->data)
          {
            RCLCPP_INFO(this->get_logger(), "[bayes] global correction %s",
                        msg->data ? "ENABLED" : "DISABLED (drift-only mode)");
            if (!msg->data)
            {
              // Reset consecutive when disabling — prevent stale reloc state
              this->bayes_consecutive_accepts_ = 0;
            }
          }
        });

    bool enable_global_default = true;
    dlio::declare_param(this, "map/continuous_localize/enable_global_correction",
                        enable_global_default, true);
    this->enable_global_correction_.store(enable_global_default);

    RCLCPP_INFO(this->get_logger(), "Continuous localization enabled (%s mode): interval=%.1fs, fitness_thresh=%.2f, global_corr=%s",
                this->map_mode_.c_str(),
                this->continuous_localize_interval_, this->continuous_localize_fitness_thresh_,
                enable_global_default ? "on" : "off");
  }

  // Submap-based relocalization (GPS-denied alternative)
  this->submap_loc_state_ = SubmapLocState::IDLE;
  if (this->submap_loc_enabled_ && this->use_prior_map_ && this->map_mode_ == "localization")
  {
    this->initSubmapLocalization();
    this->submap_loc_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(this->submap_loc_interval_),
        std::bind(&dlio::OdomNode::submapLocalizeTick, this));
    if (!this->confidence_pub_)
      this->confidence_pub_ = this->create_publisher<std_msgs::msg::Float32>("localization_confidence", 10);
    RCLCPP_INFO(this->get_logger(), "Submap relocalization enabled: interval=%.1fs, group_size=%d, search_radius=%.1fm",
                this->submap_loc_interval_, this->submap_loc_group_size_, this->submap_loc_search_radius_);
  }

  // Register atexit handler for KFDB saving (mapping mode)
  if (this->map_mode_ == "mapping" && !this->map_path_.empty())
  {
    g_odom_node.store(this);
    std::atexit(odomAtexitSave);
  }
}

dlio::OdomNode::~OdomNode()
{
  if (this->map_mode_ == "mapping" && !this->map_path_.empty())
  {
    this->saveKeyframeDatabase();
    this->saveCorrectedKeyframeDatabase();
  }
}

void dlio::OdomNode::getParams()
{

  // Debug
  dlio::declare_param(this, "debug/odom", this->debug_, false);
  dlio::declare_param(this, "debug/print_odom", this->debug_print_, false);
  dlio::declare_param(this, "deep_debug/odom", this->deep_debug_, false);
  dlio::declare_param(this, "odom/use_2d_imu", this->use_2d_imu_, false);
  dlio::declare_param(this, "odom/use_imu", this->use_imu_, true);

  // Version
  dlio::declare_param(this, "version", this->version_, "0.0.0");

  // Frames
  dlio::declare_param(this, "frames/odom", this->odom_frame, "odom");
  dlio::declare_param(this, "frames/baselink", this->baselink_frame, "base_link");
  dlio::declare_param(this, "frames/lidar", this->lidar_frame, "lidar");
  dlio::declare_param(this, "frames/imu", this->imu_frame, "imu");

  // Deskew Flag
  dlio::declare_param(this, "pointcloud/deskew", this->deskew_, true);

  // Gravity
  dlio::declare_param(this, "odom/gravity", this->gravity_, 9.80665);

  // Compute time offset between lidar and imu
  dlio::declare_param(this, "odom/computeTimeOffset", this->time_offset_, false);

  // Keyframe Threshold
  dlio::declare_param(this, "odom/keyframe/threshD", this->keyframe_thresh_dist_, 0.1);
  dlio::declare_param(this, "odom/keyframe/threshR", this->keyframe_thresh_rot_, 1.0);

  // Submap
  dlio::declare_param(this, "odom/submap/method", this->submap_method_, std::string("keyframe"));
  dlio::declare_param(this, "odom/submap/keyframe/knn", this->submap_knn_, 10);
  dlio::declare_param(this, "odom/submap/keyframe/kcv", this->submap_kcv_, 10);
  dlio::declare_param(this, "odom/submap/keyframe/kcc", this->submap_kcc_, 10);

  if (this->submap_method_ == "voxel_hash_map")
  {
    double vm_voxel_size = 1.0, vm_max_dist = 100.0;
    int vm_max_pts = 20;
    dlio::declare_param(this, "odom/submap/voxel_hash_map/voxel_size", vm_voxel_size, 1.0);
    dlio::declare_param(this, "odom/submap/voxel_hash_map/max_distance", vm_max_dist, 100.0);
    dlio::declare_param(this, "odom/submap/voxel_hash_map/max_points_per_voxel", vm_max_pts, 20);

    if (this->engine_.needsCovariances())
    {
      RCLCPP_WARN(this->get_logger(),
                  "[odom] %s incompatible with voxel_hash_map (no covariances), falling back to keyframe submap",
                  this->registration_method_.c_str());
      this->submap_method_ = "keyframe";
    }
    else
    {
      this->voxel_map_ = std::make_unique<dlio::VoxelHashMap>(vm_voxel_size, vm_max_dist, vm_max_pts);
      RCLCPP_INFO(this->get_logger(),
                  "[odom] Using voxel hash map submap (voxel=%.2fm, max_dist=%.0fm, max_pts=%d)",
                  vm_voxel_size, vm_max_dist, vm_max_pts);
    }
  }

  // Dense map resolution
  dlio::declare_param(this, "map/dense/filtered", this->densemap_filtered_, true);

  // Wait until movement to publish map
  dlio::declare_param(this, "map/waitUntilMove", this->wait_until_move_, false);

  // Crop Box Filter
  dlio::declare_param(this, "odom/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

  // Voxel Grid Filter
  dlio::declare_param(this, "pointcloud/voxelize", this->vf_use_, true);
  dlio::declare_param(this, "odom/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

  // GPU preprocessing (deskewing + voxel filter)
  dlio::declare_param(this, "odom/preprocessing/gpu", this->gpu_preprocess_, false);
#if !DLIO_HAS_CUDA
  if (this->gpu_preprocess_) {
    RCLCPP_WARN(this->get_logger(), "GPU preprocessing requested but CUDA not available, falling back to CPU");
    this->gpu_preprocess_ = false;
  }
#endif
  if (this->gpu_preprocess_)
    RCLCPP_INFO(this->get_logger(), "[odom] GPU preprocessing enabled (deskew + voxel filter)");

  // Prefilter (noise removal)
  {
    std::string prefilter_method;
    dlio::declare_param(this, "odom/prefilter/method", prefilter_method, std::string("none"));
    dlio::PrefilterParams pf_params;
    dlio::declare_param(this, "odom/prefilter/mean_k", pf_params.mean_k, 10);
    dlio::declare_param(this, "odom/prefilter/stddev_mul", pf_params.stddev_mul, 1.0);
    dlio::declare_param(this, "odom/prefilter/radius", pf_params.radius, 0.5);
    dlio::declare_param(this, "odom/prefilter/min_neighbors", pf_params.min_neighbors, 3);
    dlio::declare_param(this, "odom/prefilter/dror_min_radius", pf_params.dror_min_radius, 0.1);
    dlio::declare_param(this, "odom/prefilter/dror_max_radius", pf_params.dror_max_radius, 2.0);
    dlio::declare_param(this, "odom/prefilter/dror_range_scale", pf_params.dror_range_scale, 0.01);
    dlio::declare_param(this, "odom/prefilter/dror_min_neighbors", pf_params.dror_min_neighbors, 3);
    this->prefilter_.init(dlio::parsePrefilterMethod(prefilter_method), pf_params);
    if (this->prefilter_.isEnabled())
      RCLCPP_INFO(this->get_logger(), "[odom] Prefilter: %s", prefilter_method.c_str());
  }

  // Adaptive Parameters
  dlio::declare_param(this, "adaptive", this->adaptive_params_, true);

  // Extrinsics
  std::vector<double> t_default{0., 0., 0.};
  std::vector<double> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<double> rpy_default{0., 0., 0.};

  // Helper: RPY (degrees) → rotation matrix
  auto rpyToRotation = [](double roll_deg, double pitch_deg, double yaw_deg) -> Eigen::Matrix3f
  {
    float r = roll_deg * M_PI / 180.0;
    float p = pitch_deg * M_PI / 180.0;
    float y = yaw_deg * M_PI / 180.0;
    Eigen::Matrix3f R;
    R = Eigen::AngleAxisf(y, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(p, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(r, Eigen::Vector3f::UnitX());
    return R;
  };

  // center of gravity to imu
  std::vector<double> baselink2imu_t, baselink2imu_R, baselink2imu_rpy;
  dlio::declare_param(this, "extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  dlio::declare_param(this, "extrinsics/baselink2imu/rpy", baselink2imu_rpy, rpy_default);
  dlio::declare_param(this, "extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
      Eigen::Vector3f(baselink2imu_t[0], baselink2imu_t[1], baselink2imu_t[2]);
  if (baselink2imu_rpy != rpy_default)
  {
    // Use RPY (degrees) if specified
    this->extrinsics.baselink2imu.R = rpyToRotation(baselink2imu_rpy[0], baselink2imu_rpy[1], baselink2imu_rpy[2]);
    RCLCPP_INFO(this->get_logger(), "[odom] baselink2imu: rpy=[%.1f, %.1f, %.1f] deg",
                baselink2imu_rpy[0], baselink2imu_rpy[1], baselink2imu_rpy[2]);
  }
  else
  {
    // Fallback to rotation matrix
    this->extrinsics.baselink2imu.R =
        Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2imu_R.begin(), baselink2imu_R.end()).data(), 3, 3);
  }
  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) = this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) = this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<double> baselink2lidar_t, baselink2lidar_R, baselink2lidar_rpy;
  dlio::declare_param(this, "extrinsics/baselink2lidar/t", baselink2lidar_t, t_default);
  dlio::declare_param(this, "extrinsics/baselink2lidar/rpy", baselink2lidar_rpy, rpy_default);
  dlio::declare_param(this, "extrinsics/baselink2lidar/R", baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t =
      Eigen::Vector3f(baselink2lidar_t[0], baselink2lidar_t[1], baselink2lidar_t[2]);
  if (baselink2lidar_rpy != rpy_default)
  {
    // Use RPY (degrees) if specified
    this->extrinsics.baselink2lidar.R = rpyToRotation(baselink2lidar_rpy[0], baselink2lidar_rpy[1], baselink2lidar_rpy[2]);
    RCLCPP_INFO(this->get_logger(), "[odom] baselink2lidar: rpy=[%.1f, %.1f, %.1f] deg",
                baselink2lidar_rpy[0], baselink2lidar_rpy[1], baselink2lidar_rpy[2]);
  }
  else
  {
    // Fallback to rotation matrix
    this->extrinsics.baselink2lidar.R =
        Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2lidar_R.begin(), baselink2lidar_R.end()).data(), 3, 3);
  }

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) = this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) = this->extrinsics.baselink2lidar.R;

  // IMU
  dlio::declare_param(this, "odom/imu/calibration/accel", this->calibrate_accel_, true);
  dlio::declare_param(this, "odom/imu/calibration/gyro", this->calibrate_gyro_, true);
  dlio::declare_param(this, "odom/imu/calibration/time", this->imu_calib_time_, 3.0);
  dlio::declare_param(this, "odom/imu/bufferSize", this->imu_buffer_size_, 2000);

  std::vector<double> accel_default{0., 0., 0.};
  std::vector<double> prior_accel_bias;
  std::vector<double> gyro_default{0., 0., 0.};
  std::vector<double> prior_gyro_bias;

  dlio::declare_param(this, "odom/imu/approximateGravity", this->gravity_align_, true);
  dlio::declare_param(this, "imu/calibration", this->imu_calibrate_, true);
  dlio::declare_param(this, "imu/intrinsics/accel/bias", prior_accel_bias, accel_default);
  dlio::declare_param(this, "imu/intrinsics/gyro/bias", prior_gyro_bias, gyro_default);

  // scale-misalignment matrix
  std::vector<double> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<double> imu_sm;

  dlio::declare_param(this, "imu/intrinsics/accel/sm", imu_sm, imu_sm_default);

  if (!this->imu_calibrate_)
  {
    this->state.b.accel[0] = prior_accel_bias[0];
    this->state.b.accel[1] = prior_accel_bias[1];
    this->state.b.accel[2] = prior_accel_bias[2];
    this->state.b.gyro[0] = prior_gyro_bias[0];
    this->state.b.gyro[1] = prior_gyro_bias[1];
    this->state.b.gyro[2] = prior_gyro_bias[2];
    this->imu_accel_sm_ = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(imu_sm.begin(), imu_sm.end()).data(), 3, 3);
  }
  else
  {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // Force off IMU-dependent features when use_imu=false
  if (!this->use_imu_)
  {
    this->deskew_ = false;
    this->imu_calibrate_ = false;
    this->gravity_align_ = false;
    this->calibrate_accel_ = false;
    this->calibrate_gyro_ = false;
    this->adaptive_params_ = false;
    RCLCPP_INFO(this->get_logger(), "[odom] use_imu=false: deskew/calibration/gravity/adaptive forced off");
  }

  // Registration method
  dlio::declare_param(this, "odom/registration_method", this->registration_method_, std::string("gicp"));

  // NDT params
  dlio::declare_param(this, "odom/ndt/resolution", this->ndt_resolution_, 2.0);
  dlio::declare_param(this, "odom/ndt/num_threads", this->ndt_num_threads_, 4);
  dlio::declare_param(this, "odom/ndt/search_method", this->ndt_search_method_, std::string("DIRECT7"));
  dlio::declare_param(this, "odom/ndt/step_size", this->ndt_step_size_, 0.1);

  // GICP
  dlio::declare_param(this, "odom/gicp/minNumPoints", this->gicp_min_num_points_, 100);
  dlio::declare_param(this, "odom/gicp/kCorrespondences", this->gicp_k_correspondences_, 20);
  dlio::declare_param(this, "odom/gicp/maxCorrespondenceDistance", this->gicp_max_corr_dist_,
                      std::sqrt(std::numeric_limits<double>::max()));
  dlio::declare_param(this, "odom/gicp/maxIterations", this->gicp_max_iter_, 64);
  dlio::declare_param(this, "odom/gicp/transformationEpsilon", this->gicp_transformation_ep_, 0.0005);
  dlio::declare_param(this, "odom/gicp/rotationEpsilon", this->gicp_rotation_ep_, 0.0005);
  dlio::declare_param(this, "odom/gicp/initLambdaFactor", this->gicp_init_lambda_factor_, 1e-9);

  // Geometric Observer
  dlio::declare_param(this, "odom/geo/Kp", this->geo_Kp_, 1.0);
  dlio::declare_param(this, "odom/geo/Kv", this->geo_Kv_, 1.0);
  dlio::declare_param(this, "odom/geo/Kq", this->geo_Kq_, 1.0);
  dlio::declare_param(this, "odom/geo/Kab", this->geo_Kab_, 1.0);
  dlio::declare_param(this, "odom/geo/Kgb", this->geo_Kgb_, 1.0);
  dlio::declare_param(this, "odom/geo/abias_max", this->geo_abias_max_, 1.0);
  dlio::declare_param(this, "odom/geo/gbias_max", this->geo_gbias_max_, 1.0);

  // Localization registration helper (continuous loc, submap loc, SC reloc)
  {
    std::string loc_reg_method;
    dlio::declare_param(this, "odom/localization/registration_method", loc_reg_method, std::string("gicp"));

    auto setupEngine = [&](dlio::RegistrationEngine &eng, const std::string &method, double corr_dist)
    {
      dlio::RegistrationParams rp;
      rp.gicp_k_correspondences = this->gicp_k_correspondences_;
      rp.gicp_max_corr_dist = corr_dist;
      rp.gicp_max_iter = this->gicp_max_iter_;
      rp.gicp_transformation_ep = this->gicp_transformation_ep_;
      rp.gicp_rotation_ep = this->gicp_rotation_ep_;
      rp.gicp_init_lambda_factor = this->gicp_init_lambda_factor_;
      rp.gicp_gpu_voxel_size = this->vf_res_ > 0.0 ? static_cast<float>(this->vf_res_) : 0.5f;
      rp.ndt_resolution = static_cast<float>(this->ndt_resolution_);
      rp.ndt_num_threads = this->ndt_num_threads_;
      rp.ndt_search_method = this->ndt_search_method_;
      rp.ndt_step_size = this->ndt_step_size_;
      eng.init(dlio::parseRegistrationMethod(method), rp, this->get_logger());
    };

    setupEngine(this->loc_registration_, loc_reg_method, this->gicp_max_corr_dist_);
    setupEngine(this->reloc_registration_, loc_reg_method, 5.0); // wider for relocalization

    RCLCPP_INFO(this->get_logger(), "[odom] Localization registration: %s", loc_reg_method.c_str());
  }

  // Map load/save
  dlio::declare_param(this, "map/mode", this->map_mode_, std::string("localization"));
  dlio::declare_param(this, "map/tf_source", this->tf_map_odom_source_, std::string("odom"));
  dlio::declare_param(this, "map/path", this->map_path_, std::string(""));
  if (this->map_path_.empty())
  {
    const char *home = std::getenv("HOME");
    if (home)
    {
      this->map_path_ = std::string(home) + "/.ros/dlio_map.pcd";
      RCLCPP_INFO(this->get_logger(), "map/path not set, defaulting to: %s", this->map_path_.c_str());
    }
  }
  dlio::declare_param(this, "map/use_corrected", this->use_corrected_, true);
  dlio::declare_param(this, "map/voxel_size", this->map_voxel_size_, 0.25);
  dlio::declare_param(this, "map/chunk_size", this->map_chunk_size_, 20.0);

  double init_x = 0., init_y = 0., init_z = 0., init_yaw = 0.;
  dlio::declare_param(this, "map/initial_pose/x", init_x, 0.0);
  dlio::declare_param(this, "map/initial_pose/y", init_y, 0.0);
  dlio::declare_param(this, "map/initial_pose/z", init_z, 0.0);
  dlio::declare_param(this, "map/initial_pose/yaw", init_yaw, 0.0);
  this->initial_position_ = Eigen::Vector3f(init_x, init_y, init_z);
  this->initial_yaw_ = init_yaw * M_PI / 180.0;

  // Scan Context Relocalization
  dlio::declare_param(this, "map/relocalize", this->relocalize_, false);
  double sc_max_range_d = 40.0;
  dlio::declare_param(this, "map/relocalize/sc_max_range", sc_max_range_d, 40.0);
  this->sc_max_range_ = static_cast<float>(sc_max_range_d);
  dlio::declare_param(this, "map/relocalize/sc_num_candidates", this->sc_num_candidates_, 10);
  dlio::declare_param(this, "map/relocalize/sc_distance_threshold", this->sc_distance_threshold_, 0.3);
  dlio::declare_param(this, "map/relocalize/sc_max_attempts", this->sc_max_attempts_, 10);
  double sc_ground_ht = 0.3;
  dlio::declare_param(this, "map/relocalize/sc_ground_height_threshold", sc_ground_ht, 0.3);
  this->sc_ground_height_threshold_ = static_cast<float>(sc_ground_ht);
  dlio::declare_param(this, "map/relocalize/sc_search_window", this->sc_search_window_, 7);

  // Appearance engine (loop closure descriptor)
  {
    std::string appearance_method;
    dlio::declare_param(this, "map/appearance_method", appearance_method, std::string("sc++"));
    dlio::AppearanceParams ap;
    ap.max_range = this->sc_max_range_;
    ap.ground_height_threshold = this->sc_ground_height_threshold_;
    ap.search_window = this->sc_search_window_;
    this->appearance_.init(dlio::parseAppearanceMethod(appearance_method), ap, this->get_logger());
  }

  // Continuous localization (Bayesian)
  dlio::declare_param(this, "map/continuous_localize", this->continuous_localize_, true);
  dlio::declare_param(this, "map/continuous_localize/on_mapping", this->continuous_localize_on_mapping_, false);
  dlio::declare_param(this, "map/continuous_localize/interval", this->continuous_localize_interval_, 2.0);
  dlio::declare_param(this, "map/continuous_localize/fitness_threshold", this->continuous_localize_fitness_thresh_, 0.15);
  dlio::declare_param(this, "map/continuous_localize/max_correction", this->continuous_localize_max_correction_, 2.0);
  double vp_prior = 0.9;
  dlio::declare_param(this, "map/continuous_localize/virtual_place_prior", vp_prior, 0.9);
  this->bayes_virtual_place_prior_ = static_cast<float>(vp_prior);
  double loop_thr = 0.5;
  dlio::declare_param(this, "map/continuous_localize/loop_threshold", loop_thr, 0.5);
  this->bayes_loop_threshold_ = static_cast<float>(loop_thr);
  dlio::declare_param(this, "map/continuous_localize/min_consecutive", this->bayes_min_consecutive_, 2);
  double sc_dist_thr = 0.4;
  dlio::declare_param(this, "map/continuous_localize/sc_distance_threshold", sc_dist_thr, 0.4);
  this->bayes_sc_dist_threshold_ = static_cast<float>(sc_dist_thr);
  dlio::declare_param(this, "map/continuous_localize/sc_top_k", this->bayes_sc_top_k_, 5);
  dlio::declare_param(this, "map/continuous_localize/g2o_verification", this->g2o_verification_enabled_, true);
  dlio::declare_param(this, "map/continuous_localize/g2o_chi2_threshold", this->g2o_chi2_threshold_, 50.0);
  dlio::declare_param(this, "map/continuous_localize/g2o_iterations", this->g2o_iterations_, 10);

  dlio::declare_param(this, "frames/map", this->map_frame_, std::string("map"));

  // Submap-based relocalization (GPS-denied)
  dlio::declare_param(this, "map/submap_localize", this->submap_loc_enabled_, false);
  dlio::declare_param(this, "map/submap_localize/interval", this->submap_loc_interval_, 2.0);
  dlio::declare_param(this, "map/submap_localize/group_size", this->submap_loc_group_size_, 10);
  dlio::declare_param(this, "map/submap_localize/search_radius", this->submap_loc_search_radius_, 50.0);
  dlio::declare_param(this, "map/submap_localize/fitness_threshold", this->submap_loc_fitness_thresh_, 0.15);
  dlio::declare_param(this, "map/submap_localize/prob_threshold", this->submap_loc_prob_threshold_, 0.6);
  dlio::declare_param(this, "map/submap_localize/motion_error_threshold", this->submap_loc_motion_error_thresh_, 1.0);
  dlio::declare_param(this, "map/submap_localize/motion_rot_threshold", this->submap_loc_motion_rot_thresh_, 5.0);
  dlio::declare_param(this, "map/submap_localize/min_motion_validations", this->submap_loc_min_motion_valid_, 3);
  dlio::declare_param(this, "map/submap_localize/prob_drop_threshold", this->submap_loc_prob_drop_thresh_, 0.3);
  dlio::declare_param(this, "map/submap_localize/max_correction", this->submap_loc_max_correction_, 5.0);
  {
    double sc_thresh = 0.5;
    dlio::declare_param(this, "map/submap_localize/sc_distance_threshold", sc_thresh, 0.5);
    this->submap_loc_sc_dist_thresh_ = static_cast<float>(sc_thresh);
  }
  dlio::declare_param(this, "map/submap_localize/sc_accum_scans", this->submap_loc_sc_accum_scans_, 5);

  // GPS
  dlio::declare_param(this, "gps/enabled", this->gps_enabled_, false);
  dlio::declare_param(this, "gps/topic", this->gps_topic_, std::string("fix"));
  dlio::declare_param(this, "gps/origin/latitude", this->gps_origin_param_lat_, 0.0);
  dlio::declare_param(this, "gps/origin/longitude", this->gps_origin_param_lon_, 0.0);
  dlio::declare_param(this, "gps/origin/altitude", this->gps_origin_param_alt_, 0.0);
  double gps_sr = 30.0;
  dlio::declare_param(this, "gps/search_radius", gps_sr, 30.0);
  this->gps_search_radius_ = static_cast<float>(gps_sr);
  double gps_ma = 5.0;
  dlio::declare_param(this, "gps/min_accuracy", gps_ma, 5.0);
  this->gps_min_accuracy_ = static_cast<float>(gps_ma);
  dlio::declare_param(this, "gps/publish_earth_tf", this->gps_publish_earth_tf_, true);
  dlio::declare_param(this, "gps/trust_all", this->gps_trust_all_, false);

  // Occupancy Grid
  dlio::declare_param(this, "occupancy_grid/enabled", this->occupancy_grid_enabled_, false);
  if (this->occupancy_grid_enabled_)
  {
    dlio::OccupancyGridGenerator::Params ogp;
    dlio::declare_param(this, "occupancy_grid/grid_size_x", ogp.grid_size_x, 100.0);
    dlio::declare_param(this, "occupancy_grid/grid_size_y", ogp.grid_size_y, 100.0);
    dlio::declare_param(this, "occupancy_grid/resolution", ogp.resolution, 0.2);
    dlio::declare_param(this, "occupancy_grid/ground_threshold", ogp.ground_threshold, -0.3);
    dlio::declare_param(this, "occupancy_grid/obstacle_min_height", ogp.obstacle_min_height, 0.1);
    dlio::declare_param(this, "occupancy_grid/obstacle_max_height", ogp.obstacle_max_height, 3.0);
    dlio::declare_param(this, "occupancy_grid/p_occupied", ogp.p_occupied, 0.7);
    dlio::declare_param(this, "occupancy_grid/p_free", ogp.p_free, 0.3);
    dlio::declare_param(this, "occupancy_grid/p_prior", ogp.p_prior, 0.5);
    dlio::declare_param(this, "occupancy_grid/lo_clamped_min", ogp.lo_clamped_min, -4.0);
    dlio::declare_param(this, "occupancy_grid/lo_clamped_max", ogp.lo_clamped_max, 4.0);
    dlio::declare_param(this, "occupancy_grid/decay_rate", ogp.decay_rate, 0.0);
    dlio::declare_param(this, "occupancy_grid/obstacle_margin", ogp.obstacle_margin, 0.3);
    dlio::declare_param(this, "occupancy_grid/height_offset", ogp.height_offset, 0.0);
    dlio::declare_param(this, "occupancy_grid/adaptive_ground", ogp.adaptive_ground, false);
    dlio::declare_param(this, "occupancy_grid/ground_slope_threshold", ogp.ground_slope_threshold, 10.0);
    dlio::declare_param(this, "occupancy_grid/scan_angle_target", ogp.scan_angle_target, 0.0);
    dlio::declare_param(this, "occupancy_grid/scan_angle_tolerance", ogp.scan_angle_tolerance, 180.0);
    ogp.frame_id = this->map_frame_;
    this->occupancy_grid_gen_ = std::make_unique<dlio::OccupancyGridGenerator>(ogp);
  }
}

void dlio::OdomNode::callbackGPS(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
  float h_acc = 0.0f;

  if (!this->gps_trust_all_)
  {
    // Reject invalid fix
    if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX)
      return;

    // Extract horizontal accuracy from covariance (position_covariance[0] = east variance)
    h_acc = std::sqrt(static_cast<float>(msg->position_covariance[0]));

    // Reject poor accuracy
    if (h_acc > this->gps_min_accuracy_ && msg->position_covariance_type != 0)
      return;
  }

  // Initialize origin from first fix if not set from params
  if (!this->gps_origin_set_)
  {
    this->gps_origin_lat_ = msg->latitude;
    this->gps_origin_lon_ = msg->longitude;
    this->gps_origin_alt_ = msg->altitude;
    this->gps_converter_ = std::make_unique<GeographicLib::LocalCartesian>(
        this->gps_origin_lat_, this->gps_origin_lon_, this->gps_origin_alt_);
    this->gps_origin_set_ = true;
    RCLCPP_INFO(this->get_logger(),
                "[GPS] Origin set: lat=%.8f lon=%.8f alt=%.2f",
                this->gps_origin_lat_, this->gps_origin_lon_, this->gps_origin_alt_);
  }

  // Publish earth→map static TF (once)
  if (this->gps_publish_earth_tf_ && !this->earth_tf_published_)
  {
    this->publishEarthToMapTF();
    this->earth_tf_published_ = true;
  }

  // Buffer GPS measurement — always use node clock (sim or wall) for consistency
  // with latest_scan_time_ which also uses this->now().seconds()
  double ts = this->now().seconds();

  {
    std::lock_guard<std::mutex> lock(this->gps_buffer_mtx_);
    if (this->gps_buffer_.size() >= GPS_BUFFER_MAX)
      this->gps_buffer_.pop_front();
    this->gps_buffer_.push_back({msg->latitude, msg->longitude, msg->altitude,
                                 ts, h_acc, static_cast<uint8_t>(msg->status.status)});
  }
}

bool dlio::OdomNode::getGPSAtTime(double timestamp, GPSMeasurement &out)
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
    return false; // max 1s age
  out = this->gps_buffer_[best_idx];
  return true;
}

bool dlio::OdomNode::gpsToLocal(double lat, double lon, double alt,
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

void dlio::OdomNode::start()
{

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_ << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(dlio::OdomNode)
