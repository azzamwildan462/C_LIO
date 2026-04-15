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

#include "dlio/mapping/lio_sam_map_optimization.h"
#include "dlio/odom/utils.h"

#include <filesystem>

using gtsam::symbol_shorthand::X; // Pose3 variables

// ============================================================
//  Constructor / Destructor
// ============================================================

dlio::LioSamMapOptimizationNode::LioSamMapOptimizationNode(const rclcpp::NodeOptions &options)
    : Node("dlio_lio_sam_map_opt_node", options)
{
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    this->getParams();

    // --- GTSAM iSAM2 ---
    gtsam::ISAM2Params isam_params;
    isam_params.relinearizeThreshold = this->isam_relinearize_threshold_;
    isam_params.relinearizeSkip = this->isam_relinearize_skip_;
    this->isam_ = new gtsam::ISAM2(isam_params);
    this->pose_covariance_ = Eigen::MatrixXd::Identity(6, 6) * 1e8;

    // --- KD-tree ---
    this->keyframe_poses_3d_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    this->kdtree_history_keyposes_.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>());

    // --- Subscribers ---

    // Keyframe
    this->keyframe_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    auto kf_sub_opt = rclcpp::SubscriptionOptions();
    kf_sub_opt.callback_group = this->keyframe_cb_group_;
    this->keyframe_sub_ = this->create_subscription<direct_lidar_inertial_odometry::msg::KeyframeStamped>(
        "keyframe_stamped", 100,
        std::bind(&dlio::LioSamMapOptimizationNode::callbackKeyframe, this, std::placeholders::_1),
        kf_sub_opt);

    // Deskewed scan subscription removed — SC computation falls back to kf.cloud_world.
    // Subscribing to the large raw deskewed cloud from a separate process causes heavy
    // DDS serialization in OdomNode's publish thread, degrading odometry quality.

    // GPS
    if (this->gps_enabled_)
    {
        this->gps_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        auto gps_sub_opt = rclcpp::SubscriptionOptions();
        gps_sub_opt.callback_group = this->gps_cb_group_;
        this->gps_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            this->gps_topic_, rclcpp::SensorDataQoS(),
            std::bind(&dlio::LioSamMapOptimizationNode::callbackGPS, this, std::placeholders::_1),
            gps_sub_opt);
        RCLCPP_INFO(this->get_logger(), "[lio_sam_opt] GPS enabled on topic: %s", this->gps_topic_.c_str());
    }

    // --- Publishers ---
    this->corrected_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("corrected_path", 1);
    this->corrected_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("corrected_map", 1);
    this->corrected_kf_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("corrected_kf_poses", 10);
    this->loop_closure_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("loop_closures", 1);
    this->corrected_fusion_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("corrected_fusion_path", 1);
    this->corrected_fusion_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("corrected_fusion_odom", 1);
    this->corrected_fusion_path_.header.frame_id = this->map_frame_;

    // --- Odom subscriber (for corrected fusion pose at odom rate) ---
    this->odom_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    auto odom_sub_opt = rclcpp::SubscriptionOptions();
    odom_sub_opt.callback_group = this->odom_cb_group_;
    this->odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odom", rclcpp::SensorDataQoS(),
        std::bind(&dlio::LioSamMapOptimizationNode::callbackOdom, this, std::placeholders::_1),
        odom_sub_opt);

    // --- TF broadcaster ---
    if (this->tf_map_odom_source_ == "lio_sam_opt")
    {
        this->tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        this->tf_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&dlio::LioSamMapOptimizationNode::publishMapToOdomTF, this));
        RCLCPP_INFO(this->get_logger(), "[lio_sam_opt] Publishing map->odom TF from this node (20 Hz)");
    }

    // --- Service ---
    this->save_pcd_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    this->save_pcd_srv_ = this->create_service<direct_lidar_inertial_odometry::srv::SavePCD>(
        "save_corrected_pcd",
        std::bind(&dlio::LioSamMapOptimizationNode::savePCD, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default,
        this->save_pcd_cb_group_);

    // --- Loop closure timer ---
    this->loop_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(this->loop_detection_period_ms_),
        std::bind(&dlio::LioSamMapOptimizationNode::loopClosureThread, this));

    // --- Visualization timer (corrected path/map) ---
    this->vis_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(this->publish_interval_),
        std::bind(&dlio::LioSamMapOptimizationNode::publishCorrectedData, this));

    // --- Auto-save timer ---
    if (this->map_mode_ == "mapping" && !this->map_path_.empty() && this->auto_save_interval_ > 0.)
    {
        this->auto_save_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(this->auto_save_interval_),
            std::bind(&dlio::LioSamMapOptimizationNode::autoSave, this));
    }

    RCLCPP_INFO(this->get_logger(),
                "[lio_sam_opt] Initialized (iSAM2, lc_radius=%.1fm, lc_time_diff=%.1fs, "
                "fitness_thresh=%.2f, sc=%s, gps=%s)",
                this->history_keyframe_search_radius_,
                this->history_keyframe_search_time_diff_,
                this->history_keyframe_fitness_score_,
                this->sc_enabled_ ? "on" : "off",
                this->gps_enabled_ ? "on" : "off");
}

dlio::LioSamMapOptimizationNode::~LioSamMapOptimizationNode()
{
    this->saveOnShutdown();
    delete this->isam_;
}

// ============================================================
//  Parameters
// ============================================================

void dlio::LioSamMapOptimizationNode::getParams()
{
    dlio::declare_param(this, "debug/lio_sam_map_opt", this->debug_, false);
    dlio::declare_param(this, "debug/print_lio_sam_opt", this->debug_print_, false);
    dlio::declare_param(this, "frames/odom", this->odom_frame_, std::string("odom"));
    dlio::declare_param(this, "frames/map", this->map_frame_, std::string("map"));

    // Map correction (master switch for loop closure + GPS correction)
    dlio::declare_param(this, "map_correction", this->map_correction_enabled_, true);

    // Loop closure detection (LIO-SAM style)
    dlio::declare_param(this, "loop_detection_period_ms", this->loop_detection_period_ms_, 1000);
    dlio::declare_param(this, "loop_closure/search_radius", this->history_keyframe_search_radius_, 15.0);
    dlio::declare_param(this, "loop_closure/search_time_diff", this->history_keyframe_search_time_diff_, 30.0);
    dlio::declare_param(this, "loop_closure/search_num", this->history_keyframe_search_num_, 25);
    dlio::declare_param(this, "loop_closure/fitness_score_threshold", this->history_keyframe_fitness_score_, 0.3);
    dlio::declare_param(this, "loop_closure/min_keyframe_gap", this->min_keyframe_gap_, 30);

    // SC++ (secondary loop closure)
    dlio::declare_param(this, "sc/enabled", this->sc_enabled_, true);
    double sc_mr = 20.0;
    dlio::declare_param(this, "sc/max_range", sc_mr, 20.0);
    this->sc_max_range_ = static_cast<float>(sc_mr);
    double sc_dt = 0.5;
    dlio::declare_param(this, "sc/distance_threshold", sc_dt, 0.5);
    this->sc_distance_threshold_ = static_cast<float>(sc_dt);
    double sc_ght = 0.0;
    dlio::declare_param(this, "sc/ground_height_threshold", sc_ght, 0.0);
    this->sc_ground_height_threshold_ = static_cast<float>(sc_ght);
    dlio::declare_param(this, "sc/search_window", this->sc_search_window_, 7);

    // Init appearance engine for loop closure descriptor
    {
        dlio::AppearanceParams ap;
        ap.max_range = this->sc_max_range_;
        ap.ground_height_threshold = this->sc_ground_height_threshold_;
        ap.search_window = this->sc_search_window_;
        this->appearance_.init(dlio::AppearanceMethod::SC_PLUS_PLUS, ap, this->get_logger());
    }

    // Registration method for loop closure verification
    dlio::declare_param(this, "registration_method", this->lc_registration_method_, std::string("icp"));
    dlio::declare_param(this, "icp/max_iterations", this->lc_max_iterations_, 100);
    dlio::declare_param(this, "icp/max_correspondence_distance", this->lc_max_corr_dist_, 30.0);
    dlio::declare_param(this, "icp/transformation_epsilon", this->lc_transformation_ep_, 1e-6);
    dlio::declare_param(this, "gicp/k_correspondences", this->lc_gicp_k_correspondences_, 20);
    dlio::declare_param(this, "gicp/rotation_epsilon", this->lc_rotation_ep_, 0.01);
    dlio::declare_param(this, "ndt/resolution", this->ndt_resolution_, 1.0);
    dlio::declare_param(this, "ndt/num_threads", this->ndt_num_threads_, 4);

    dlio::declare_param(this, "voxel_leaf_size", this->voxel_leaf_size_, 0.2);

    // GPS
    dlio::declare_param(this, "gps/enabled", this->gps_enabled_, false);
    dlio::declare_param(this, "gps/topic", this->gps_topic_, std::string("/gps/fix"));
    dlio::declare_param(this, "gps/cov_threshold", this->gps_cov_threshold_, 2.0);
    dlio::declare_param(this, "gps/pose_cov_threshold", this->pose_cov_threshold_, 25.0);
    dlio::declare_param(this, "gps/use_elevation", this->use_gps_elevation_, false);
    double gps_ma = 5.0;
    dlio::declare_param(this, "gps/min_accuracy", gps_ma, 5.0);
    this->gps_min_accuracy_ = static_cast<float>(gps_ma);
    double gps_mnf = 0.1;
    dlio::declare_param(this, "gps/min_noise_floor", gps_mnf, 0.1);
    this->gps_min_noise_floor_ = static_cast<float>(gps_mnf);
    std::vector<double> gps_t_default = {0.0, 0.0, 0.0};
    std::vector<double> gps_t_vec;
    dlio::declare_param(this, "extrinsics/baselink2gps/t", gps_t_vec, gps_t_default);
    this->gps_extrinsic_t_ = Eigen::Vector3f(gps_t_vec[0], gps_t_vec[1], gps_t_vec[2]);
    std::vector<double> gps_rpy_default = {0.0, 0.0, 0.0};
    std::vector<double> gps_rpy_vec;
    dlio::declare_param(this, "extrinsics/baselink2gps/rpy", gps_rpy_vec, gps_rpy_default);
    {
        float r = static_cast<float>(gps_rpy_vec[0]) * M_PI / 180.f;
        float p = static_cast<float>(gps_rpy_vec[1]) * M_PI / 180.f;
        float yy = static_cast<float>(gps_rpy_vec[2]) * M_PI / 180.f;
        this->gps_extrinsic_R_ = Eigen::AngleAxisf(yy, Eigen::Vector3f::UnitZ())
                                * Eigen::AngleAxisf(p, Eigen::Vector3f::UnitY())
                                * Eigen::AngleAxisf(r, Eigen::Vector3f::UnitX());
    }
    // GPS extrinsic info moved to debugPrint()
    dlio::declare_param(this, "gps/gating_mode", this->gps_gating_mode_, std::string("covariance"));
    double gps_lc_r = 20.0;
    dlio::declare_param(this, "gps/lc_search_radius", gps_lc_r, 20.0);
    this->gps_lc_search_radius_ = static_cast<float>(gps_lc_r);
    double fused_dr = 0.5;
    dlio::declare_param(this, "gps/fused_drift_rate", fused_dr, 0.5);
    this->fused_drift_rate_ = static_cast<float>(fused_dr);
    double gps_bm = 10.0;
    dlio::declare_param(this, "gps/buffer_margin", gps_bm, 10.0);
    this->gps_buffer_margin_ = static_cast<float>(gps_bm);

    // Fusion KF
    double fon = 0.01, fcn = 1.0;
    dlio::declare_param(this, "fusion/odom_noise", fon, 0.01);
    dlio::declare_param(this, "fusion/correction_noise", fcn, 1.0);
    this->fusion_odom_noise_ = static_cast<float>(fon);
    this->fusion_gps_noise_ = static_cast<float>(fcn);

    // iSAM2
    dlio::declare_param(this, "isam/relinearize_threshold", this->isam_relinearize_threshold_, 0.1);
    dlio::declare_param(this, "isam/relinearize_skip", this->isam_relinearize_skip_, 1);
    dlio::declare_param(this, "isam/batch_optimization", this->batch_optimization_, false);
    dlio::declare_param(this, "isam/batch_optimization_interval", this->batch_optimization_interval_, 100);

    // Noise model
    dlio::declare_param(this, "odom_noise/rotation", this->odom_noise_rot_, 0.1);
    dlio::declare_param(this, "odom_noise/translation", this->odom_noise_trans_, 0.5);
    dlio::declare_param(this, "loop_noise/multiplier", this->loop_noise_multiplier_, 0.001);

    // Map save
    dlio::declare_param(this, "map/mode", this->map_mode_, std::string("mapping"));
    dlio::declare_param(this, "map/tf_source", this->tf_map_odom_source_, std::string("odom"));
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
    dlio::declare_param(this, "map/global_map_vis_radius", this->global_map_vis_radius_, 1000.0);
}

// ============================================================
//  TF: map → odom
// ============================================================

void dlio::LioSamMapOptimizationNode::publishMapToOdomTF()
{
    if (!this->tf_broadcaster_)
        return;

    // Read cached T_map_odom (computed in correctPoses() under keyframes_mtx_)
    Eigen::Isometry3d T_map_odom;
    {
        std::lock_guard<std::mutex> lock(this->tf_map_odom_mtx_);
        T_map_odom = this->T_map_odom_cached_;
    }

    Eigen::Quaterniond q(T_map_odom.rotation());
    q.normalize();

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = this->tf_stamp_cached_;
    tf_msg.header.frame_id = this->map_frame_;
    tf_msg.child_frame_id = this->odom_frame_;

    tf_msg.transform.translation.x = T_map_odom.translation().x();
    tf_msg.transform.translation.y = T_map_odom.translation().y();
    tf_msg.transform.translation.z = T_map_odom.translation().z();
    tf_msg.transform.rotation.w = q.w();
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();

    this->tf_broadcaster_->sendTransform(tf_msg);
}

// ============================================================
//  Utility
// ============================================================

gtsam::Pose3 dlio::LioSamMapOptimizationNode::isometryToGtsamPose(const Eigen::Isometry3d &iso)
{
    return gtsam::Pose3(gtsam::Rot3(iso.rotation()), gtsam::Point3(iso.translation()));
}

Eigen::Isometry3d dlio::LioSamMapOptimizationNode::gtsamPoseToIsometry(const gtsam::Pose3 &pose)
{
    Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
    iso.linear() = pose.rotation().matrix();
    iso.translation() = pose.translation();
    return iso;
}

// ============================================================
//  Callbacks
// ============================================================

void dlio::LioSamMapOptimizationNode::callbackKeyframe(
    const direct_lidar_inertial_odometry::msg::KeyframeStamped::SharedPtr msg)
{
    // Update KF timing prediction for adaptive GPS buffer
    {
        double kf_stamp = rclcpp::Time(msg->header.stamp).seconds();
        if (this->last_kf_stamp_ > 0.0)
        {
            double dt = kf_stamp - this->last_kf_stamp_;
            if (dt > 0.0 && dt < 10.0) // sanity check
                this->avg_kf_dt_ = 0.9 * this->avg_kf_dt_ + 0.1 * dt;
        }
        this->last_kf_stamp_ = kf_stamp;
    }

    Keyframe kf;
    kf.id = msg->id;
    kf.timestamp = msg->header.stamp;

    // Extract pose
    Eigen::Quaterniond q(msg->pose.orientation.w, msg->pose.orientation.x,
                         msg->pose.orientation.y, msg->pose.orientation.z);
    Eigen::Vector3d t(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    kf.pose = Eigen::Isometry3d::Identity();
    kf.pose.linear() = q.toRotationMatrix();
    kf.pose.translation() = t;

    // Cloud from DLIO is already in world frame
    kf.cloud_world = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::fromROSMsg(msg->cloud, *kf.cloud_world);

    // Compute body-frame cloud for re-transformation after optimization
    kf.cloud_local = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud(*kf.cloud_world, *kf.cloud_local,
                             kf.pose.inverse().matrix().cast<float>());

    // Compute SC++ descriptor if enabled
    if (this->sc_enabled_)
    {
        pcl::PointCloud<PointType>::Ptr sc_cloud;

        // Try dense deskewed scan first
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
                    sc_cloud = std::make_shared<pcl::PointCloud<PointType>>(
                        *this->deskewed_buffer_[best_idx].cloud);
                }
            }
        }

        if (!sc_cloud)
            sc_cloud = std::make_shared<pcl::PointCloud<PointType>>(*kf.cloud_world);

        // Center at sensor position
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

        // Ground removal
        if (this->sc_ground_height_threshold_ > 0.0f)
        {
            auto filtered = std::make_shared<pcl::PointCloud<PointType>>();
            filtered->points.reserve(sc_cloud->points.size());
            for (const auto &pt : sc_cloud->points)
                if (pt.z >= this->sc_ground_height_threshold_)
                    filtered->points.push_back(pt);
            sc_cloud = filtered;
        }

        kf.appearance_desc = this->appearance_.computeDescriptor(sc_cloud);
    }

    // Attach GPS
    if (this->gps_enabled_)
    {
        double kf_time = rclcpp::Time(msg->header.stamp).seconds();
        GPSMeasurement gps;
        if (this->getGPSAtTime(kf_time, gps))
        {
            kf.gps_valid = this->gpsToLocal(gps.latitude, gps.longitude, gps.altitude,
                                            kf.gps_x, kf.gps_y, kf.gps_z);
            kf.gps_horizontal_accuracy = gps.horizontal_accuracy;
        }
    }

    // Auto-calibrate ENU→odom yaw from movement direction
    if (!this->enu_yaw_calibrated_ && kf.gps_valid)
    {
        if (this->enu_calib_kf_idx_ < 0)
        {
            // Store first GPS point + odom position
            this->first_gps_enu_ = Eigen::Vector3f(kf.gps_x, kf.gps_y, kf.gps_z);
            this->first_odom_pos_ = kf.pose.translation().cast<float>();
            this->enu_calib_kf_idx_ = static_cast<int>(this->keyframes_.size());
        }
        else
        {
            Eigen::Vector3f gps_now(kf.gps_x, kf.gps_y, kf.gps_z);
            Eigen::Vector3f odom_now = kf.pose.translation().cast<float>();
            Eigen::Vector3f gps_delta = gps_now - this->first_gps_enu_;
            Eigen::Vector3f odom_delta = odom_now - this->first_odom_pos_;
            float gps_dist = gps_delta.head<2>().norm();
            float odom_dist = odom_delta.head<2>().norm();

            if (gps_dist > 5.0f && odom_dist > 5.0f)
            {
                // Compute yaw offset: angle from ENU direction to odom direction
                float gps_yaw = std::atan2(gps_delta[1], gps_delta[0]);
                float odom_yaw = std::atan2(odom_delta[1], odom_delta[0]);
                float yaw_offset = odom_yaw - gps_yaw;

                this->R_enu_to_odom_ = Eigen::AngleAxisf(yaw_offset, Eigen::Vector3f::UnitZ()).toRotationMatrix();
                this->enu_yaw_calibrated_ = true;

                // Re-convert stored GPS coordinates with new rotation
                // (first_gps_enu_ was converted before calibration, but gpsToLocal now uses R_enu_to_odom_)
                RCLCPP_INFO(this->get_logger(),
                            "[lio_sam_opt] ENU->odom yaw calibrated: %.1f deg (gps_dist=%.1fm, odom_dist=%.1fm)",
                            yaw_offset * 180.0f / M_PI, gps_dist, odom_dist);
            }
        }
    }

    // Fused pose computation (GPS direct XY+Z, odom orientation, dead-reckon when GPS lost)
    {
        int idx = static_cast<int>(this->keyframes_.size()); // pre-lock peek for idx
        if (kf.gps_valid)
        {
            kf.fused_pose = kf.pose; // copy orientation from odom
            kf.fused_pose.translation().x() = static_cast<double>(kf.gps_x);
            kf.fused_pose.translation().y() = static_cast<double>(kf.gps_y);
            if (this->use_gps_elevation_)
                kf.fused_pose.translation().z() = static_cast<double>(kf.gps_z);

            kf.fused_covariance = kf.gps_horizontal_accuracy * kf.gps_horizontal_accuracy;
            kf.fused_valid = true;

            this->last_gps_kf_idx_ = idx;
            this->last_gps_fused_pose_ = kf.fused_pose;
            this->last_gps_odom_pose_ = kf.pose;
            this->last_gps_accuracy_ = kf.gps_horizontal_accuracy;
        }
        else if (this->last_gps_kf_idx_ >= 0)
        {
            // GPS lost: differential from last GPS anchor (XY + Z all smooth)
            Eigen::Isometry3d odom_delta = this->last_gps_odom_pose_.inverse() * kf.pose;
            kf.fused_pose = this->last_gps_fused_pose_ * odom_delta;

            double dt = rclcpp::Time(msg->header.stamp).seconds() -
                        rclcpp::Time(this->keyframes_[this->last_gps_kf_idx_].timestamp).seconds();
            kf.fused_covariance = this->last_gps_accuracy_ * this->last_gps_accuracy_
                                + this->fused_drift_rate_ * static_cast<float>(std::abs(dt));
            kf.fused_valid = true;
        }
    }

    // Brief lock: add keyframe + factors, then release immediately
    {
        std::lock_guard<std::mutex> lock(this->keyframes_mtx_);

        int idx = static_cast<int>(this->keyframes_.size());
        gtsam::Pose3 current_pose = this->isometryToGtsamPose(kf.pose);

        if (idx == 0)
        {
            auto prior_noise = gtsam::noiseModel::Diagonal::Variances(
                (gtsam::Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());
            this->gtsam_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), current_pose, prior_noise));
            this->initial_estimate_.insert(X(0), current_pose);
        }
        else
        {
            gtsam::Pose3 prev_pose = this->isometryToGtsamPose(this->keyframes_.back().pose);
            gtsam::Pose3 relative = prev_pose.between(current_pose);
            auto odom_noise = gtsam::noiseModel::Diagonal::Variances(
                (gtsam::Vector(6) << this->odom_noise_rot_, this->odom_noise_rot_, this->odom_noise_rot_,
                 this->odom_noise_trans_, this->odom_noise_trans_, this->odom_noise_trans_)
                    .finished());
            this->gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                X(idx - 1), X(idx), relative, odom_noise));
            this->initial_estimate_.insert(X(idx), current_pose);
        }

        if (this->map_correction_enabled_)
        {
            this->addGPSFactor(idx, kf);
            this->addLoopFactors();
        }
        this->updateISAM();
        this->keyframes_.push_back(kf);
        this->batchOptimize();

        pcl::PointXYZ pose_pt;
        pose_pt.x = static_cast<float>(kf.pose.translation().x());
        pose_pt.y = static_cast<float>(kf.pose.translation().y());
        pose_pt.z = static_cast<float>(kf.pose.translation().z());
        this->keyframe_poses_3d_->push_back(pose_pt);

        this->correctPoses();
        this->publishMapToOdomTF();
    }
    // Lock released — heavy work above is unavoidable but iSAM2 incremental
    // updates are fast (O(affected nodes), not O(all nodes))

    if (this->debug_)
    {
        RCLCPP_INFO(this->get_logger(), "[lio_sam_opt] Keyframe %u added (idx=%zu, pos=[%.2f,%.2f,%.2f], %zu pts)",
                    kf.id, this->keyframes_.size() - 1,
                    kf.pose.translation().x(), kf.pose.translation().y(), kf.pose.translation().z(),
                    kf.cloud_world->points.size());
    }

    this->debugPrint();
}

void dlio::LioSamMapOptimizationNode::callbackDeskewed(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    DeskewedScan scan;
    scan.timestamp = msg->header.stamp;
    scan.cloud = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::fromROSMsg(*msg, *scan.cloud);

    std::lock_guard<std::mutex> lock(this->deskewed_buffer_mtx_);
    if (this->deskewed_buffer_.size() >= DESKEWED_BUFFER_MAX)
        this->deskewed_buffer_.pop_front();
    this->deskewed_buffer_.push_back(std::move(scan));
}

void dlio::LioSamMapOptimizationNode::callbackGPS(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
    if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX)
        return;

    float h_acc = std::sqrt(static_cast<float>(msg->position_covariance[0]));
    if (h_acc > this->gps_min_accuracy_ && msg->position_covariance_type != 0)
        return;

    if (!this->gps_origin_set_)
    {
        this->gps_converter_ = std::make_unique<GeographicLib::LocalCartesian>(
            msg->latitude, msg->longitude, msg->altitude);
        this->gps_origin_set_ = true;
        RCLCPP_INFO(this->get_logger(), "[lio_sam_opt] GPS origin set: lat=%.8f lon=%.8f alt=%.2f",
                    msg->latitude, msg->longitude, msg->altitude);
    }

    double ts = rclcpp::Time(msg->header.stamp).seconds();
    {
        std::lock_guard<std::mutex> lock(this->gps_buffer_mtx_);
        if (this->gps_buffer_.size() >= GPS_BUFFER_MAX)
            this->gps_buffer_.pop_front();
        this->gps_buffer_.push_back({msg->latitude, msg->longitude, msg->altitude, ts, h_acc});
    }
}

void dlio::LioSamMapOptimizationNode::callbackOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // Extract current odom pose
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    Eigen::Vector3d t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
    odom_pose.linear() = q.toRotationMatrix();
    odom_pose.translation() = t;

    // ===== KF PREDICT: apply differential odom =====
    if (this->prev_odom_valid_)
    {
        // Differential: delta = prev⁻¹ × current
        Eigen::Isometry3d delta = this->prev_odom_pose_.inverse() * odom_pose;
        Eigen::Vector3d dt_vec = delta.translation();
        double dist = dt_vec.norm();

        // Predict: apply delta to fusion state
        this->fusion_pos_ += this->fusion_q_.toRotationMatrix() * dt_vec;
        Eigen::Quaterniond dq(delta.rotation());
        this->fusion_q_ = (this->fusion_q_ * dq).normalized();

        // Process noise: covariance grows with distance traveled
        this->fusion_P_ += Eigen::Vector3d::Constant(this->fusion_odom_noise_ * dist);
    }
    else
    {
        // Initialize fusion state from first odom
        this->fusion_pos_ = t;
        this->fusion_q_ = q;
    }
    this->prev_odom_pose_ = odom_pose;
    this->prev_odom_valid_ = true;

    // ===== KF UPDATE: correct from PGO when available =====
    // try_lock: never block odom callback — skip if keyframe processing holds the lock
    if (this->correction_mtx_.try_lock())
    {
        if (this->corrected_pose_updated_)
        {
            // Kalman gain per axis: K = P / (P + R)
            Eigen::Vector3d R = Eigen::Vector3d::Constant(this->fusion_gps_noise_);
            Eigen::Vector3d K = this->fusion_P_.cwiseQuotient(this->fusion_P_ + R);

            // Position update: x = x + K * (measurement - x)
            Eigen::Vector3d innovation = this->corrected_pos_ - this->fusion_pos_;
            this->fusion_pos_ += K.cwiseProduct(innovation);

            // Covariance update: P = (1-K) * P
            this->fusion_P_ = (Eigen::Vector3d::Ones() - K).cwiseProduct(this->fusion_P_);

            // Orientation: slerp toward corrected
            double q_alpha = K.mean();
            this->fusion_q_ = this->fusion_q_.slerp(q_alpha, this->corrected_q_).normalized();

            this->corrected_pose_updated_ = false;
        }
        this->correction_mtx_.unlock();
    }

    // ===== PUBLISH =====
    nav_msgs::msg::Odometry fused_odom;
    fused_odom.header = msg->header;
    fused_odom.header.frame_id = this->map_frame_;
    fused_odom.child_frame_id = "base_link";
    fused_odom.pose.pose.position.x = this->fusion_pos_.x();
    fused_odom.pose.pose.position.y = this->fusion_pos_.y();
    fused_odom.pose.pose.position.z = this->fusion_pos_.z();
    fused_odom.pose.pose.orientation.w = this->fusion_q_.w();
    fused_odom.pose.pose.orientation.x = this->fusion_q_.x();
    fused_odom.pose.pose.orientation.y = this->fusion_q_.y();
    fused_odom.pose.pose.orientation.z = this->fusion_q_.z();
    this->corrected_fusion_odom_pub_->publish(fused_odom);

    // Append to path
    geometry_msgs::msg::PoseStamped ps;
    ps.header = fused_odom.header;
    ps.pose = fused_odom.pose.pose;
    this->corrected_fusion_path_.header.stamp = msg->header.stamp;
    this->corrected_fusion_path_.poses.push_back(ps);
    this->corrected_fusion_path_pub_->publish(this->corrected_fusion_path_);
}

// ============================================================
//  GPS helpers
// ============================================================

bool dlio::LioSamMapOptimizationNode::getGPSAtTime(double timestamp, GPSMeasurement &out)
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

    double margin = static_cast<double>(this->gps_buffer_margin_);
    if (best_idx < 0 || best_dt > margin)
        return false;

    out = this->gps_buffer_[best_idx];
    return true;
}

bool dlio::LioSamMapOptimizationNode::gpsToLocal(double lat, double lon, double alt,
                                                 float &x, float &y, float &z)
{
    if (!this->gps_origin_set_ || !this->gps_converter_)
        return false;
    double dx, dy, dz;
    this->gps_converter_->Forward(lat, lon, alt, dx, dy, dz);
    // Rotate ENU → odom frame (yaw alignment, auto-calibrated or from config)
    Eigen::Vector3f gps_enu(static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz));
    Eigen::Vector3f rotated = this->R_enu_to_odom_ * gps_enu;
    x = rotated[0];
    y = rotated[1];
    z = rotated[2];
    return true;
}

// ============================================================
//  GTSAM Factor Addition
// ============================================================

void dlio::LioSamMapOptimizationNode::addGPSFactor(int idx, const Keyframe &kf)
{
    if (!this->gps_enabled_ || !kf.gps_valid)
        return;

    // Wait for system to settle (like LIO-SAM: need at least some travel)
    if (idx < 2)
        return;

    // Only add GPS when pose covariance is high (uncertain)
    // In "always" mode, skip this gate to ensure GPS anchors at tunnel entry/exit
    if (this->gps_gating_mode_ == "covariance")
    {
        if (this->pose_covariance_(3, 3) < this->pose_cov_threshold_ &&
            this->pose_covariance_(4, 4) < this->pose_cov_threshold_)
            return;
    }

    float gps_x = kf.gps_x;
    float gps_y = kf.gps_y;
    float gps_z = kf.gps_z;

    if (!this->use_gps_elevation_)
    {
        gps_z = static_cast<float>(kf.pose.translation().z());
    }

    // Skip near-zero GPS (not initialized)
    if (std::abs(gps_x) < 1e-6f && std::abs(gps_y) < 1e-6f)
        return;

    // Throttle: only add if moved > 5m from last GPS factor
    static pcl::PointXYZ last_gps_pt{0, 0, 0};
    pcl::PointXYZ cur_gps_pt;
    cur_gps_pt.x = gps_x;
    cur_gps_pt.y = gps_y;
    cur_gps_pt.z = gps_z;
    float dx = cur_gps_pt.x - last_gps_pt.x;
    float dy = cur_gps_pt.y - last_gps_pt.y;
    float dz = cur_gps_pt.z - last_gps_pt.z;
    if (std::sqrt(dx * dx + dy * dy + dz * dz) < 5.0f)
        return;
    last_gps_pt = cur_gps_pt;

    // Noise: use actual GPS horizontal accuracy (sigma) with 1.0m floor
    // horizontal_accuracy is sigma (std dev) from sqrt(position_covariance[0])
    // GTSAM Diagonal::Variances expects variance = sigma^2
    float h_sigma = std::max(kf.gps_horizontal_accuracy, this->gps_min_noise_floor_);
    float noise_x = h_sigma * h_sigma;
    float noise_y = h_sigma * h_sigma;
    // When use_gps_elevation=false, set Z noise very large so GPS doesn't constrain Z
    // (let LiDAR/IMU handle Z — they're much better at vertical constraint)
    float noise_z = this->use_gps_elevation_ ? (h_sigma * h_sigma) : 1000.0f;

    gtsam::Vector3 gps_noise_vec;
    gps_noise_vec << noise_x, noise_y, noise_z;
    auto gps_noise = gtsam::noiseModel::Diagonal::Variances(gps_noise_vec);

    gtsam::GPSFactor gps_factor(X(idx), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
    this->gtsam_graph_.add(gps_factor);
    this->gps_factor_count_++;

    this->a_loop_is_closed_ = true;

    if (this->debug_)
    {
        RCLCPP_INFO(this->get_logger(),
                    "[lio_sam_opt] GPS factor added at idx %d: [%.2f, %.2f, %.2f] (h_acc=%.2fm, var=%.2f)",
                    idx, gps_x, gps_y, gps_z, kf.gps_horizontal_accuracy, noise_x);
    }
}

void dlio::LioSamMapOptimizationNode::addLoopFactors()
{
    std::lock_guard<std::mutex> lock(this->loop_queue_mtx_);
    if (this->loop_queue_.empty())
        return;

    for (const auto &lc : this->loop_queue_)
    {
        this->gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            X(lc.from_idx), X(lc.to_idx), lc.relative_pose, lc.noise));
    }

    this->loop_queue_.clear();
    this->a_loop_is_closed_ = true;
}

void dlio::LioSamMapOptimizationNode::updateISAM()
{
    try
    {
        this->isam_->update(this->gtsam_graph_, this->initial_estimate_);
        this->isam_->update();

        // Extra iterations for loop closure convergence (like LIO-SAM)
        if (this->a_loop_is_closed_)
        {
            this->isam_->update();
            this->isam_->update();
            this->isam_->update();
            this->isam_->update();
            this->isam_->update();
        }

        this->gtsam_graph_.resize(0);
        this->initial_estimate_.clear();

        // Extract latest estimate and covariance
        this->isam_current_estimate_ = this->isam_->calculateEstimate();

        int latest_idx = static_cast<int>(this->isam_current_estimate_.size()) - 1;
        if (latest_idx >= 0)
        {
            this->pose_covariance_ = this->isam_->marginalCovariance(X(latest_idx));
        }
    }
    catch (const gtsam::IndeterminantLinearSystemException &e)
    {
        RCLCPP_WARN(this->get_logger(),
                    "[lio_sam_opt] Indeterminant linear system — rebuilding iSAM2 from last good state");
        this->gtsam_graph_.resize(0);
        this->initial_estimate_.clear();
        this->a_loop_is_closed_ = false;

        // Rebuild iSAM2 from keyframe odometry factors only (drop problematic factors)
        try
        {
            gtsam::NonlinearFactorGraph clean_graph;
            gtsam::Values clean_values;
            int num_kf = static_cast<int>(this->keyframes_.size());

            for (int i = 0; i < num_kf; ++i)
            {
                gtsam::Pose3 pose_i = this->isometryToGtsamPose(this->keyframes_[i].pose);
                clean_values.insert(X(i), pose_i);

                if (i == 0)
                {
                    auto prior_noise = gtsam::noiseModel::Diagonal::Variances(
                        (gtsam::Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());
                    clean_graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), pose_i, prior_noise));
                }
                else
                {
                    gtsam::Pose3 pose_prev = this->isometryToGtsamPose(this->keyframes_[i - 1].pose);
                    gtsam::Pose3 relative = pose_prev.between(pose_i);
                    auto odom_noise = gtsam::noiseModel::Diagonal::Variances(
                        (gtsam::Vector(6) << this->odom_noise_rot_, this->odom_noise_rot_, this->odom_noise_rot_,
                         this->odom_noise_trans_, this->odom_noise_trans_, this->odom_noise_trans_).finished());
                    clean_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(X(i - 1), X(i), relative, odom_noise));
                }
            }

            gtsam::ISAM2Params isam_params;
            isam_params.relinearizeThreshold = this->isam_relinearize_threshold_;
            isam_params.relinearizeSkip = this->isam_relinearize_skip_;
            delete this->isam_;
            this->isam_ = new gtsam::ISAM2(isam_params);
            this->isam_->update(clean_graph, clean_values);
            this->isam_current_estimate_ = this->isam_->calculateEstimate();
            RCLCPP_INFO(this->get_logger(), "[lio_sam_opt] iSAM2 rebuilt with %d clean keyframes", num_kf);
        }
        catch (const std::exception &rebuild_e)
        {
            RCLCPP_ERROR(this->get_logger(), "[lio_sam_opt] iSAM2 rebuild failed: %s", rebuild_e.what());
        }
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(this->get_logger(),
                     "[lio_sam_opt] GTSAM exception: %s", e.what());
        this->gtsam_graph_.resize(0);
        this->initial_estimate_.clear();
        this->a_loop_is_closed_ = false;
    }
}

void dlio::LioSamMapOptimizationNode::batchOptimize()
{
    if (!this->batch_optimization_ || !this->isam_)
        return;

    int num_kf = static_cast<int>(this->keyframes_.size());
    if (num_kf < 10)
        return;

    // Only run every N keyframes
    if (num_kf % this->batch_optimization_interval_ != 0)
        return;

    try
    {
        // Extract full accumulated graph from iSAM2
        gtsam::NonlinearFactorGraph full_graph = this->isam_->getFactorsUnsafe();
        gtsam::Values current_values = this->isam_->calculateEstimate();

        // Run Levenberg-Marquardt batch optimization on the full graph
        gtsam::LevenbergMarquardtParams lm_params;
        lm_params.maxIterations = 100;
        lm_params.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;

        gtsam::LevenbergMarquardtOptimizer lm(full_graph, current_values, lm_params);
        double initial_error = lm.error();
        gtsam::Values result = lm.optimize();
        double final_error = lm.error();

        // Reinitialize iSAM2 with the batch-optimized result
        // so future incremental updates start from the corrected state
        gtsam::ISAM2Params isam_params;
        isam_params.relinearizeThreshold = this->isam_relinearize_threshold_;
        isam_params.relinearizeSkip = this->isam_relinearize_skip_;
        delete this->isam_;
        this->isam_ = new gtsam::ISAM2(isam_params);
        this->isam_->update(full_graph, result);
        this->isam_->update(); // extra iteration for convergence

        this->isam_current_estimate_ = this->isam_->calculateEstimate();
        this->a_loop_is_closed_ = true;

        RCLCPP_INFO(this->get_logger(),
                    "[lio_sam_opt] Batch LM optimization done (%d poses, %zu factors, %d iters, error %.4f -> %.4f)",
                    num_kf, full_graph.size(), lm.iterations(),
                    initial_error, final_error);
    }
    catch (const std::exception &e)
    {
        RCLCPP_WARN(this->get_logger(),
                    "[lio_sam_opt] Batch optimization failed: %s", e.what());
    }
}

void dlio::LioSamMapOptimizationNode::correctPoses()
{
    if (this->keyframes_.empty())
        return;

    int num_poses = static_cast<int>(this->isam_current_estimate_.size());
    this->optimized_poses_.resize(num_poses);

    for (int i = 0; i < num_poses; ++i)
    {
        this->optimized_poses_[i] = this->isam_current_estimate_.at<gtsam::Pose3>(X(i));
    }

    if (this->a_loop_is_closed_)
    {
        // Update KD-tree with corrected poses
        this->keyframe_poses_3d_->clear();
        for (int i = 0; i < num_poses; ++i)
        {
            pcl::PointXYZ pt;
            pt.x = static_cast<float>(this->optimized_poses_[i].translation().x());
            pt.y = static_cast<float>(this->optimized_poses_[i].translation().y());
            pt.z = static_cast<float>(this->optimized_poses_[i].translation().z());
            this->keyframe_poses_3d_->push_back(pt);
        }

        this->a_loop_is_closed_ = false;
        this->poses_corrected_count_++;
    }

    // Cache T_map_odom for the 20 Hz TF timer (avoids race on keyframes_/optimized_poses_)
    int latest_idx = num_poses - 1;
    if (latest_idx >= 0 && latest_idx < static_cast<int>(this->keyframes_.size()))
    {
        Eigen::Isometry3d optimized_pose = this->gtsamPoseToIsometry(this->optimized_poses_[latest_idx]);
        Eigen::Isometry3d raw_odom_pose = this->keyframes_[latest_idx].pose;
        Eigen::Isometry3d T_map_odom = optimized_pose * raw_odom_pose.inverse();
        {
            std::lock_guard<std::mutex> lock(this->correction_mtx_);
            this->corrected_pos_ = optimized_pose.translation();
            this->corrected_q_ = Eigen::Quaterniond(optimized_pose.rotation());
            this->corrected_pose_updated_ = true;
        }
        {
            std::lock_guard<std::mutex> lock(this->tf_map_odom_mtx_);
            this->T_map_odom_cached_ = T_map_odom;
            this->tf_stamp_cached_ = this->keyframes_[latest_idx].timestamp;
        }
    }
}

// ============================================================
//  Loop Closure Detection (LIO-SAM style)
// ============================================================

void dlio::LioSamMapOptimizationNode::loopClosureThread()
{
    if (this->map_mode_ != "mapping")
        return;

    this->performLoopClosure();
}

bool dlio::LioSamMapOptimizationNode::detectLoopClosureDistance(int &loop_cur, int &loop_pre)
{
    // Must have keyframes_mtx_ locked by caller or work on snapshot
    int num_kf = static_cast<int>(this->keyframe_poses_3d_->size());
    if (num_kf < this->min_keyframe_gap_ + 1)
        return false;

    loop_cur = num_kf - 1;
    loop_pre = -1;

    // Check if already found loop for this keyframe
    auto it = this->loop_index_container_.find(loop_cur);
    if (it != this->loop_index_container_.end())
        return false;

    // KD-tree radius search
    this->kdtree_history_keyposes_->setInputCloud(this->keyframe_poses_3d_);
    std::vector<int> search_ind;
    std::vector<float> search_dist;
    this->kdtree_history_keyposes_->radiusSearch(
        this->keyframe_poses_3d_->back(),
        this->history_keyframe_search_radius_,
        search_ind, search_dist, 0);

    // Find oldest keyframe within radius that satisfies time AND index gap
    double cur_time = rclcpp::Time(this->keyframes_[loop_cur].timestamp).seconds();
    for (int i = 0; i < static_cast<int>(search_ind.size()); ++i)
    {
        int id = search_ind[i];
        // Skip candidates that are too close in index (adjacent poses)
        if (std::abs(loop_cur - id) < this->min_keyframe_gap_)
            continue;
        double id_time = rclcpp::Time(this->keyframes_[id].timestamp).seconds();
        if (std::abs(cur_time - id_time) > this->history_keyframe_search_time_diff_)
        {
            loop_pre = id;
            break;
        }
    }

    if (loop_pre == -1 || loop_cur == loop_pre)
        return false;

    return true;
}

bool dlio::LioSamMapOptimizationNode::detectLoopClosureSC(int &loop_cur, int &loop_pre, int &sc_shift)
{
    if (!this->sc_enabled_)
        return false;

    int num_kf = static_cast<int>(this->keyframes_.size());
    if (num_kf < this->min_keyframe_gap_ + 1)
        return false;

    loop_cur = num_kf - 1;
    loop_pre = -1;
    sc_shift = 0;

    // Check duplicate
    auto it = this->loop_index_container_.find(loop_cur);
    if (it != this->loop_index_container_.end())
        return false;

    const auto &cur_desc = this->keyframes_[loop_cur].appearance_desc;
    float best_dist = std::numeric_limits<float>::max();

    for (int i = 0; i < loop_cur - this->min_keyframe_gap_; ++i)
    {
        auto [dist, shift] = this->appearance_.compareDescriptors(
            cur_desc, this->keyframes_[i].appearance_desc);

        if (dist < this->sc_distance_threshold_ && dist < best_dist)
        {
            best_dist = dist;
            loop_pre = i;
            sc_shift = shift;
        }
    }

    return (loop_pre >= 0);
}

bool dlio::LioSamMapOptimizationNode::detectLoopClosureGPS(int &loop_cur, int &loop_pre)
{
    if (!this->gps_enabled_)
        return false;

    int num_kf = static_cast<int>(this->keyframes_.size());
    if (num_kf < this->min_keyframe_gap_ + 1)
        return false;

    loop_cur = num_kf - 1;
    loop_pre = -1;
    const auto &cur_kf = this->keyframes_[loop_cur];

    // Use fused pose (works even when GPS is lost — dead-reckoned)
    if (!cur_kf.fused_valid)
        return false;

    // Check duplicate
    if (this->loop_index_container_.count(loop_cur))
        return false;

    // Search radius adapts to fused covariance (larger when GPS lost longer)
    float search_r = this->gps_lc_search_radius_ + std::sqrt(cur_kf.fused_covariance);
    float best_dist = std::numeric_limits<float>::max();

    for (int i = 0; i < loop_cur - this->min_keyframe_gap_; ++i)
    {
        const auto &cand_kf = this->keyframes_[i];
        if (!cand_kf.fused_valid)
            continue;

        // Temporal constraint
        double dt = std::abs(
            rclcpp::Time(cur_kf.timestamp).seconds() -
            rclcpp::Time(cand_kf.timestamp).seconds());
        if (dt < this->history_keyframe_search_time_diff_)
            continue;

        // Fused pose distance (2D)
        float dx = static_cast<float>(cur_kf.fused_pose.translation().x() - cand_kf.fused_pose.translation().x());
        float dy = static_cast<float>(cur_kf.fused_pose.translation().y() - cand_kf.fused_pose.translation().y());
        float fused_dist = std::sqrt(dx * dx + dy * dy);

        // Effective radius = base + uncertainty of both keyframes
        float effective_r = search_r + std::sqrt(cand_kf.fused_covariance);

        if (fused_dist < effective_r && fused_dist < best_dist)
        {
            best_dist = fused_dist;
            loop_pre = i;
        }
    }

    if (loop_pre >= 0 && this->debug_)
    {
        RCLCPP_INFO(this->get_logger(),
                    "[lio_sam_opt] GPS-fused loop candidate: %d <-> %d (fused_dist=%.2fm, search_r=%.1fm)",
                    loop_cur, loop_pre, best_dist, search_r + std::sqrt(this->keyframes_[loop_pre].fused_covariance));
    }

    return (loop_pre >= 0);
}

void dlio::LioSamMapOptimizationNode::performLoopClosure()
{
    // ---- Phase 1: brief lock to detect candidates + build clouds ----
    int loop_cur = -1, loop_pre = -1;
    int sc_shift = 0;
    bool found_distance = false, found_sc = false, found_gps = false;
    pcl::PointCloud<PointType>::Ptr cur_keyframe_cloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prev_keyframe_cloud(new pcl::PointCloud<PointType>());
    Eigen::Isometry3d cur_pose_snap, pre_pose_snap;

    {
        std::lock_guard<std::mutex> lock(this->keyframes_mtx_);

        if (this->keyframes_.empty())
            return;

        // Detection order: GPS-fused first (primary), then distance, then SC++
        if (this->gps_enabled_)
            found_gps = this->detectLoopClosureGPS(loop_cur, loop_pre);
        if (!found_gps)
            found_distance = this->detectLoopClosureDistance(loop_cur, loop_pre);
        if (!found_gps && !found_distance)
            found_sc = this->detectLoopClosureSC(loop_cur, loop_pre, sc_shift);

        if (!found_gps && !found_distance && !found_sc)
            return;

        // Snapshot poses + build clouds
        int num_opt = static_cast<int>(this->optimized_poses_.size());

        if (found_gps)
        {
            // GPS-fused LC: use fused poses for cloud building (not drifted odom)
            cur_pose_snap = this->keyframes_[loop_cur].fused_valid
                                ? this->keyframes_[loop_cur].fused_pose
                                : this->keyframes_[loop_cur].pose;
            pre_pose_snap = this->keyframes_[loop_pre].fused_valid
                                ? this->keyframes_[loop_pre].fused_pose
                                : this->keyframes_[loop_pre].pose;
        }
        else
        {
            // Distance/SC++ LC: use optimized or odom poses (existing behavior)
            cur_pose_snap = (loop_cur < num_opt)
                                ? this->gtsamPoseToIsometry(this->optimized_poses_[loop_cur])
                                : this->keyframes_[loop_cur].pose;
            pre_pose_snap = (loop_pre < num_opt)
                                ? this->gtsamPoseToIsometry(this->optimized_poses_[loop_pre])
                                : this->keyframes_[loop_pre].pose;
        }

        // Build current keyframe cloud
        pcl::transformPointCloud(*this->keyframes_[loop_cur].cloud_local,
                                 *cur_keyframe_cloud, cur_pose_snap.matrix().cast<float>());

        // Build history submap
        int cloud_size = static_cast<int>(this->keyframes_.size());
        for (int i = -this->history_keyframe_search_num_; i <= this->history_keyframe_search_num_; ++i)
        {
            int key_near = loop_pre + i;
            if (key_near < 0 || key_near >= cloud_size)
                continue;
            Eigen::Isometry3d pose;
            if (found_gps && this->keyframes_[key_near].fused_valid)
                pose = this->keyframes_[key_near].fused_pose;
            else if (key_near < num_opt)
                pose = this->gtsamPoseToIsometry(this->optimized_poses_[key_near]);
            else
                pose = this->keyframes_[key_near].pose;
            pcl::PointCloud<PointType> tmp;
            pcl::transformPointCloud(*this->keyframes_[key_near].cloud_local, tmp, pose.matrix().cast<float>());
            *prev_keyframe_cloud += tmp;
        }
    }
    // ---- Lock released — all heavy work below runs without holding mutex ----

    if (this->debug_)
    {
        RCLCPP_INFO(this->get_logger(), "[lio_sam_opt] Loop candidate: kf%d <-> kf%d (%s)",
                    loop_cur, loop_pre,
                    found_gps ? "GPS-fused" : (found_distance ? "distance" : "SC++"));
    }

    if (cur_keyframe_cloud->size() < 300 || prev_keyframe_cloud->size() < 1000)
        return;

    // Downsample
    pcl::VoxelGrid<PointType> ds;
    ds.setLeafSize(this->voxel_leaf_size_, this->voxel_leaf_size_, this->voxel_leaf_size_);
    ds.setInputCloud(prev_keyframe_cloud);
    ds.filter(*prev_keyframe_cloud);

    // ---- Phase 2: registration (heavy, no lock held) ----
    pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>());
    bool converged = false;
    double fitness_score = std::numeric_limits<double>::max();
    Eigen::Matrix4f final_T = Eigen::Matrix4f::Identity();

    // Compute initial guess from SC++ heading shift (rotate around source centroid)
    Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
    if (found_sc && sc_shift != 0)
    {
        float yaw = this->appearance_.shiftToYaw(sc_shift);
        Eigen::AngleAxisf rot(yaw, Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f R = rot.toRotationMatrix();
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cur_keyframe_cloud, centroid);
        Eigen::Vector3f c = centroid.head<3>();
        initial_guess.block<3, 3>(0, 0) = R;
        initial_guess.block<3, 1>(0, 3) = c - R * c;

        if (this->debug_)
        {
            RCLCPP_INFO(this->get_logger(),
                        "[lio_sam_opt] SC++ initial guess: yaw=%.1f deg (shift=%d/%d)",
                        yaw * 180.0f / M_PI, sc_shift, this->appearance_.numSectors());
        }
    }

    {
        dlio::RegistrationParams lc_params;
        lc_params.gicp_k_correspondences = this->lc_gicp_k_correspondences_;
        lc_params.gicp_max_corr_dist = this->lc_max_corr_dist_;
        lc_params.gicp_max_iter = this->lc_max_iterations_;
        lc_params.gicp_transformation_ep = this->lc_transformation_ep_;
        lc_params.gicp_rotation_ep = this->lc_rotation_ep_;
        lc_params.ndt_resolution = static_cast<float>(this->ndt_resolution_);
        lc_params.ndt_num_threads = this->ndt_num_threads_;

        dlio::RegistrationEngine lc_engine;
        lc_engine.init(dlio::parseRegistrationMethod(this->lc_registration_method_), lc_params, this->get_logger());
        lc_engine.setInputSource(cur_keyframe_cloud);
        lc_engine.setInputTarget(prev_keyframe_cloud);
        pcl::PointCloud<PointType> aligned;
        lc_engine.align(aligned);
        converged = lc_engine.hasConverged();
        fitness_score = lc_engine.getFitnessScore(1.0);
        final_T = lc_engine.getFinalTransformation();
    }

    if (!converged || fitness_score > this->history_keyframe_fitness_score_)
    {
        this->lc_rejected_count_++;
        if (this->debug_)
        {
            RCLCPP_INFO(this->get_logger(), "[lio_sam_opt] Loop REJECTED: %d<->%d fitness=%.4f (thresh=%.2f)",
                        loop_cur, loop_pre, fitness_score, this->history_keyframe_fitness_score_);
        }
        return;
    }

    RCLCPP_INFO(this->get_logger(), "[lio_sam_opt] Loop CONFIRMED: %d<->%d fitness=%.4f",
                loop_cur, loop_pre, fitness_score);

    // ---- Phase 3: compute constraint from snapshots (no lock needed) ----
    Eigen::Affine3f correction(final_T);
    Eigen::Affine3f t_wrong(cur_pose_snap.matrix().cast<float>());
    Eigen::Affine3f t_correct = correction * t_wrong;

    // Log ICP correction magnitude
    {
        Eigen::Matrix3f R_corr = correction.rotation();
        float angle_rad = std::acos(std::min(1.0f, std::max(-1.0f,
                                                            (R_corr.trace() - 1.0f) / 2.0f)));
        Eigen::Vector3f t_corr = correction.translation();
        RCLCPP_INFO(this->get_logger(),
                    "[lio_sam_opt] Loop ICP correction: rot=%.2f deg, trans=[%.3f,%.3f,%.3f] (norm=%.3f)",
                    angle_rad * 180.0f / M_PI, t_corr.x(), t_corr.y(), t_corr.z(), t_corr.norm());
    }

    gtsam::Pose3 pose_from = gtsam::Pose3(t_correct.matrix().cast<double>());
    gtsam::Pose3 pose_to = this->isometryToGtsamPose(pre_pose_snap);
    gtsam::Pose3 pose_between = pose_from.between(pose_to);

    // ICP fitness is mean squared point distance — not a good variance directly.
    float noise_score = static_cast<float>(fitness_score) * static_cast<float>(this->loop_noise_multiplier_);
    float noise_rot = noise_score;
    float noise_trans = noise_score;
    gtsam::Vector6 noise_vec;
    noise_vec << noise_rot, noise_rot, noise_rot, noise_trans, noise_trans, noise_trans;
    auto constraint_noise = gtsam::noiseModel::Diagonal::Variances(noise_vec);

    // Brief lock: queue the constraint
    {
        std::lock_guard<std::mutex> lk(this->loop_queue_mtx_);
        this->loop_queue_.push_back({loop_cur, loop_pre, pose_between, constraint_noise});
    }
    this->lc_found_count_++;

    {
        std::lock_guard<std::mutex> lk2(this->keyframes_mtx_);
        this->loop_index_container_[loop_cur] = loop_pre;
    }
}

// ============================================================
//  Publishing
// ============================================================

void dlio::LioSamMapOptimizationNode::publishCorrectedData()
{
    // ---- Brief lock: snapshot poses and timestamps ----
    std::vector<PublishCorrectedData_PoseSnapshot> pose_snap;
    std::vector<pcl::PointCloud<PointType>::Ptr> cloud_snap;

    {
        std::lock_guard<std::mutex> lock(this->keyframes_mtx_);

        int num_kf = static_cast<int>(this->keyframes_.size());
        if (num_kf == 0)
            return;

        int num_opt = static_cast<int>(this->optimized_poses_.size());

        pose_snap.resize(num_kf);
        cloud_snap.resize(num_kf);
        for (int i = 0; i < num_kf; ++i)
        {
            pose_snap[i].pose = (i < num_opt)
                                    ? this->gtsamPoseToIsometry(this->optimized_poses_[i])
                                    : this->keyframes_[i].pose;
            pose_snap[i].timestamp = this->keyframes_[i].timestamp;
            cloud_snap[i] = this->keyframes_[i].cloud_local;
        }
    }
    // ---- Lock released — all heavy work below ----

    int num_kf = static_cast<int>(pose_snap.size());

    // Corrected path + keyframe poses (lightweight)
    nav_msgs::msg::Path path;
    path.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
    path.header.frame_id = this->map_frame_;

    geometry_msgs::msg::PoseArray kf_poses;
    kf_poses.header = path.header;

    for (int i = 0; i < num_kf; ++i)
    {
        const auto &ps_data = pose_snap[i];
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = ps_data.timestamp;
        ps.header.frame_id = this->map_frame_;
        Eigen::Quaterniond q(ps_data.pose.rotation());
        ps.pose.position.x = ps_data.pose.translation().x();
        ps.pose.position.y = ps_data.pose.translation().y();
        ps.pose.position.z = ps_data.pose.translation().z();
        ps.pose.orientation.w = q.w();
        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();
        path.poses.push_back(ps);
        kf_poses.poses.push_back(ps.pose);
    }

    this->corrected_path_pub_->publish(path);
    this->corrected_kf_pose_pub_->publish(kf_poses);

    // Global map (heavy — transform all clouds, no lock held)
    this->publishGlobalMap(pose_snap, cloud_snap);

    // Loop closure markers
    this->publishLoopClosureMarkers();
}

void dlio::LioSamMapOptimizationNode::publishGlobalMap(
    const std::vector<PublishCorrectedData_PoseSnapshot> &pose_snap,
    const std::vector<pcl::PointCloud<PointType>::Ptr> &cloud_snap)
{
    int num_kf = static_cast<int>(pose_snap.size());
    if (num_kf == 0)
        return;

    pcl::PointCloud<PointType>::Ptr global_map(new pcl::PointCloud<PointType>());

    for (int i = 0; i < num_kf; ++i)
    {
        pcl::PointCloud<PointType> tmp;
        pcl::transformPointCloud(*cloud_snap[i], tmp, pose_snap[i].pose.matrix().cast<float>());
        *global_map += tmp;
    }

    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(this->voxel_leaf_size_, this->voxel_leaf_size_, this->voxel_leaf_size_);
    voxel.setInputCloud(global_map);
    voxel.filter(*global_map);

    sensor_msgs::msg::PointCloud2 map_ros;
    pcl::toROSMsg(*global_map, map_ros);
    map_ros.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME); // use latest TF
    map_ros.header.frame_id = this->map_frame_;
    this->corrected_map_pub_->publish(map_ros);
}

void dlio::LioSamMapOptimizationNode::publishLoopClosureMarkers()
{
    // Snapshot loop edges and their poses under lock
    struct LoopEdgeSnap
    {
        Eigen::Vector3d pos_from, pos_to;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
    std::vector<LoopEdgeSnap> edge_snap;

    {
        std::lock_guard<std::mutex> lock(this->keyframes_mtx_);
        int num_opt = static_cast<int>(this->optimized_poses_.size());
        int num_kf = static_cast<int>(this->keyframes_.size());
        edge_snap.reserve(this->loop_index_container_.size());
        for (const auto &[from, to] : this->loop_index_container_)
        {
            if (from >= num_kf || to >= num_kf)
                continue;
            Eigen::Isometry3d pose_from = (from < num_opt)
                                              ? this->gtsamPoseToIsometry(this->optimized_poses_[from])
                                              : this->keyframes_[from].pose;
            Eigen::Isometry3d pose_to = (to < num_opt)
                                            ? this->gtsamPoseToIsometry(this->optimized_poses_[to])
                                            : this->keyframes_[to].pose;
            edge_snap.push_back({pose_from.translation(), pose_to.translation()});
        }
    }

    visualization_msgs::msg::MarkerArray markers;
    int idx = 0;
    for (const auto &edge : edge_snap)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = this->map_frame_;
        marker.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
        marker.ns = "loop_closures";
        marker.id = idx++;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.4;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        geometry_msgs::msg::Point p1, p2;
        p1.x = edge.pos_from.x();
        p1.y = edge.pos_from.y();
        p1.z = edge.pos_from.z();
        p2.x = edge.pos_to.x();
        p2.y = edge.pos_to.y();
        p2.z = edge.pos_to.z();
        marker.points.push_back(p1);
        marker.points.push_back(p2);

        markers.markers.push_back(marker);
    }

    this->loop_closure_pub_->publish(markers);
}

// ============================================================
//  Map Save
// ============================================================

void dlio::LioSamMapOptimizationNode::saveGraphMaps(const std::string &save_dir, float leaf_size)
{
    // Snapshot poses and clouds under lock, then release before disk I/O
    struct KfSnap
    {
        Eigen::Isometry3d pose;
        pcl::PointCloud<PointType>::Ptr cloud_local;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
    std::vector<KfSnap> kf_snap;

    {
        std::lock_guard<std::mutex> lock(this->keyframes_mtx_);

        int num_kf = static_cast<int>(this->keyframes_.size());
        if (num_kf == 0)
            return;

        int num_opt = static_cast<int>(this->optimized_poses_.size());

        kf_snap.resize(num_kf);
        for (int i = 0; i < num_kf; ++i)
        {
            kf_snap[i].pose = (i < num_opt)
                                  ? this->gtsamPoseToIsometry(this->optimized_poses_[i])
                                  : this->keyframes_[i].pose;
            kf_snap[i].cloud_local = this->keyframes_[i].cloud_local;
        }
    }
    // Lock released — disk I/O below does NOT block callbackKeyframe()

    int num_kf = static_cast<int>(kf_snap.size());

    std::filesystem::create_directories(save_dir);

    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);

    constexpr int CHUNK_SIZE = 50;
    pcl::PointCloud<PointType>::Ptr result(new pcl::PointCloud<PointType>());

    for (int start = 0; start < num_kf; start += CHUNK_SIZE)
    {
        int end = std::min(start + CHUNK_SIZE, num_kf);

        pcl::PointCloud<PointType>::Ptr chunk(new pcl::PointCloud<PointType>());
        for (int i = start; i < end; ++i)
        {
            pcl::PointCloud<PointType> tmp;
            pcl::transformPointCloud(*kf_snap[i].cloud_local, tmp, kf_snap[i].pose.matrix().cast<float>());
            *chunk += tmp;
        }
        chunk->width = chunk->points.size();
        chunk->height = 1;
        vg.setInputCloud(chunk);
        vg.filter(*chunk);
        *result += *chunk;
    }

    result->width = result->points.size();
    result->height = 1;
    vg.setInputCloud(result);
    vg.filter(*result);

    if (result->points.empty())
    {
        RCLCPP_WARN(this->get_logger(), "[lio_sam_opt] Corrected cloud is empty, skipping save");
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
        RCLCPP_INFO(this->get_logger(), "[lio_sam_opt] Saved corrected map: %zu pts -> %s",
                    result->points.size(), corr_file.c_str());
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "[lio_sam_opt] FAILED to save: %s", corr_file.c_str());
    }
}

void dlio::LioSamMapOptimizationNode::autoSave()
{
    std::filesystem::path map_fp(this->map_path_);
    std::string save_dir = map_fp.parent_path().string();
    if (save_dir.empty())
        save_dir = ".";
    this->saveGraphMaps(save_dir, static_cast<float>(this->map_voxel_size_));
}

void dlio::LioSamMapOptimizationNode::debugPrint()
{
    if (!this->debug_print_)
        return;
    if (this->debug_print_counter_++ < 2)
        return;
    this->debug_print_counter_ = 0;

    std::lock_guard<std::mutex> lock(this->keyframes_mtx_);
    if (this->keyframes_.empty())
        return;

    int num_kf = static_cast<int>(this->keyframes_.size());
    int num_opt = static_cast<int>(this->optimized_poses_.size());
    const auto &kf = this->keyframes_.back();
    auto pos = kf.pose.translation();

    printf("\033[2J\033[1;1H");
    printf("+-------------------------------------------------------------------+\n");
    printf("|             DLIO LIO-SAM Map Optimization Debug                   |\n");
    printf("+-------------------------------------------------------------------+\n");
    printf("| GRAPH                                                             |\n");
    printf("|   Keyframes:   %-6d    Optimized: %-6d                        |\n", num_kf, num_opt);
    printf("|   GPS factors: %-6d    LC found:  %-6d  rejected: %-6d      |\n",
           this->gps_factor_count_, this->lc_found_count_, this->lc_rejected_count_);
    printf("|   LC edges:    %-6d    Poses corrected: %-6d                  |\n",
           static_cast<int>(this->loop_index_container_.size()), this->poses_corrected_count_);
    printf("+-------------------------------------------------------------------+\n");
    printf("| LATEST KEYFRAME (#%u, idx=%d)                                     |\n", kf.id, num_kf - 1);
    printf("|   Odom Pos:  [%10.3f, %10.3f, %10.3f]                   |\n",
           pos.x(), pos.y(), pos.z());
    if (kf.fused_valid)
    {
        auto fp = kf.fused_pose.translation();
        printf("|   Fused Pos: [%10.3f, %10.3f, %10.3f]  cov=%.1f m^2    |\n",
               fp.x(), fp.y(), fp.z(), kf.fused_covariance);
    }
    if (kf.gps_valid)
    {
        printf("|   GPS:       [%10.3f, %10.3f, %10.3f]  acc=%.2fm       |\n",
               static_cast<double>(kf.gps_x), static_cast<double>(kf.gps_y),
               static_cast<double>(kf.gps_z), kf.gps_horizontal_accuracy);
    }
    else
    {
        printf("|   GPS:       INVALID                                              |\n");
    }
    printf("+-------------------------------------------------------------------+\n");
    printf("| GPS STATE                                                         |\n");
    printf("|   Last GPS KF: %-6d  GPS origin: %-3s                           |\n",
           this->last_gps_kf_idx_, this->gps_origin_set_ ? "SET" : "NO");
    {
        std::lock_guard<std::mutex> gps_lock(this->gps_buffer_mtx_);
        int buf_sz = static_cast<int>(this->gps_buffer_.size());
        printf("|   GPS buf: %-6d  margin=%.1fs                                  |\n",
               buf_sz, this->gps_buffer_margin_);
        if (!this->gps_buffer_.empty())
        {
            const auto &oldest = this->gps_buffer_.front();
            const auto &newest = this->gps_buffer_.back();
            double kf_s = rclcpp::Time(kf.timestamp).seconds();
            printf("|   Buf range: [%.1f .. %.1f]  span=%.1fs                  |\n",
                   oldest.timestamp, newest.timestamp, newest.timestamp - oldest.timestamp);
            printf("|   KF stamp:  %.3f  nearest_dt=%.3fs                      |\n",
                   kf_s, std::abs(newest.timestamp - kf_s));
        }
    }
    printf("+-------------------------------------------------------------------+\n");
    printf("| POSE COVARIANCE (diagonal)                                        |\n");
    printf("|   Rot:   [%10.4f, %10.4f, %10.4f]                      |\n",
           this->pose_covariance_(0, 0), this->pose_covariance_(1, 1), this->pose_covariance_(2, 2));
    printf("|   Trans: [%10.4f, %10.4f, %10.4f]                      |\n",
           this->pose_covariance_(3, 3), this->pose_covariance_(4, 4), this->pose_covariance_(5, 5));
    if (kf.fused_valid)
    {
        auto drift = pos - kf.fused_pose.translation();
        printf("+-------------------------------------------------------------------+\n");
        printf("| ODOM-FUSED DRIFT                                                  |\n");
        printf("|   Delta: [%10.3f, %10.3f, %10.3f]  norm=%.3fm          |\n",
               drift.x(), drift.y(), drift.z(), drift.norm());
    }
    if (this->batch_optimization_)
    {
        int next_batch = ((num_kf / this->batch_optimization_interval_) + 1) * this->batch_optimization_interval_;
        printf("+-------------------------------------------------------------------+\n");
        printf("| BATCH: every %d KF  (next at KF #%d)                              |\n",
               this->batch_optimization_interval_, next_batch);
    }
    printf("+-------------------------------------------------------------------+\n");
    fflush(stdout);
}

void dlio::LioSamMapOptimizationNode::saveOnShutdown()
{
    if (this->map_mode_ != "mapping")
        return;
    if (this->map_path_.empty())
        return;
    if (this->shutdown_saved_.exchange(true))
        return;

    std::filesystem::path map_fp(this->map_path_);
    std::string save_dir = map_fp.parent_path().string();
    if (save_dir.empty())
        save_dir = ".";
    this->saveGraphMaps(save_dir, static_cast<float>(this->map_voxel_size_));
}

void dlio::LioSamMapOptimizationNode::savePCD(
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res)
{
    this->saveGraphMaps(req->save_path, req->leaf_size);
    res->success = true;
}

// ============================================================
//  Component Registration
// ============================================================

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(dlio::LioSamMapOptimizationNode)
