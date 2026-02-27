#include "dlio/odom.h"
#include "dlio/utils.h"
#include "dlio/kfdb_io.h"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <unistd.h>

#include <pcl/common/transforms.h>

// Access global atexit pointer defined in odom.cc
extern std::atomic<dlio::OdomNode *> g_odom_node;

void dlio::OdomNode::publishPose()
{

  // nav_msgs::msg::Odometry
  this->odom_ros.header.stamp = this->imu_stamp;
  this->odom_ros.header.frame_id = this->odom_frame;
  this->odom_ros.child_frame_id = this->baselink_frame;

  this->odom_ros.pose.pose.position.x = this->state.p[0];
  this->odom_ros.pose.pose.position.y = this->state.p[1];
  this->odom_ros.pose.pose.position.z = this->state.p[2];

  this->odom_ros.pose.pose.orientation.w = this->state.q.w();
  this->odom_ros.pose.pose.orientation.x = this->state.q.x();
  this->odom_ros.pose.pose.orientation.y = this->state.q.y();
  this->odom_ros.pose.pose.orientation.z = this->state.q.z();

  this->odom_ros.twist.twist.linear.x = this->state.v.lin.w[0];
  this->odom_ros.twist.twist.linear.y = this->state.v.lin.w[1];
  this->odom_ros.twist.twist.linear.z = this->state.v.lin.w[2];

  this->odom_ros.twist.twist.angular.x = this->state.v.ang.b[0];
  this->odom_ros.twist.twist.angular.y = this->state.v.ang.b[1];
  this->odom_ros.twist.twist.angular.z = this->state.v.ang.b[2];

  this->odom_pub->publish(this->odom_ros);

  // geometry_msgs::msg::PoseStamped
  this->pose_ros.header.stamp = this->imu_stamp;
  this->pose_ros.header.frame_id = this->odom_frame;

  this->pose_ros.pose.position.x = this->state.p[0];
  this->pose_ros.pose.position.y = this->state.p[1];
  this->pose_ros.pose.position.z = this->state.p[2];

  this->pose_ros.pose.orientation.w = this->state.q.w();
  this->pose_ros.pose.orientation.x = this->state.q.x();
  this->pose_ros.pose.orientation.y = this->state.q.y();
  this->pose_ros.pose.orientation.z = this->state.q.z();

  this->pose_pub->publish(this->pose_ros);

  if (this->debug_print_)
  {
    if (this->debug_print_counter_++ >= 2)
    {
      this->debug_print_counter_ = 0;
      this->debug();
    }
  }
}

void dlio::OdomNode::publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud)
{
  this->publishCloud(published_cloud, T_cloud);

  // nav_msgs::msg::Path — use lidarPose directly (no smoothing)
  this->path_ros.header.stamp = this->imu_stamp;
  this->path_ros.header.frame_id = this->odom_frame;

  geometry_msgs::msg::PoseStamped p;
  p.header.stamp = this->imu_stamp;
  p.header.frame_id = this->odom_frame;
  p.pose.position.x = this->lidarPose.p[0];
  p.pose.position.y = this->lidarPose.p[1];
  p.pose.position.z = this->lidarPose.p[2];
  p.pose.orientation.w = this->lidarPose.q.w();
  p.pose.orientation.x = this->lidarPose.q.x();
  p.pose.orientation.y = this->lidarPose.q.y();
  p.pose.orientation.z = this->lidarPose.q.z();

  {
    std::lock_guard<std::mutex> lock(this->publish_mtx_);
    this->path_ros.poses.push_back(p);
    this->path_pub->publish(this->path_ros);
  }

  // transform: odom to baselink
  geometry_msgs::msg::TransformStamped transformStamped;

  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->odom_frame;
  transformStamped.child_frame_id = this->baselink_frame;

  transformStamped.transform.translation.x = this->lidarPose.p[0];
  transformStamped.transform.translation.y = this->lidarPose.p[1];
  transformStamped.transform.translation.z = this->lidarPose.p[2];

  transformStamped.transform.rotation.w = this->lidarPose.q.w();
  transformStamped.transform.rotation.x = this->lidarPose.q.x();
  transformStamped.transform.rotation.y = this->lidarPose.q.y();
  transformStamped.transform.rotation.z = this->lidarPose.q.z();

  br->sendTransform(transformStamped);

  // transform: baselink to imu
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->imu_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2imu.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2imu.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2imu.t[2];

  Eigen::Quaternionf q(this->extrinsics.baselink2imu.R);
  transformStamped.transform.rotation.w = q.w();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();

  br->sendTransform(transformStamped);

  // transform: baselink to lidar
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->lidar_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2lidar.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2lidar.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2lidar.t[2];

  Eigen::Quaternionf qq(this->extrinsics.baselink2lidar.R);
  transformStamped.transform.rotation.w = qq.w();
  transformStamped.transform.rotation.x = qq.x();
  transformStamped.transform.rotation.y = qq.y();
  transformStamped.transform.rotation.z = qq.z();

  br->sendTransform(transformStamped);

  // transform: map to odom (continuous localization correction)
  if (this->continuous_localize_)
  {
    Eigen::Matrix4f T_m2o;
    {
      std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
      T_m2o = this->T_map_odom_;
    }

    geometry_msgs::msg::TransformStamped map_odom_tf;
    map_odom_tf.header.stamp = this->imu_stamp;
    map_odom_tf.header.frame_id = this->map_frame_;
    map_odom_tf.child_frame_id = this->odom_frame;

    Eigen::Quaternionf q_m2o(T_m2o.block<3, 3>(0, 0));
    q_m2o.normalize();

    map_odom_tf.transform.translation.x = T_m2o(0, 3);
    map_odom_tf.transform.translation.y = T_m2o(1, 3);
    map_odom_tf.transform.translation.z = T_m2o(2, 3);
    map_odom_tf.transform.rotation.w = q_m2o.w();
    map_odom_tf.transform.rotation.x = q_m2o.x();
    map_odom_tf.transform.rotation.y = q_m2o.y();
    map_odom_tf.transform.rotation.z = q_m2o.z();

    br->sendTransform(map_odom_tf);
  }
}

void dlio::OdomNode::publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud)
{

  if (this->wait_until_move_)
  {
    if (this->length_traversed < 0.1)
    {
      return;
    }
  }

  pcl::PointCloud<PointType>::Ptr deskewed_scan_t_ = std::make_shared<pcl::PointCloud<PointType>>();

  pcl::transformPointCloud(*published_cloud, *deskewed_scan_t_, T_cloud);

  // published deskewed cloud (use unique_ptr for IPC zero-copy)
  auto deskewed_ros = std::make_unique<sensor_msgs::msg::PointCloud2>();
  pcl::toROSMsg(*deskewed_scan_t_, *deskewed_ros);
  deskewed_ros->header.stamp = this->scan_header_stamp;
  deskewed_ros->header.frame_id = this->odom_frame;
  this->deskewed_pub->publish(std::move(deskewed_ros));
}

void dlio::OdomNode::publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp,
                                     pcl::PointCloud<PointType>::ConstPtr local_cloud)
{

  // Push back
  geometry_msgs::msg::Pose p;
  p.position.x = kf.first.first[0];
  p.position.y = kf.first.first[1];
  p.position.z = kf.first.first[2];
  p.orientation.w = kf.first.second.w();
  p.orientation.x = kf.first.second.x();
  p.orientation.y = kf.first.second.y();
  p.orientation.z = kf.first.second.z();
  {
    std::lock_guard<std::mutex> lock(this->kf_publish_mtx_);
    this->kf_pose_ros.poses.push_back(p);

    // Publish
    this->kf_pose_ros.header.stamp = timestamp;
    this->kf_pose_ros.header.frame_id = this->odom_frame;
    this->kf_pose_pub->publish(this->kf_pose_ros);
  }

  // publish keyframe scan for map (use unique_ptr for IPC zero-copy)
  if (!this->vf_use_ || (kf.second->points.size() == kf.second->width * kf.second->height))
  {
    auto keyframe_cloud_ros = std::make_unique<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*kf.second, *keyframe_cloud_ros);
    keyframe_cloud_ros->header.stamp = timestamp;
    keyframe_cloud_ros->header.frame_id = this->odom_frame;
    this->kf_cloud_pub->publish(std::move(keyframe_cloud_ros));

    if (this->debug_)
    {
      RCLCPP_INFO(this->get_logger(), "[odom] Published kf_cloud: %zu pts, pos=[%.2f,%.2f,%.2f]",
                  kf.second->points.size(), kf.first.first[0], kf.first.first[1], kf.first.first[2]);
    }
  }

  // Publish combined keyframe message for graph SLAM (local-frame cloud)
  auto kf_msg = std::make_unique<direct_lidar_inertial_odometry::msg::KeyframeStamped>();
  kf_msg->header.stamp = timestamp;
  kf_msg->header.frame_id = this->odom_frame;
  kf_msg->id = static_cast<uint32_t>(this->kf_pose_ros.poses.size() - 1);
  kf_msg->pose = p;
  pcl::toROSMsg(*local_cloud, kf_msg->cloud);
  kf_msg->cloud.header.stamp = timestamp;
  kf_msg->cloud.header.frame_id = this->baselink_frame;
  uint32_t kf_id = kf_msg->id;
  this->kf_stamped_pub->publish(std::move(kf_msg));

  if (this->debug_)
  {
    RCLCPP_INFO(this->get_logger(), "[odom] Published kf_stamped #%u: local_cloud=%zu pts",
                kf_id, local_cloud->points.size());
  }
}

void dlio::OdomNode::srvSetMode(
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SetMode::Request> req,
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SetMode::Response> res)
{
  if (req->mode != "mapping" && req->mode != "localization")
  {
    res->success = false;
    res->message = "Invalid mode '" + req->mode + "'. Must be 'mapping' or 'localization'.";
    res->current_mode = this->map_mode_;
    return;
  }

  std::string old_mode = this->map_mode_;

  // If switching from mapping to localization: save map PCD + KFDB
  if (old_mode == "mapping" && req->mode == "localization")
  {
    if (!this->map_path_.empty())
    {
      // Save PCD via MapNode service
      if (this->callSavePCD())
      {
        RCLCPP_INFO(this->get_logger(), "[SetMode] Saved map PCD before switching to localization");
      }
      // Save corrected maps via GraphSlam service (odom + corrected)
      if (this->callSaveCorrectedPCD())
      {
        RCLCPP_INFO(this->get_logger(), "[SetMode] Saved corrected maps before switching to localization");
      }
      // Save KFDB (raw + corrected)
      this->saveKeyframeDatabase();
      RCLCPP_INFO(this->get_logger(), "[SetMode] Saved KFDB before switching to localization");
      this->saveCorrectedKeyframeDatabase();
    }
  }

  // Update map_path if provided
  if (!req->map_path.empty())
  {
    this->map_path_ = req->map_path;
    RCLCPP_INFO(this->get_logger(), "[SetMode] Updated map_path to: %s", this->map_path_.c_str());
  }

  // If switching to localization and we have a map_path, ensure prior map is loaded
  if (req->mode == "localization" && !this->map_path_.empty() && !this->use_prior_map_)
  {
    std::ifstream f(this->map_path_);
    if (f.good())
    {
      f.close();
      this->use_prior_map_ = true;
      this->loadPriorMap();
      RCLCPP_INFO(this->get_logger(), "[SetMode] Loaded prior map for localization");
    }
    else
    {
      res->success = false;
      res->message = "Map file not found: " + this->map_path_;
      res->current_mode = this->map_mode_;
      return;
    }
  }

  this->map_mode_ = req->mode;

  // Update atexit handler for mapping mode
  if (req->mode == "mapping" && !this->map_path_.empty())
  {
    g_odom_node.store(this);
  }
  else
  {
    g_odom_node.store(nullptr);
  }

  // If switching to mapping: seed kfdb_entries_ from loaded sc_database_
  // so that saves include all prior entries + new ones
  if (req->mode == "mapping")
  {
    std::lock_guard<std::mutex> lock(this->kfdb_mutex_);
    if (this->kfdb_entries_.empty() && !this->sc_database_.empty())
    {
      this->kfdb_entries_ = this->sc_database_;
      RCLCPP_INFO(this->get_logger(),
                  "[SetMode] Seeded kfdb_entries_ with %zu entries from loaded KFDB",
                  this->kfdb_entries_.size());
    }
  }

  // If starting in mapping mode with existing KFDB, load and seed
  if (req->mode == "mapping" && !this->map_path_.empty())
  {
    std::lock_guard<std::mutex> lock(this->kfdb_mutex_);
    if (this->kfdb_entries_.empty())
    {
      std::string kfdb_path = this->getKfdbPath();
      float file_max_range = this->sc_max_range_;
      Eigen::Quaternionf file_gravity_q = this->kfdb_gravity_q_;
      std::vector<dlio::sc::ScanContextEntry> loaded;
      if (dlio::kfdb::load(kfdb_path, loaded, file_max_range, file_gravity_q))
      {
        this->kfdb_entries_ = std::move(loaded);
        this->kfdb_gravity_q_ = file_gravity_q;
        if (std::abs(file_max_range - this->sc_max_range_) > 0.1f)
          this->sc_max_range_ = file_max_range;
        RCLCPP_INFO(this->get_logger(),
                    "[Mapping] Loaded %zu existing KFDB entries for incremental mapping",
                    this->kfdb_entries_.size());
      }
    }
  }

  // Enable continuous localization in mapping mode (drift correction against prior map)
  if (req->mode == "mapping" && this->continuous_localize_ &&
      this->continuous_localize_on_mapping_ && this->use_prior_map_)
  {
    this->relocalized_ = true; // position known from localization phase

    if (!this->continuous_localize_timer_)
    {
      this->continuous_localize_timer_ = this->create_wall_timer(
          std::chrono::duration<double>(this->continuous_localize_interval_),
          std::bind(&dlio::OdomNode::continuousLocalize, this));

      if (!this->confidence_pub_)
        this->confidence_pub_ = this->create_publisher<std_msgs::msg::Float32>("localization_confidence", 10);

      if (!this->global_correction_sub_)
      {
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
                  this->bayes_consecutive_accepts_ = 0;
              }
            });
      }

      RCLCPP_INFO(this->get_logger(),
                  "[SetMode] Continuous localization enabled for mapping mode (interval=%.1fs)",
                  this->continuous_localize_interval_);
    }
  }

  res->success = true;
  res->message = "Mode changed from '" + old_mode + "' to '" + req->mode + "'";
  res->current_mode = this->map_mode_;
  RCLCPP_INFO(this->get_logger(), "[SetMode] %s", res->message.c_str());
}

void dlio::OdomNode::reloadPriorMapForRelocalization()
{
  if (this->prior_map_cloud_ && this->prior_map_cloud_->size() > 0)
  {
    // Resources still available, just reload SC database if needed
    {
      std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
      if (this->sc_database_.empty())
      {
        if (!this->loadKeyframeDatabase())
        {
          this->buildScanContextDatabase();
        }
      }
    }
    return;
  }

  // Need to reload from disk
  RCLCPP_INFO(this->get_logger(), "[Relocalize] Reloading prior map from: %s", this->map_path_.c_str());

  pcl::PointCloud<PointType>::Ptr cloud = std::make_shared<pcl::PointCloud<PointType>>();
  if (pcl::io::loadPCDFile(this->map_path_, *cloud) == -1)
  {
    RCLCPP_ERROR(this->get_logger(), "[Relocalize] Failed to load PCD file: %s", this->map_path_.c_str());
    return;
  }

  // Voxel filter
  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(this->map_voxel_size_, this->map_voxel_size_, this->map_voxel_size_);
  vg.setInputCloud(cloud);
  vg.filter(*cloud);

  this->prior_map_cloud_ = std::make_shared<pcl::PointCloud<PointType>>(*cloud);
  this->prior_map_kdtree_ = std::make_shared<nanoflann::KdTreeFLANN<PointType>>();
  this->prior_map_kdtree_->setInputCloud(this->prior_map_cloud_);

  RCLCPP_INFO(this->get_logger(), "[Relocalize] Prior map reloaded: %zu pts, KdTree built",
              this->prior_map_cloud_->size());

  // Load SC database (lock to prevent race with continuousLocalize iterating sc_database_)
  {
    std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
    this->sc_database_.clear();
    if (!this->loadKeyframeDatabase())
    {
      this->buildScanContextDatabase();
    }
  }
  RCLCPP_INFO(this->get_logger(), "[Relocalize] SC database: %zu entries", this->sc_database_.size());
}

void dlio::OdomNode::srvRelocalize(
    std::shared_ptr<direct_lidar_inertial_odometry::srv::Relocalize::Request> /*req*/,
    std::shared_ptr<direct_lidar_inertial_odometry::srv::Relocalize::Response> res)
{
  if (this->map_mode_ != "localization")
  {
    res->success = false;
    res->message = "Must be in localization mode to relocalize (current: " + this->map_mode_ + ")";
    return;
  }

  if (this->map_path_.empty())
  {
    res->success = false;
    res->message = "No map_path set. Use SetMode service to set map_path first.";
    return;
  }

  RCLCPP_INFO(this->get_logger(), "[Relocalize] Service called, reloading resources...");

  // Reload prior map and SC database if freed
  this->reloadPriorMapForRelocalization();

  if (!this->prior_map_cloud_ || this->prior_map_cloud_->empty())
  {
    res->success = false;
    res->message = "Failed to load prior map from: " + this->map_path_;
    return;
  }

  // Reset relocalization flags — callbackPointCloud will handle the actual SC+GICP
  this->use_prior_map_ = true;
  this->relocalize_ = true;
  this->relocalized_ = false;
  this->prior_map_pose_set_ = false;
  this->sc_attempt_count_ = 0;
  this->last_reloc_fitness_ = -1.0;

  RCLCPP_INFO(this->get_logger(), "[Relocalize] Flags reset, waiting for relocalization...");

  // Poll for relocalization result (up to 30 seconds)
  auto start = std::chrono::steady_clock::now();
  while (!this->relocalized_)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto elapsed = std::chrono::steady_clock::now() - start;
    if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 30)
    {
      res->success = false;
      res->message = "Relocalization timed out after 30 seconds (" +
                     std::to_string(this->sc_attempt_count_) + " attempts)";
      return;
    }
  }

  // Return result
  res->success = true;
  float yaw = std::atan2(
      2.f * (this->state.q.w() * this->state.q.z() + this->state.q.x() * this->state.q.y()),
      1.f - 2.f * (this->state.q.y() * this->state.q.y() + this->state.q.z() * this->state.q.z()));
  res->x = this->state.p[0];
  res->y = this->state.p[1];
  res->z = this->state.p[2];
  res->yaw_deg = yaw * 180.0 / M_PI;
  res->fitness_score = this->last_reloc_fitness_;
  res->message = "Relocalized successfully after " + std::to_string(this->sc_attempt_count_) + " attempt(s)";

  RCLCPP_INFO(this->get_logger(), "[Relocalize] %s — pos=[%.1f,%.1f,%.1f] yaw=%.1f fitness=%.4f",
              res->message.c_str(), res->x, res->y, res->z, res->yaw_deg, res->fitness_score);
}

void dlio::OdomNode::srvSetPose(
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SetPose::Request> req,
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SetPose::Response> res)
{
  // SetPose specifies desired position in MAP frame.
  // We NEVER jump odom — only update T_map_odom so that:
  //   map→body = T_map_odom × T_odom_body = desired pose
  // This keeps odom continuous, preserves accumulated keyframes/IMU state,
  // and ensures the odom-frame map stays consistent with the loaded PCD map.

  Eigen::Vector3f new_pos(req->x, req->y, req->z);
  float yaw_rad = req->yaw_deg * M_PI / 180.0;

  // Keep current roll/pitch from IMU, only override yaw
  Eigen::Matrix3f R;
  Eigen::Matrix4f T_odom_body;
  {
    std::lock_guard<std::mutex> state_lock(this->state_mtx_);
    R = this->state.q.toRotationMatrix();
    T_odom_body = this->T;
  }
  float roll = std::atan2(R(2, 1), R(2, 2));
  float pitch = std::asin(-R(2, 0));

  Eigen::Quaternionf new_q =
      Eigen::AngleAxisf(yaw_rad, Eigen::Vector3f::UnitZ()) *
      Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());

  // Build desired map→body transform
  Eigen::Matrix4f T_map_body_desired = Eigen::Matrix4f::Identity();
  T_map_body_desired.block<3, 3>(0, 0) = new_q.toRotationMatrix();
  T_map_body_desired.block<3, 1>(0, 3) = new_pos;

  // T_map_body = T_map_odom × T_odom_body  →  T_map_odom = T_map_body × T_odom_body⁻¹
  Eigen::Matrix4f T_map_odom_new = T_map_body_desired * T_odom_body.inverse();

  {
    std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
    this->T_map_odom_ = T_map_odom_new;
    this->bayes_posterior_.clear();
    this->bayes_consecutive_accepts_ = 0;
  }

  // Mark as relocalized if not already
  this->prior_map_pose_set_ = true;
  this->relocalized_ = true;

  Eigen::Vector3f t_corr = T_map_odom_new.block<3, 1>(0, 3);
  res->success = true;
  res->message = "Map pose set to [" + std::to_string(req->x) + ", " + std::to_string(req->y) + ", " +
                 std::to_string(req->z) + "] yaw=" + std::to_string(req->yaw_deg) + " deg";

  RCLCPP_INFO(this->get_logger(), "[SetPose] %s | T_map_odom=[%.2f,%.2f,%.2f] (odom untouched)",
              res->message.c_str(), t_corr[0], t_corr[1], t_corr[2]);
}

void dlio::OdomNode::srvGetState(
    std::shared_ptr<direct_lidar_inertial_odometry::srv::GetState::Request> /*req*/,
    std::shared_ptr<direct_lidar_inertial_odometry::srv::GetState::Response> res)
{
  res->mode = this->map_mode_;
  res->relocalized = this->relocalized_;

  // Report MAP-frame position (T_map_odom × T_odom_body), not odom-frame
  Eigen::Matrix4f T_odom_body;
  {
    std::lock_guard<std::mutex> state_lock(this->state_mtx_);
    T_odom_body = this->T;
  }

  Eigen::Matrix4f T_map_odom;
  {
    std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
    T_map_odom = this->T_map_odom_;
  }

  Eigen::Matrix4f T_map_body = T_map_odom * T_odom_body;
  Eigen::Vector3f map_pos = T_map_body.block<3, 1>(0, 3);
  Eigen::Matrix3f R = T_map_body.block<3, 3>(0, 0);

  res->x = map_pos[0];
  res->y = map_pos[1];
  res->z = map_pos[2];

  // Extract roll/pitch/yaw from map-frame orientation
  float roll = std::atan2(R(2, 1), R(2, 2));
  float pitch = std::asin(-R(2, 0));
  float yaw = std::atan2(R(1, 0), R(0, 0));

  res->roll_deg = roll * 180.0 / M_PI;
  res->pitch_deg = pitch * 180.0 / M_PI;
  res->yaw_deg = yaw * 180.0 / M_PI;
  res->length_traversed = this->length_traversed;
  res->num_keyframes = static_cast<int32_t>(this->keyframes.size());
}

bool dlio::OdomNode::callSavePCD()
{
  if (!this->save_pcd_client_->wait_for_service(std::chrono::seconds(2)))
  {
    RCLCPP_WARN(this->get_logger(), "[SavePCD] MapNode save_pcd service not available");
    return false;
  }

  auto request = std::make_shared<direct_lidar_inertial_odometry::srv::SavePCD::Request>();
  request->leaf_size = static_cast<float>(this->map_voxel_size_);

  std::filesystem::path p(this->map_path_);
  request->save_path = p.parent_path().string();

  RCLCPP_INFO(this->get_logger(), "[SavePCD] Requesting save_pcd (async): path='%s', leaf=%.2f",
              request->save_path.c_str(), request->leaf_size);

  // Use async callback to avoid deadlock (service-within-service in same executor)
  auto logger = this->get_logger();
  this->save_pcd_client_->async_send_request(request,
                                             [logger](rclcpp::Client<direct_lidar_inertial_odometry::srv::SavePCD>::SharedFuture future)
                                             {
                                               auto result = future.get();
                                               if (result->success)
                                               {
                                                 RCLCPP_INFO(logger, "[SavePCD] Map saved successfully");
                                               }
                                               else
                                               {
                                                 RCLCPP_WARN(logger, "[SavePCD] Map save failed");
                                               }
                                             });
  return true; // request sent, will complete asynchronously
}

bool dlio::OdomNode::callSaveCorrectedPCD()
{
  if (!this->save_corrected_pcd_client_->wait_for_service(std::chrono::seconds(2)))
  {
    RCLCPP_WARN(this->get_logger(), "[SavePCD] GraphSlam save_corrected_pcd service not available");
    return false;
  }

  auto request = std::make_shared<direct_lidar_inertial_odometry::srv::SavePCD::Request>();
  request->leaf_size = static_cast<float>(this->map_voxel_size_);

  std::filesystem::path p(this->map_path_);
  request->save_path = p.parent_path().string();

  RCLCPP_INFO(this->get_logger(), "[SavePCD] Requesting save_corrected_pcd (async): path='%s', leaf=%.2f",
              request->save_path.c_str(), request->leaf_size);

  // Use async callback to avoid deadlock (service-within-service in same executor)
  auto logger = this->get_logger();
  this->save_corrected_pcd_client_->async_send_request(request,
                                                       [logger](rclcpp::Client<direct_lidar_inertial_odometry::srv::SavePCD>::SharedFuture future)
                                                       {
                                                         auto result = future.get();
                                                         if (result->success)
                                                         {
                                                           RCLCPP_INFO(logger, "[SavePCD] Corrected map saved successfully");
                                                         }
                                                         else
                                                         {
                                                           RCLCPP_WARN(logger, "[SavePCD] Corrected map save failed");
                                                         }
                                                       });
  return true; // request sent, will complete asynchronously
}

void dlio::OdomNode::clearAllMapData()
{
  std::lock_guard<std::mutex> state_lock(this->state_mtx_);
  std::lock_guard<std::mutex> kf_lock(this->keyframes_mutex);

  // Clear keyframes
  this->keyframes.clear();
  this->keyframe_timestamps.clear();
  this->keyframe_normals.clear();
  this->keyframe_transformations.clear();
  this->num_processed_keyframes = 0;
  this->num_prior_keyframes_ = 0;

  // Clear submap
  this->submap_kf_idx_curr.clear();
  this->submap_kf_idx_prev.clear();
  this->keyframe_convex.clear();
  this->keyframe_concave.clear();

  // Clear SC/relocalization data (lock to prevent race with continuousLocalize)
  {
    std::lock_guard<std::mutex> cl_lock(this->continuous_localize_mtx_);
    this->sc_database_.clear();
  }
  {
    std::lock_guard<std::mutex> kfdb_lock(this->kfdb_mutex_);
    this->kfdb_entries_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(this->corrected_kf_poses_mutex_);
    this->corrected_kf_poses_.clear();
  }

  // Clear prior map
  this->prior_map_cloud_.reset();
  this->prior_map_kdtree_.reset();
  this->use_prior_map_ = false;
  this->relocalize_ = false;
  this->relocalized_ = false;
  this->prior_map_pose_set_ = false;
  this->sc_attempt_count_ = 0;

  // Clear trajectory
  this->trajectory.clear();
  this->length_traversed = 0.0;

  // Delete map files
  std::string pcd_path = this->map_path_;
  std::string kfdb_path = this->getKfdbPath();
  if (!pcd_path.empty() && std::filesystem::exists(pcd_path))
  {
    std::filesystem::remove(pcd_path);
    RCLCPP_INFO(this->get_logger(), "[NewMap] Deleted: %s", pcd_path.c_str());
  }
  if (!kfdb_path.empty() && std::filesystem::exists(kfdb_path))
  {
    std::filesystem::remove(kfdb_path);
    RCLCPP_INFO(this->get_logger(), "[NewMap] Deleted: %s", kfdb_path.c_str());
  }

  // Delete corrected files
  if (!pcd_path.empty())
  {
    size_t dot = pcd_path.rfind('.');
    std::string corrected_pcd = (dot != std::string::npos && pcd_path.substr(dot) == ".pcd")
                                    ? pcd_path.substr(0, dot) + "_corrected.pcd"
                                    : pcd_path + "_corrected.pcd";
    std::string corrected_kfdb = (dot != std::string::npos && pcd_path.substr(dot) == ".pcd")
                                     ? pcd_path.substr(0, dot) + "_corrected.kfdb"
                                     : pcd_path + "_corrected.kfdb";
    if (std::filesystem::exists(corrected_pcd))
    {
      std::filesystem::remove(corrected_pcd);
      RCLCPP_INFO(this->get_logger(), "[NewMap] Deleted: %s", corrected_pcd.c_str());
    }
    if (std::filesystem::exists(corrected_kfdb))
    {
      std::filesystem::remove(corrected_kfdb);
      RCLCPP_INFO(this->get_logger(), "[NewMap] Deleted: %s", corrected_kfdb.c_str());
    }
  }

  RCLCPP_INFO(this->get_logger(), "[NewMap] All map data cleared");
}

void dlio::OdomNode::srvNewMap(
    std::shared_ptr<direct_lidar_inertial_odometry::srv::NewMap::Request> /*req*/,
    std::shared_ptr<direct_lidar_inertial_odometry::srv::NewMap::Response> res)
{
  RCLCPP_INFO(this->get_logger(), "[NewMap] Clearing all map data and switching to mapping mode...");

  this->clearAllMapData();
  this->map_mode_ = "mapping";

  // Update atexit handler
  if (!this->map_path_.empty())
  {
    g_odom_node.store(this);
  }

  res->success = true;
  res->message = "All map data cleared, switched to mapping mode. Map will save to: " + this->map_path_;
  RCLCPP_INFO(this->get_logger(), "[NewMap] %s", res->message.c_str());
}

void dlio::OdomNode::srvNewMapWZero(
    std::shared_ptr<direct_lidar_inertial_odometry::srv::NewMapWZero::Request> /*req*/,
    std::shared_ptr<direct_lidar_inertial_odometry::srv::NewMapWZero::Response> res)
{
  RCLCPP_INFO(this->get_logger(), "[NewMapWZero] Clearing all map data, resetting pose, switching to mapping...");

  this->clearAllMapData();
  this->map_mode_ = "mapping";

  // Reset pose to origin
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);
  this->origin = Eigen::Vector3f(0., 0., 0.);

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();
  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  // Reset map→odom correction and Bayesian state
  {
    std::lock_guard<std::mutex> lock(this->continuous_localize_mtx_);
    this->T_map_odom_ = Eigen::Matrix4f::Identity();
    this->bayes_posterior_.clear();
    this->bayes_consecutive_accepts_ = 0;
  }

  // Update atexit handler
  if (!this->map_path_.empty())
  {
    g_odom_node.store(this);
  }

  res->success = true;
  res->message = "All map data cleared, pose reset to zero, switched to mapping mode. Map will save to: " + this->map_path_;
  RCLCPP_INFO(this->get_logger(), "[NewMapWZero] %s", res->message.c_str());
}

void dlio::OdomNode::debug()
{
  if (!this->geo.first_opt_done)
  {
    return;
  }

  if (!this->first_imu_received)
  {
    return;
  }

  // length_traversed is updated incrementally in callbackPointCloud

  // Average computation time
  double avg_comp_time =
      std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0) / this->comp_times.size();

  // Average sensor rates
  int win_size = 100;
  double avg_imu_rate;
  double avg_lidar_rate;
  if (this->imu_rates.size() < win_size)
  {
    avg_imu_rate =
        std::accumulate(this->imu_rates.begin(), this->imu_rates.end(), 0.0) / this->imu_rates.size();
  }
  else
  {
    avg_imu_rate =
        std::accumulate(this->imu_rates.end() - win_size, this->imu_rates.end(), 0.0) / win_size;
  }
  if (this->lidar_rates.size() < win_size)
  {
    avg_lidar_rate =
        std::accumulate(this->lidar_rates.begin(), this->lidar_rates.end(), 0.0) / this->lidar_rates.size();
  }
  else
  {
    avg_lidar_rate =
        std::accumulate(this->lidar_rates.end() - win_size, this->lidar_rates.end(), 0.0) / win_size;
  }

  // RAM Usage
  double vm_usage = 0.0;
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); // get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >> stime >> cutime >> cstime >> priority >> nice >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU)
  {
    cpu_percent = -1.0;
  }
  else
  {
    cpu_percent = (timeSample.tms_stime - this->lastSysCPU) + (timeSample.tms_utime - this->lastUserCPU);
    cpu_percent /= (now - this->lastCPU);
    cpu_percent /= this->numProcessors;
    cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;
  this->cpu_percents.push_back(cpu_percent);
  double avg_cpu_usage =
      std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) / this->cpu_percents.size();

  // Print to terminal
  printf("\033[2J\033[1;1H");

  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_ << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

  std::time_t curr_time = this->scan_stamp;
  std::string asc_time = std::asctime(std::localtime(&curr_time));
  asc_time.pop_back();
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
            << "Elapsed Time: " + to_string_with_precision(this->elapsed_time, 2) + " seconds "
            << "|" << std::endl;

  if (!this->cpu_type.empty())
  {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
              << this->cpu_type + " x " + std::to_string(this->numProcessors)
              << "|" << std::endl;
  }

  if (this->sensor == dlio::SensorType::OUSTER)
  {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
              << "Sensor Rates: Ouster @ " + to_string_with_precision(avg_lidar_rate, 2) + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
              << "|" << std::endl;
  }
  else if (this->sensor == dlio::SensorType::VELODYNE)
  {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
              << "Sensor Rates: Velodyne @ " + to_string_with_precision(avg_lidar_rate, 2) + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
              << "|" << std::endl;
  }
  else if (this->sensor == dlio::SensorType::HESAI)
  {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
              << "Sensor Rates: Hesai @ " + to_string_with_precision(avg_lidar_rate, 2) + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
              << "|" << std::endl;
  }
  else if (this->sensor == dlio::SensorType::LIVOX)
  {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
              << "Sensor Rates: Livox @ " + to_string_with_precision(avg_lidar_rate, 2) + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
              << "|" << std::endl;
  }
  else
  {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
              << "Sensor Rates: Unknown LiDAR @ " + to_string_with_precision(avg_lidar_rate, 2) + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
              << "|" << std::endl;
  }

  std::cout << "|===================================================================|" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Position     {W}  [xyz] :: " + to_string_with_precision(this->state.p[0], 4) + " " + to_string_with_precision(this->state.p[1], 4) + " " + to_string_with_precision(this->state.p[2], 4)
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Orientation  {W} [wxyz] :: " + to_string_with_precision(this->state.q.w(), 4) + " " + to_string_with_precision(this->state.q.x(), 4) + " " + to_string_with_precision(this->state.q.y(), 4) + " " + to_string_with_precision(this->state.q.z(), 4)
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.lin.b[0], 4) + " " + to_string_with_precision(this->state.v.lin.b[1], 4) + " " + to_string_with_precision(this->state.v.lin.b[2], 4)
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.ang.b[0], 4) + " " + to_string_with_precision(this->state.v.ang.b[1], 4) + " " + to_string_with_precision(this->state.v.ang.b[2], 4)
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Accel Bias        [xyz] :: " + to_string_with_precision(this->state.b.accel[0], 8) + " " + to_string_with_precision(this->state.b.accel[1], 8) + " " + to_string_with_precision(this->state.b.accel[2], 8)
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Gyro Bias         [xyz] :: " + to_string_with_precision(this->state.b.gyro[0], 8) + " " + to_string_with_precision(this->state.b.gyro[1], 8) + " " + to_string_with_precision(this->state.b.gyro[2], 8)
            << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Distance Traveled  :: " + to_string_with_precision(this->length_traversed, 4) + " meters"
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Distance to Origin :: " + to_string_with_precision(sqrt(pow(this->state.p[0] - this->origin[0], 2) + pow(this->state.p[1] - this->origin[1], 2) + pow(this->state.p[2] - this->origin[2], 2)), 4) + " meters"
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Registration       :: keyframes: " + std::to_string(this->keyframes.size()) + ", " + "deskewed points: " + std::to_string(this->deskew_size)
            << "|" << std::endl;
  std::cout << "|                                                                   |" << std::endl;

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: "
            << std::setfill(' ') << std::setw(6) << this->comp_times.back() * 1000. << " ms    // Avg: "
            << std::setw(6) << avg_comp_time * 1000. << " / Max: "
            << std::setw(6) << *std::max_element(this->comp_times.begin(), this->comp_times.end()) * 1000.
            << "     |" << std::endl;
  std::cout << "| Cores Utilized   :: "
            << std::setfill(' ') << std::setw(6) << (cpu_percent / 100.) * this->numProcessors << " cores // Avg: "
            << std::setw(6) << (avg_cpu_usage / 100.) * this->numProcessors << " / Max: "
            << std::setw(6) << (*std::max_element(this->cpu_percents.begin(), this->cpu_percents.end()) / 100.) * this->numProcessors
            << "     |" << std::endl;
  std::cout << "| CPU Load         :: "
            << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
            << std::setw(6) << avg_cpu_usage << " / Max: "
            << std::setw(6) << *std::max_element(this->cpu_percents.begin(), this->cpu_percents.end())
            << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "RAM Allocation   :: " + to_string_with_precision(resident_set / 1000., 2) + " MB"
            << "|" << std::endl;

  std::cout << "+-------------------------------------------------------------------+" << std::endl;
}

void dlio::OdomNode::publishEarthToMapTF()
{
  // Get ECEF coordinates of map origin
  GeographicLib::Geocentric earth(GeographicLib::Constants::WGS84_a(),
                                  GeographicLib::Constants::WGS84_f());
  double X0, Y0, Z0;
  earth.Forward(this->gps_origin_lat_, this->gps_origin_lon_,
                this->gps_origin_alt_, X0, Y0, Z0);

  // Rotation matrix: ECEF → ENU at origin
  double lat_rad = this->gps_origin_lat_ * M_PI / 180.0;
  double lon_rad = this->gps_origin_lon_ * M_PI / 180.0;
  double sl = std::sin(lat_rad), cl = std::cos(lat_rad);
  double sn = std::sin(lon_rad), cn = std::cos(lon_rad);

  Eigen::Matrix3d R;
  R << -sn, cn, 0.0,
      -sl * cn, -sl * sn, cl,
      cl * cn, cl * sn, sl;

  Eigen::Quaterniond q(R);

  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = this->now();
  tf.header.frame_id = "earth";
  tf.child_frame_id = this->map_frame_;
  tf.transform.translation.x = X0;
  tf.transform.translation.y = Y0;
  tf.transform.translation.z = Z0;
  tf.transform.rotation.x = q.x();
  tf.transform.rotation.y = q.y();
  tf.transform.rotation.z = q.z();
  tf.transform.rotation.w = q.w();

  this->static_tf_broadcaster_->sendTransform(tf);
  RCLCPP_INFO(this->get_logger(),
              "[GPS] Published earth->%s static TF (ECEF=[%.1f,%.1f,%.1f])",
              this->map_frame_.c_str(), X0, Y0, Z0);
}

void dlio::OdomNode::continuousLocalize()
{
  // Prerequisites
  if (!this->dlio_initialized || !this->relocalized_)
    return;

  // Brief lock: copy shared state, then release for heavy computation
  std::vector<dlio::sc::ScanContextEntry> sc_snap;
  std::vector<float> bayes_snap;
  Eigen::Matrix4f T_map_odom_snap;
  pcl::PointCloud<PointType>::ConstPtr prior_cloud_snap;
  std::shared_ptr<nanoflann::KdTreeFLANN<PointType>> prior_kdtree_snap;
  int bayes_consec;
  {
    std::lock_guard<std::mutex> cl_lock(this->continuous_localize_mtx_);
    if (!this->prior_map_cloud_ || !this->prior_map_kdtree_)
      return;
    if (this->sc_database_.empty())
      return;
    sc_snap = this->sc_database_;
    bayes_snap = this->bayes_posterior_;
    T_map_odom_snap = this->T_map_odom_;
    prior_cloud_snap = this->prior_map_cloud_;
    prior_kdtree_snap = this->prior_map_kdtree_;
    bayes_consec = this->bayes_consecutive_accepts_;
  }
  // --- All computation below uses local copies, no lock held ---

  // 1. Get latest scan (body frame) + T_odom_body
  pcl::PointCloud<PointType>::ConstPtr scan_body;
  Eigen::Matrix4f T_odom_body;
  double scan_time;
  {
    std::lock_guard<std::mutex> lock(this->latest_scan_mtx_);
    scan_body = this->latest_scan_;
    T_odom_body = this->latest_scan_T_;
    scan_time = this->latest_scan_time_;
  }
  if (!scan_body || scan_body->empty())
  {
    return;
  }

  // Skip if scan is stale (no new data arriving)
  double now_sec = this->now().seconds();
  double scan_age = now_sec - scan_time;
  if (scan_time > 0.0 && scan_age > 3.0 * this->continuous_localize_interval_)
  {
    return;
  }

  const int N = static_cast<int>(sc_snap.size());

  // Precompute lidar extrinsic here for GPS path (also used later in SC++ path)
  Eigen::Matrix4f T_body_lidar = this->extrinsics.baselink2lidar_T;

  // ── GPS-based loop closure (skip SC++ if GPS available) ─────────────
  if (this->gps_enabled_ && this->gps_origin_set_ && this->enable_global_correction_.load())
  {
    GPSMeasurement current_gps;
    bool gps_time_ok = this->getGPSAtTime(scan_time, current_gps);
    bool gps_acc_ok = gps_time_ok && (this->gps_trust_all_ ||
                                      current_gps.horizontal_accuracy < this->gps_min_accuracy_);

    if (!gps_time_ok && this->debug_)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                           "[GPS] No GPS fix near scan time (buffer=%zu)",
                           this->gps_buffer_.size());
    }
    else if (!gps_acc_ok && this->debug_)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                           "[GPS] Accuracy rejected: h_acc=%.2f > %.2f",
                           current_gps.horizontal_accuracy, this->gps_min_accuracy_);
    }

    if (gps_acc_ok)
    {

      float gps_x, gps_y, gps_z;
      if (this->gpsToLocal(current_gps.latitude, current_gps.longitude,
                           current_gps.altitude, gps_x, gps_y, gps_z))
      {

        Eigen::Vector3f gps_pos(gps_x, gps_y, gps_z);

        // Find KFs within search radius using stored GPS lat/lon (same frame)
        // NOTE: sc_snap[i].position is in map frame, gps_pos is in local ENU —
        // these are different frames! Must use stored GPS coords for comparison.
        std::vector<std::pair<int, float>> gps_candidates;
        for (int i = 0; i < N; i++)
        {
          if (!sc_snap[i].gps_valid)
            continue;

          float kf_x, kf_y, kf_z;
          if (!this->gpsToLocal(sc_snap[i].gps_latitude, sc_snap[i].gps_longitude,
                                sc_snap[i].gps_altitude, kf_x, kf_y, kf_z))
            continue;

          float dist = std::sqrt((kf_x - gps_x) * (kf_x - gps_x) +
                                 (kf_y - gps_y) * (kf_y - gps_y));
          if (dist < this->gps_search_radius_)
            gps_candidates.push_back({i, dist});
        }

        // Sort by distance (closest first)
        std::sort(gps_candidates.begin(), gps_candidates.end(),
                  [](auto &a, auto &b)
                  { return a.second < b.second; });

        if (!gps_candidates.empty())
        {
          Eigen::Matrix4f T_map_odom_cur = T_map_odom_snap;
          Eigen::Matrix4f T_map_body_est = T_map_odom_cur * T_odom_body;
          Eigen::Matrix4f T_map_lidar_est = T_map_body_est * T_body_lidar;

          // GICP lambda (same as main path but defined locally for GPS scope)
          auto tryGICP_gps = [&](const Eigen::Vector3f &center, const Eigen::Matrix4f &init_guess,
                                 Eigen::Matrix4f &result_T, float &result_fitness) -> bool
          {
            PointType search_pt;
            search_pt.x = center[0];
            search_pt.y = center[1];
            search_pt.z = center[2];

            std::vector<int> nn_indices;
            std::vector<float> nn_dists;
            prior_kdtree_snap->radiusSearch(search_pt, 35.f * 35.f, nn_indices, nn_dists);

            if (nn_indices.size() < 50)
            {
              RCLCPP_DEBUG(this->get_logger(),
                           "[GPS] tryGICP: too few pts=%zu at [%.1f,%.1f,%.1f]",
                           nn_indices.size(), center[0], center[1], center[2]);
              return false;
            }

            pcl::PointCloud<PointType>::Ptr local_map = std::make_shared<pcl::PointCloud<PointType>>();
            local_map->points.resize(nn_indices.size());
            for (size_t i = 0; i < nn_indices.size(); i++)
              local_map->points[i] = prior_cloud_snap->points[nn_indices[i]];
            local_map->width = local_map->points.size();
            local_map->height = 1;
            local_map->is_dense = true;

            pcl::VoxelGrid<PointType> vf;
            vf.setLeafSize(0.5f, 0.5f, 0.5f);
            vf.setInputCloud(local_map);
            vf.filter(*local_map);
            local_map->width = local_map->points.size();
            local_map->height = 1;

            pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
            bool converged = false;

            if (this->use_gicp_)
            {
              nano_gicp::NanoGICP<PointType, PointType> gicp;
              gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
              gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
              gicp.setMaximumIterations(32);
              gicp.setTransformationEpsilon(0.01);
              gicp.setRotationEpsilon(0.01);
              gicp.setInputSource(scan_body);
              gicp.calculateSourceCovariances();
              gicp.setInputTarget(local_map);
              gicp.calculateTargetCovariances();
              gicp.align(*aligned, init_guess);
              converged = gicp.hasConverged();
              result_fitness = gicp.getFitnessScore(1.0);
              result_T = gicp.getFinalTransformation();
            }
            else
            {
              pclomp::NormalDistributionsTransform<PointType, PointType> ndt_local;
              ndt_local.setResolution(this->ndt_resolution_);
              ndt_local.setNumThreads(this->ndt_num_threads_);
              ndt_local.setNeighborhoodSearchMethod(pclomp::DIRECT7);
              ndt_local.setMaximumIterations(32);
              ndt_local.setTransformationEpsilon(0.01);
              ndt_local.setInputSource(scan_body);
              ndt_local.setInputTarget(local_map);
              ndt_local.align(*aligned, init_guess);
              converged = ndt_local.hasConverged();
              result_fitness = ndt_local.getFitnessScore(1.0);
              result_T = ndt_local.getFinalTransformation();
            }

            if (!converged)
            {
              RCLCPP_DEBUG(this->get_logger(), "[GPS] tryGICP: not converged");
              return false;
            }
            if (result_fitness > this->continuous_localize_fitness_thresh_)
            {
              RCLCPP_DEBUG(this->get_logger(),
                           "[GPS] tryGICP: fitness=%.4f > thresh=%.4f",
                           result_fitness, this->continuous_localize_fitness_thresh_);
              return false;
            }

            float disp = (result_T.block<3, 1>(0, 3) - init_guess.block<3, 1>(0, 3)).norm();
            if (disp > 10.0f)
            {
              RCLCPP_DEBUG(this->get_logger(), "[GPS] tryGICP: disp=%.2fm > 10m", disp);
              return false;
            }

            return true;
          };

          int n_try = std::min(5, static_cast<int>(gps_candidates.size()));

          RCLCPP_DEBUG(this->get_logger(),
                       "[GPS] %zu KFs within %.0fm of GPS [%.1f,%.1f], trying top %d",
                       gps_candidates.size(), this->gps_search_radius_,
                       gps_x, gps_y, n_try);

          // Extract roll/pitch from IMU (reliable), try multiple yaws
          Eigen::Matrix3f R_est = T_map_body_est.block<3, 3>(0, 0);
          float est_yaw = std::atan2(R_est(1, 0), R_est(0, 0));
          Eigen::Quaternionf q_est(R_est);
          Eigen::Quaternionf q_yaw_inv(Eigen::AngleAxisf(-est_yaw, Eigen::Vector3f::UnitZ()));
          Eigen::Quaternionf q_rp = q_yaw_inv * q_est; // roll+pitch only

          const float yaw_offsets[] = {0.f, M_PI / 3.f, 2.f * M_PI / 3.f,
                                       M_PI, -2.f * M_PI / 3.f, -M_PI / 3.f};

          bool gps_gicp_ok = false;
          Eigen::Matrix4f gps_T_map_lidar;
          float gps_fitness = std::numeric_limits<float>::max();
          int gps_best_kf = -1;

          for (int ci = 0; ci < n_try; ci++)
          {
            int kf_idx = gps_candidates[ci].first;
            Eigen::Vector3f cand_pos = sc_snap[kf_idx].position;

            for (float yaw_offset : yaw_offsets)
            {
              float yaw = est_yaw + yaw_offset;
              Eigen::Quaternionf q_full =
                  Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ())) * q_rp;

              Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
              init_guess.block<3, 3>(0, 0) = q_full.toRotationMatrix();
              init_guess.block<3, 1>(0, 3) = cand_pos;
              init_guess = init_guess * T_body_lidar; // body→lidar

              Eigen::Matrix4f result_T;
              float result_fit;
              bool ok = tryGICP_gps(cand_pos, init_guess, result_T, result_fit);

              if (ok && result_fit < gps_fitness)
              {
                gps_fitness = result_fit;
                gps_T_map_lidar = result_T;
                gps_best_kf = kf_idx;
                gps_gicp_ok = true;

                // Early exit on excellent fitness
                if (result_fit < 0.05f)
                  goto gps_search_done;
              }
            }
          }
        gps_search_done:

          if (gps_gicp_ok)
          {
            Eigen::Matrix4f T_map_body_gps = gps_T_map_lidar * T_body_lidar.inverse();

            // Consistency check: matched KF's GPS should be close to current GPS
            // (both in local ENU, same frame — safe to compare)
            float kf_gps_x, kf_gps_y, kf_gps_z;
            float gps_kf_dist = 0.0f;
            if (sc_snap[gps_best_kf].gps_valid &&
                this->gpsToLocal(sc_snap[gps_best_kf].gps_latitude,
                                 sc_snap[gps_best_kf].gps_longitude,
                                 sc_snap[gps_best_kf].gps_altitude,
                                 kf_gps_x, kf_gps_y, kf_gps_z))
            {
              gps_kf_dist = std::sqrt((kf_gps_x - gps_x) * (kf_gps_x - gps_x) +
                                      (kf_gps_y - gps_y) * (kf_gps_y - gps_y));
            }

            Eigen::Matrix4f T_map_odom_new = T_map_body_gps * T_odom_body.inverse();
            Eigen::Matrix4f delta = T_map_odom_new * T_map_odom_cur.inverse();
            float corr_dist = delta.block<3, 1>(0, 3).norm();

            RCLCPP_INFO(this->get_logger(),
                        "[GPS] OK: kf=%d fitness=%.4f corr=%.3fm gps_kf_dist=%.1fm",
                        gps_best_kf, gps_fitness, corr_dist, gps_kf_dist);

            {
              std::lock_guard<std::mutex> wb(this->continuous_localize_mtx_);
              this->T_map_odom_ = T_map_odom_new;
            }

            float conf = std::max(0.0f, 1.0f - gps_fitness /
                                                   static_cast<float>(this->continuous_localize_fitness_thresh_));
            this->last_confidence_ = conf;
            if (this->confidence_pub_)
            {
              std_msgs::msg::Float32 msg;
              msg.data = conf;
              this->confidence_pub_->publish(msg);
            }

            return; // GPS path done, skip SC++
          }

          if (this->debug_)
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                                 "[GPS] GICP failed on %d candidates, falling back to SC++", n_try);
        }
        else if (this->debug_)
        {
          // Count how many KFs have GPS data at all
          int n_gps_valid = 0;
          for (int i = 0; i < N; i++)
            if (sc_snap[i].gps_valid)
              n_gps_valid++;
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                               "[GPS] No KFs within %.0fm of GPS [%.1f,%.1f] "
                               "(gps_valid_kfs=%d/%d), falling back to SC++",
                               this->gps_search_radius_, gps_x, gps_y,
                               n_gps_valid, N);
        }
      }
    }
  }
  // ── End GPS path, continue with SC++ below ──────────────────────────

  // Initialize Bayesian posterior if needed (lazy init / size change)
  if (static_cast<int>(bayes_snap.size()) != N + 1)
  {
    bayes_snap.resize(N + 1);
    bayes_snap[0] = 1.0f; // virtual place = certain "new place"
    for (int i = 1; i <= N; i++)
      bayes_snap[i] = 0.0f;
    bayes_consec = 0;
    if (this->debug_)
      RCLCPP_INFO(this->get_logger(), "[bayes] initialized posterior: N=%d keyframes", N);
  }

  // ── 2. Compute SC likelihoods ──────────────────────────────────────────

  // Rotate scan to gravity-aligned frame for SC descriptor (with ground removal)
  auto sc_scan = dlio::sc::prepareGravityAlignedScan(
      scan_body, this->kfdb_gravity_q_, this->extrinsics.baselink2lidar.R,
      this->sc_ground_height_threshold_);

  auto sc_desc = dlio::sc::computeScanContext(sc_scan, this->sc_max_range_);
  auto query_sector_key = dlio::sc::computeSectorKey(sc_desc);

  // Raw likelihoods: invert SC distance (lower distance = better match = higher likelihood)
  // Two filters applied:
  //   1. SC distance threshold: reject candidates with dist > threshold (too dissimilar)
  //   2. Top-K: only keep the K best matches, zero out the rest
  std::vector<float> raw_likelihood(N);
  std::vector<float> sc_distances(N);
  std::vector<int> sc_shifts(N); // best column shift per candidate (heading alignment)
  float best_raw = 0.0f;
  int best_raw_idx = 0;
  float worst_raw = 1.0f;
  int n_dist_rejected = 0;
  for (int i = 0; i < N; i++)
  {
    auto [dist, shift] = dlio::sc::computeScanContextDistance(
        sc_desc, sc_snap[i].descriptor,
        query_sector_key, sc_snap[i].sector_key,
        this->sc_search_window_);
    sc_distances[i] = dist;
    sc_shifts[i] = shift;

    // Filter 1: SC distance threshold — reject if too dissimilar
    if (this->bayes_sc_dist_threshold_ > 0.0f && dist > this->bayes_sc_dist_threshold_)
    {
      raw_likelihood[i] = 0.0f;
      n_dist_rejected++;
    }
    else
    {
      raw_likelihood[i] = 1.0f / (1.0f + dist);
    }

    if (raw_likelihood[i] > best_raw)
    {
      best_raw = raw_likelihood[i];
      best_raw_idx = i;
    }
    if (raw_likelihood[i] < worst_raw)
      worst_raw = raw_likelihood[i];
  }

  // Filter 2: Top-K — only keep K smallest SC distances, zero out the rest
  if (this->bayes_sc_top_k_ > 0 && this->bayes_sc_top_k_ < N)
  {
    // Collect indices of non-zero candidates, sorted by SC distance (ascending)
    std::vector<int> candidates;
    candidates.reserve(N);
    for (int i = 0; i < N; i++)
    {
      if (raw_likelihood[i] > 0.0f)
        candidates.push_back(i);
    }

    if (static_cast<int>(candidates.size()) > this->bayes_sc_top_k_)
    {
      // Partial sort: move K smallest distances to front
      std::nth_element(candidates.begin(),
                       candidates.begin() + this->bayes_sc_top_k_,
                       candidates.end(),
                       [&](int a, int b)
                       { return sc_distances[a] < sc_distances[b]; });

      // Build set of kept indices
      std::vector<bool> keep(N, false);
      for (int k = 0; k < this->bayes_sc_top_k_; k++)
        keep[candidates[k]] = true;

      for (int i = 0; i < N; i++)
      {
        if (!keep[i])
          raw_likelihood[i] = 0.0f;
      }
    }
  }

  // ── 3. Angeli normalization (adapted for Scan Context) ───────────────
  //
  // NOTE: RTAB-Map's Angeli VP formula (mean/stddev + 1.0) assumes visual features
  // where most places score 0 (no shared features). SC always produces non-zero
  // distances, so mean/stddev is huge (~14x) making VP unbeatable.
  // Fix: VP adjusted = 1.0 (neutral), null hypothesis strength comes from prediction only.
  // Keyframe normalization still uses Angeli (promote statistical outliers).

  std::vector<float> nonzero_vals;
  nonzero_vals.reserve(N);
  for (int i = 0; i < N; i++)
  {
    if (raw_likelihood[i] > 0.0f)
      nonzero_vals.push_back(raw_likelihood[i]);
  }

  // adjusted_likelihood: [0]=virtual place, [1..N]=keyframes
  std::vector<float> adjusted(N + 1, 1.0f);
  float angeli_mean = 0.0f, angeli_stddev = 0.0f;
  int num_promoted = 0;

  if (nonzero_vals.size() >= 3)
  {
    // Enough candidates for statistical normalization (Angeli)
    float sum_vals = 0.0f;
    for (float v : nonzero_vals)
      sum_vals += v;
    angeli_mean = sum_vals / static_cast<float>(nonzero_vals.size());

    float sum_sq = 0.0f;
    for (float v : nonzero_vals)
      sum_sq += (v - angeli_mean) * (v - angeli_mean);
    angeli_stddev = std::sqrt(sum_sq / static_cast<float>(nonzero_vals.size()));

    float epsilon = 0.0001f;
    if (angeli_stddev > epsilon)
    {
      for (int i = 0; i < N; i++)
      {
        if (raw_likelihood[i] > angeli_mean + angeli_stddev)
        {
          adjusted[i + 1] = (raw_likelihood[i] - angeli_mean) / angeli_stddev;
          num_promoted++;
        }
        else
          adjusted[i + 1] = 1.0f; // neutral
      }
    }
  }
  else if (!nonzero_vals.empty())
  {
    // Too few candidates for Angeli — use raw likelihood directly as boost.
    // Candidates already passed distance+topK filters, so they're good matches.
    // Scale: raw_likelihood is in [0,1], boost the survivors relative to VP (1.0).
    for (int i = 0; i < N; i++)
    {
      if (raw_likelihood[i] > 0.0f)
      {
        // Boost = raw * N so that a few strong candidates dominate over VP
        adjusted[i + 1] = raw_likelihood[i] * static_cast<float>(N);
        num_promoted++;
      }
      // else stays 1.0 (neutral)
    }
  }
  // VP adjusted stays 1.0 (neutral) — SC is dense, not sparse like visual features
  adjusted[0] = 1.0f;

  // ── 4. Bayesian update (recursive prediction) ──────────────────────────
  //
  // Prediction: simple Markov transition model
  //   VP  → VP:  self_loop (0.9)    VP  → kf_i: (1-self_loop)/N
  //   kf_i→ kf_i: self_loop (0.9)   kf_i→ VP:   (1-self_loop)
  // This ensures VP probability decays as evidence accumulates (unlike fixed prior).

  std::vector<float> prior(N + 1);
  float self_loop = this->bayes_virtual_place_prior_; // reuse as self-loop probability

  float prev_vp = bayes_snap[0];
  float sum_real_posterior = 0.0f;
  for (int i = 1; i <= N; i++)
    sum_real_posterior += bayes_snap[i];

  // VP prediction: stays VP + keyframes escaping to VP
  prior[0] = self_loop * prev_vp + (1.0f - self_loop) * sum_real_posterior;

  // Keyframe prediction: stays same + VP leaking to keyframes
  float vp_to_kf = (1.0f - self_loop) * prev_vp / static_cast<float>(N);
  for (int i = 1; i <= N; i++)
    prior[i] = self_loop * bayes_snap[i] + vp_to_kf;

  // Update step: posterior = likelihood × prior, then normalize
  float sum_posterior = 0.0f;
  for (int i = 0; i <= N; i++)
  {
    bayes_snap[i] = adjusted[i] * prior[i];
    sum_posterior += bayes_snap[i];
  }
  if (sum_posterior > 0.0f)
  {
    for (int i = 0; i <= N; i++)
      bayes_snap[i] /= sum_posterior;
  }

  // ── 5. Hypothesis check ────────────────────────────────────────────────

  float P_loop = 1.0f - bayes_snap[0];

  // Find top keyframes by posterior (for multi-candidate GICP in Stage 2)
  struct BayesCandidate
  {
    int kf_idx;
    float posterior;
    int sc_shift; // SC column shift → heading alignment
  };
  std::vector<BayesCandidate> top_candidates;
  top_candidates.reserve(N);
  for (int i = 1; i <= N; i++)
  {
    if (bayes_snap[i] > 0.0f)
      top_candidates.push_back({i - 1, bayes_snap[i], sc_shifts[i - 1]});
  }
  std::sort(top_candidates.begin(), top_candidates.end(),
            [](const BayesCandidate &a, const BayesCandidate &b)
            { return a.posterior > b.posterior; });

  int best_kf_idx = top_candidates.empty() ? 0 : top_candidates[0].kf_idx;
  float best_kf_posterior = top_candidates.empty() ? 0.0f : top_candidates[0].posterior;

  if (this->debug_)
    RCLCPP_INFO(this->get_logger(),
                "[bayes] SC: best=%.4f(kf%d,d=%.3f) worst=%.4f dist_rej=%d topK=%d | "
                "Angeli: mean=%.4f std=%.4f promoted=%d/%d | "
                "P_vp=%.4f P_loop=%.4f best_kf=%d(P=%.4f) prior_vp=%.4f thr=%.2f consec=%d/%d",
                best_raw, best_raw_idx, sc_distances[best_raw_idx], worst_raw,
                n_dist_rejected, this->bayes_sc_top_k_,
                angeli_mean, angeli_stddev, num_promoted, N,
                bayes_snap[0], P_loop, best_kf_idx, best_kf_posterior,
                prior[0], this->bayes_loop_threshold_,
                bayes_consec, this->bayes_min_consecutive_);

  bool p_loop_pass = (P_loop > this->bayes_loop_threshold_);

  if (!p_loop_pass)
  {
    // Virtual place wins — no GICP needed. SC++ Stage 2 requires P_loop > threshold.
    bayes_consec = 0;

    // Write back Bayesian state
    {
      std::lock_guard<std::mutex> wb(this->continuous_localize_mtx_);
      this->bayes_posterior_ = bayes_snap;
      this->bayes_consecutive_accepts_ = bayes_consec;
    }

    this->last_confidence_ = P_loop * 0.5f;
    if (this->confidence_pub_)
    {
      std_msgs::msg::Float32 msg;
      msg.data = this->last_confidence_;
      this->confidence_pub_->publish(msg);
    }
    return;
  }

  // ── GICP verification (SC++ pipeline) ──────────────────────────────────
  //
  // Stage 1 (drift correction): GICP at current estimated position.
  // Stage 2 (SC++ relocalization): If Stage 1 fails, try top-K Bayesian
  //   candidates × 6 yaw offsets (SC shift → base yaw).
  // Stage 3 (consistency check): Re-run GICP from Stage 2 result to catch
  //   false positives (GICP local minima).

  Eigen::Matrix4f T_map_odom_cur = T_map_odom_snap;

  Eigen::Matrix4f T_map_body_est = T_map_odom_cur * T_odom_body;
  Eigen::Matrix4f T_map_lidar_est = T_map_body_est * T_body_lidar; // lidar→body→odom→map
  Eigen::Vector3f est_pos = T_map_body_est.block<3, 1>(0, 3);

  // GICP at a given search center + initial guess. Returns success, result transform, fitness.
  auto tryGICP = [&](const Eigen::Vector3f &center, const Eigen::Matrix4f &init_guess,
                     Eigen::Matrix4f &result_T, float &result_fitness) -> bool
  {
    PointType search_pt;
    search_pt.x = center[0];
    search_pt.y = center[1];
    search_pt.z = center[2];

    std::vector<int> nn_indices;
    std::vector<float> nn_dists;
    prior_kdtree_snap->radiusSearch(search_pt, 35.f * 35.f, nn_indices, nn_dists);

    if (nn_indices.size() < 50)
    {
      if (this->debug_)
        RCLCPP_WARN(this->get_logger(),
                    "[bayes] tryGICP FAIL: too few neighbors=%zu at [%.1f,%.1f,%.1f], prior_map=%zu pts",
                    nn_indices.size(), center[0], center[1], center[2], prior_cloud_snap->size());
      return false;
    }

    pcl::PointCloud<PointType>::Ptr local_map = std::make_shared<pcl::PointCloud<PointType>>();
    local_map->points.resize(nn_indices.size());
    for (size_t i = 0; i < nn_indices.size(); i++)
      local_map->points[i] = prior_cloud_snap->points[nn_indices[i]];
    local_map->width = local_map->points.size();
    local_map->height = 1;
    local_map->is_dense = true;

    pcl::VoxelGrid<PointType> vf;
    vf.setLeafSize(0.5f, 0.5f, 0.5f);
    vf.setInputCloud(local_map);
    vf.filter(*local_map);
    local_map->width = local_map->points.size();
    local_map->height = 1;

    pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
    bool converged = false;

    if (this->use_gicp_)
    {
      nano_gicp::NanoGICP<PointType, PointType> gicp;
      gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
      gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
      gicp.setMaximumIterations(32);
      gicp.setTransformationEpsilon(0.01);
      gicp.setRotationEpsilon(0.01);

      gicp.setInputSource(scan_body);
      gicp.calculateSourceCovariances();
      gicp.setInputTarget(local_map);
      gicp.calculateTargetCovariances();

      gicp.align(*aligned, init_guess);
      converged = gicp.hasConverged();
      result_fitness = gicp.getFitnessScore(1.0);
      result_T = gicp.getFinalTransformation();
    }
    else
    {
      pclomp::NormalDistributionsTransform<PointType, PointType> ndt_local;
      ndt_local.setResolution(this->ndt_resolution_);
      ndt_local.setNumThreads(this->ndt_num_threads_);
      ndt_local.setNeighborhoodSearchMethod(pclomp::DIRECT7);
      ndt_local.setMaximumIterations(32);
      ndt_local.setTransformationEpsilon(0.01);

      ndt_local.setInputSource(scan_body);
      ndt_local.setInputTarget(local_map);

      ndt_local.align(*aligned, init_guess);
      converged = ndt_local.hasConverged();
      result_fitness = ndt_local.getFitnessScore(1.0);
      result_T = ndt_local.getFinalTransformation();
    }

    if (!converged)
    {
      if (this->debug_)
        RCLCPP_WARN(this->get_logger(),
                    "[bayes] tryGICP FAIL: not converged at [%.1f,%.1f,%.1f], scan=%zu pts, local_map=%zu pts",
                    center[0], center[1], center[2], scan_body->size(), local_map->size());
      return false;
    }

    if (result_fitness > this->continuous_localize_fitness_thresh_)
    {
      if (this->debug_)
        RCLCPP_WARN(this->get_logger(),
                    "[bayes] tryGICP FAIL: fitness=%.4f > thresh=%.4f at [%.1f,%.1f,%.1f], scan=%zu, local_map=%zu",
                    result_fitness, this->continuous_localize_fitness_thresh_,
                    center[0], center[1], center[2], scan_body->size(), local_map->size());
      return false;
    }

    // Displacement check: reject if GICP moved too far from init guess
    float disp = (result_T.block<3, 1>(0, 3) - init_guess.block<3, 1>(0, 3)).norm();
    if (disp > 10.0f)
    {
      if (this->debug_)
        RCLCPP_WARN(this->get_logger(),
                    "[bayes] tryGICP FAIL: displacement=%.2fm > 10m from init guess at [%.1f,%.1f,%.1f]",
                    disp, center[0], center[1], center[2]);
      return false;
    }

    return true;
  };

  bool gicp_ok = false;
  bool is_reloc = false;
  Eigen::Matrix4f T_map_lidar_gicp;
  float fitness = 0.0f;

  // ── Stage 1: GICP at estimated position (drift correction) ──
  // scan_body is in lidar frame → init_guess must be T_map←lidar
  {
    gicp_ok = tryGICP(est_pos, T_map_lidar_est, T_map_lidar_gicp, fitness);

    if (gicp_ok)
    {
      if (this->debug_)
        RCLCPP_INFO(this->get_logger(), "[bayes] Stage1 GICP OK: fitness=%.4f at est=[%.1f,%.1f,%.1f]",
                    fitness, est_pos[0], est_pos[1], est_pos[2]);
    }
  }

  // ── Stage 2: SC++ top-K candidate relocalization ──────────────────────
  //   Uses Bayesian-ranked top candidates from SC++ matching.
  //   For each candidate: SC shift → base yaw, try 6 yaw offsets, GICP.
  //   Requires P_loop > threshold AND enable_global_correction.
  if (!gicp_ok && this->enable_global_correction_.load())
  {
    if (this->debug_)
      RCLCPP_INFO(this->get_logger(), "[bayes] Stage1 failed → SC++ Stage2 candidate search...");

    // Extract roll/pitch from IMU (reliable), will test different yaws
    Eigen::Matrix3f R_est = T_map_body_est.block<3, 3>(0, 0);
    float est_yaw = std::atan2(R_est(1, 0), R_est(0, 0));
    Eigen::Quaternionf q_est(R_est);
    Eigen::Quaternionf q_yaw_inv(Eigen::AngleAxisf(-est_yaw, Eigen::Vector3f::UnitZ()));
    Eigen::Quaternionf q_rp = q_yaw_inv * q_est; // roll+pitch only

    const float yaw_offsets[] = {0.f, M_PI / 3.f, 2.f * M_PI / 3.f, M_PI, -2.f * M_PI / 3.f, -M_PI / 3.f};

    int n_stage2_candidates = std::min(5, static_cast<int>(top_candidates.size()));
    float best_stage2_fitness = std::numeric_limits<float>::max();

    for (int ci = 0; ci < n_stage2_candidates; ci++)
    {
      int kf_idx = top_candidates[ci].kf_idx;
      Eigen::Vector3f cand_pos = sc_snap[kf_idx].position;
      float sc_yaw = static_cast<float>(top_candidates[ci].sc_shift) * 2.f * M_PI / static_cast<float>(dlio::sc::SC_NS);

      for (float yaw_offset : yaw_offsets)
      {
        float yaw = sc_yaw + yaw_offset;
        Eigen::Quaternionf q_full = Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ())) * q_rp;

        Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
        init_guess.block<3, 3>(0, 0) = q_full.toRotationMatrix();
        init_guess.block<3, 1>(0, 3) = cand_pos;
        init_guess = init_guess * T_body_lidar; // body→lidar for lidar-frame scan

        Eigen::Matrix4f result_T;
        float result_fit = 0.f;
        bool ok = tryGICP(cand_pos, init_guess, result_T, result_fit);

        if (ok && result_fit < best_stage2_fitness)
        {
          best_stage2_fitness = result_fit;
          T_map_lidar_gicp = result_T;
          fitness = result_fit;
          gicp_ok = true;
          is_reloc = true;

          if (this->debug_)
            RCLCPP_INFO(this->get_logger(),
                        "[bayes] Stage2 HIT: kf=%d fitness=%.4f sc_yaw=%.0fdeg at [%.1f,%.1f,%.1f]",
                        kf_idx, result_fit, yaw * 180.f / M_PI,
                        cand_pos[0], cand_pos[1], cand_pos[2]);

          // Early exit on excellent fitness
          if (result_fit < 0.05f)
            goto stage2_done;
        }
      }
    }

  stage2_done:
    if (gicp_ok)
    {
      if (this->debug_)
        RCLCPP_INFO(this->get_logger(), "[bayes] Stage2 SC++ RELOC OK: fitness=%.4f", fitness);
    }
    else
    {
      if (this->debug_)
        RCLCPP_INFO(this->get_logger(), "[bayes] Stage2 SC++ failed — no match (%d candidates × 6 yaw)",
                    n_stage2_candidates);
    }
  }

  // ── Stage 3: Consistency verification (re-run GICP from Stage 2 result) ──
  //   Catches GICP false positives (local minima that pass fitness but are wrong).
  //   Re-runs GICP using the Stage 2 result as init guess → result should be
  //   consistent (within 2m and fitness ≤ 1.5× original).
  if (gicp_ok && is_reloc)
  {
    Eigen::Vector3f stage2_pos = T_map_lidar_gicp.block<3, 1>(0, 3);
    Eigen::Matrix4f verify_T;
    float verify_fitness = 0.f;

    bool verify_ok = tryGICP(stage2_pos, T_map_lidar_gicp, verify_T, verify_fitness);

    if (verify_ok)
    {
      float verify_disp = (verify_T.block<3, 1>(0, 3) - T_map_lidar_gicp.block<3, 1>(0, 3)).norm();
      bool consistent = (verify_disp < 2.0f) && (verify_fitness <= fitness * 1.5f + 0.01f);

      if (consistent)
      {
        if (this->debug_)
          RCLCPP_INFO(this->get_logger(),
                      "[bayes] Stage3 verification PASS: disp=%.2fm verify_fit=%.4f orig_fit=%.4f",
                      verify_disp, verify_fitness, fitness);
        // Use the verification result (refined from Stage 2)
        T_map_lidar_gicp = verify_T;
        fitness = verify_fitness;
      }
      else
      {
        if (this->debug_)
          RCLCPP_WARN(this->get_logger(),
                      "[bayes] Stage3 verification INCONSISTENT: disp=%.2fm verify_fit=%.4f orig_fit=%.4f — rejecting",
                      verify_disp, verify_fitness, fitness);
        gicp_ok = false;
      }
    }
    else
    {
      if (this->debug_)
        RCLCPP_WARN(this->get_logger(),
                    "[bayes] Stage3 verification FAIL: re-GICP failed — rejecting Stage2 result");
      gicp_ok = false;
    }
  }

  if (!gicp_ok)
  {
    bayes_consec = 0;

    // Write back Bayesian state
    {
      std::lock_guard<std::mutex> wb(this->continuous_localize_mtx_);
      this->bayes_posterior_ = bayes_snap;
      this->bayes_consecutive_accepts_ = bayes_consec;
    }

    // Publish low confidence: P_loop is high but GICP can't match
    this->last_confidence_ = P_loop * 0.3f;
    if (this->confidence_pub_)
    {
      std_msgs::msg::Float32 msg;
      msg.data = this->last_confidence_;
      this->confidence_pub_->publish(msg);
    }
    return;
  }

  // ── Compute correction ──────────────────────────────────────────────
  // GICP result is T_map←lidar, recover T_map←body by removing lidar extrinsic
  Eigen::Matrix4f T_map_body_gicp = T_map_lidar_gicp * T_body_lidar.inverse();
  Eigen::Matrix4f T_map_odom_new = T_map_body_gicp * T_odom_body.inverse();
  Eigen::Matrix4f delta = T_map_odom_new * T_map_odom_cur.inverse();
  float correction_dist = delta.block<3, 1>(0, 3).norm();
  Eigen::Quaternionf q_corr(delta.block<3, 3>(0, 0));
  float correction_angle = 2.f * std::acos(std::min(std::abs(q_corr.w()), 1.f)) * 180.f / M_PI;

  // Confidence score: how trustworthy is this GICP correction?
  //   fitness_conf = 1 - fitness/threshold  (1.0 = perfect match, 0.0 = at threshold)
  //   confidence = fitness_conf × P_loop    (combined GICP + Bayesian certainty)
  float fitness_conf = std::max(0.0f, 1.0f - fitness / static_cast<float>(this->continuous_localize_fitness_thresh_));
  float confidence = fitness_conf * P_loop;

  // Publish confidence every tick GICP succeeds
  this->last_confidence_ = confidence;
  if (this->confidence_pub_)
  {
    std_msgs::msg::Float32 msg;
    msg.data = confidence;
    this->confidence_pub_->publish(msg);
  }

  // For normal drift correction, limit max correction.
  // For relocalization, no distance/angle limit — the 3-consecutive requirement provides safety.
  int required_consecutive = is_reloc ? std::max(this->bayes_min_consecutive_, 3) : this->bayes_min_consecutive_;

  if (!is_reloc &&
      (correction_dist > this->continuous_localize_max_correction_ ||
       correction_angle > this->continuous_localize_max_correction_ * 5.0f))
  {
    bayes_consec = 0;

    // Write back Bayesian state
    {
      std::lock_guard<std::mutex> wb(this->continuous_localize_mtx_);
      this->bayes_posterior_ = bayes_snap;
      this->bayes_consecutive_accepts_ = bayes_consec;
    }

    if (this->debug_)
      RCLCPP_WARN(this->get_logger(),
                  "[bayes] correction too large (dist=%.3fm, angle=%.1f deg), confidence=%.2f, rejected",
                  correction_dist, correction_angle, confidence);
    return;
  }

  if (this->debug_)
    RCLCPP_INFO(this->get_logger(),
                "[bayes] GICP PASS%s: kf=%d fitness=%.4f conf=%.2f dist=%.3fm angle=%.1fdeg",
                is_reloc ? " (RELOC)" : "", best_kf_idx, fitness, confidence,
                correction_dist, correction_angle);

  // ── Acceptance logic ────────────────────────────────────────────────
  //
  // SC++ Stage 2 relocalization: consecutive acceptance (Stage 3 already
  //   verified consistency, so use consecutive count for safety).
  //
  // Stage 1 drift correction: g2o verification (if enabled) or consecutive.

  bool accepted = false;

  if (is_reloc)
  {
    // SC++ Stage 2 relocalization: Stage 3 verified, use consecutive for safety
    bayes_consec++;
    accepted = (bayes_consec >= required_consecutive);
    if (this->debug_)
      RCLCPP_INFO(this->get_logger(),
                  "[bayes] RELOC consecutive=%d/%d", bayes_consec, required_consecutive);
  }
  else if (this->g2o_verification_enabled_)
  {
    double chi2 = 0.0;
    int num_anchors = 0;
    bool g2o_ok = this->verifyLoopWithG2O(
        best_kf_idx, T_map_body_gicp, T_odom_body,
        T_map_odom_cur, sc_snap, chi2, num_anchors);

    if (num_anchors == 0)
    {
      bayes_consec++;
      accepted = (bayes_consec >= required_consecutive);
      if (this->debug_)
        RCLCPP_INFO(this->get_logger(),
                    "[bayes] g2o: no anchors (init?), fallback consecutive=%d/%d",
                    bayes_consec, required_consecutive);
    }
    else if (g2o_ok)
    {
      accepted = true;
      if (this->debug_)
        RCLCPP_INFO(this->get_logger(),
                    "[bayes] g2o ACCEPTED (chi2=%.4f < %.1f, anchors=%d)",
                    chi2, this->g2o_chi2_threshold_, num_anchors);
    }
    else
    {
      if (this->debug_)
        RCLCPP_INFO(this->get_logger(),
                    "[bayes] g2o REJECTED (chi2=%.4f > %.1f, anchors=%d)",
                    chi2, this->g2o_chi2_threshold_, num_anchors);
    }
  }
  else
  {
    // g2o disabled: consecutive acceptance
    bayes_consec++;
    accepted = (bayes_consec >= required_consecutive);

    if (this->debug_)
      RCLCPP_INFO(this->get_logger(),
                  "[bayes] consecutive=%d/%d", bayes_consec, required_consecutive);
  }

  if (accepted)
  {
    bayes_consec = 0;
    if (is_reloc)
      bayes_snap.clear();
  }

  // Write back all Bayesian state + correction under brief lock
  {
    std::lock_guard<std::mutex> wb(this->continuous_localize_mtx_);
    this->bayes_posterior_ = bayes_snap;
    this->bayes_consecutive_accepts_ = bayes_consec;
    if (accepted)
      this->T_map_odom_ = T_map_odom_new;
  }

  if (accepted)
  {
    if (this->debug_)
      RCLCPP_INFO(this->get_logger(),
                  "[bayes] >>> ACCEPTED %s correction (dist=%.3fm, angle=%.1f deg, fitness=%.4f, confidence=%.2f, P_loop=%.3f)",
                  is_reloc ? "RELOCALIZATION" : "map->odom",
                  correction_dist, correction_angle, fitness, confidence, P_loop);
  }
}

bool dlio::OdomNode::verifyLoopWithG2O(
    int loop_kf_idx,
    const Eigen::Matrix4f &T_map_body_gicp,
    const Eigen::Matrix4f &T_odom_body,
    const Eigen::Matrix4f &T_map_odom_current,
    const std::vector<dlio::sc::ScanContextEntry> &sc_snap,
    double &out_chi2, int &out_num_anchors)
{
  // ── Build g2o pose graph ──────────────────────────────────────────────
  //
  // Key idea: vertices use CURRENT T_map_odom (before correction).
  // The loop edge proposes a DIFFERENT position for current pose (from GICP).
  // If the GICP correction is wrong, the loop edge contradicts the anchored
  // trajectory → high chi2. If correct, consistent → low chi2.
  //
  // Vertices (in map frame via CURRENT T_map_odom):
  //   - Recent session keyframes
  //   - Current pose (at current estimated position, NOT GICP result)
  //   - Prior map anchor keyframes (fixed, known map positions)
  //
  // Edges:
  //   - Odometry: consecutive session kfs (tight, from odom)
  //   - Anchors:  session kfs ↔ nearby SC database kfs (moderate, fixed)
  //   - Loop:     SC kf → current pose (measurement from GICP — the tension source)

  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);

  auto linear_solver = std::make_unique<
      g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
  auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
  auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
  optimizer.setAlgorithm(solver);

  // ── Snapshot session keyframes (odom frame) ──
  struct KfSnap
  {
    Eigen::Vector3f p;
    Eigen::Quaternionf q;
  };
  std::vector<KfSnap> session_kf;
  {
    std::lock_guard<std::mutex> lock(this->keyframes_mutex);
    int n = static_cast<int>(this->keyframes.size());
    int start = std::max(0, n - 50);
    for (int i = start; i < n; i++)
      session_kf.push_back({this->keyframes[i].first.first,
                            this->keyframes[i].first.second});
  }

  int num_sk = static_cast<int>(session_kf.size());
  if (num_sk < 3)
  {
    out_chi2 = 0.0;
    out_num_anchors = 0;
    return true; // too few keyframes, pass through
  }

  // ── Project session keyframes to map frame using CURRENT T_map_odom ──
  Eigen::Matrix4d T_mo = T_map_odom_current.cast<double>();
  std::vector<Eigen::Isometry3d> sk_map(num_sk);
  for (int i = 0; i < num_sk; i++)
  {
    Eigen::Isometry3d T_ob = Eigen::Isometry3d::Identity();
    T_ob.translation() = session_kf[i].p.cast<double>();
    T_ob.linear() = session_kf[i].q.cast<double>().toRotationMatrix();
    Eigen::Isometry3d T_mb;
    T_mb.matrix() = T_mo * T_ob.matrix();
    sk_map[i] = T_mb;
  }

  // Current pose in map frame using CURRENT T_map_odom (estimated, not GICP)
  Eigen::Isometry3d current_est;
  {
    Eigen::Isometry3d T_ob = Eigen::Isometry3d::Identity();
    T_ob.matrix() = T_odom_body.cast<double>();
    current_est.matrix() = T_mo * T_ob.matrix();
  }

  // GICP result in map frame (what the loop edge proposes)
  Eigen::Isometry3d current_gicp;
  current_gicp.matrix() = T_map_body_gicp.cast<double>();

  // ── Add session keyframe vertices ──
  for (int i = 0; i < num_sk; i++)
  {
    auto *v = new g2o::VertexSE3();
    v->setId(i);
    v->setEstimate(sk_map[i]);
    if (i == 0)
      v->setFixed(true);
    optimizer.addVertex(v);
  }

  // Current pose vertex — initialized at CURRENT estimated position
  {
    auto *v = new g2o::VertexSE3();
    v->setId(num_sk);
    v->setEstimate(current_est);
    optimizer.addVertex(v);
  }

  // ── Odometry edges ──
  // Relative transforms from odom (independent of T_map_odom)
  Eigen::Matrix<double, 6, 6> odom_info =
      Eigen::Matrix<double, 6, 6>::Identity() * 100.0;

  for (int i = 1; i < num_sk; i++)
  {
    // Odom-frame relative (T_map_odom cancels out)
    Eigen::Isometry3d T_ob_prev = Eigen::Isometry3d::Identity();
    T_ob_prev.translation() = session_kf[i - 1].p.cast<double>();
    T_ob_prev.linear() = session_kf[i - 1].q.cast<double>().toRotationMatrix();
    Eigen::Isometry3d T_ob_cur = Eigen::Isometry3d::Identity();
    T_ob_cur.translation() = session_kf[i].p.cast<double>();
    T_ob_cur.linear() = session_kf[i].q.cast<double>().toRotationMatrix();

    Eigen::Isometry3d rel = T_ob_prev.inverse() * T_ob_cur;
    auto *e = new g2o::EdgeSE3();
    e->setMeasurement(rel);
    e->setInformation(odom_info);
    e->vertices()[0] = optimizer.vertex(i - 1);
    e->vertices()[1] = optimizer.vertex(i);
    optimizer.addEdge(e);
  }

  // Last session kf → current pose (odom-frame relative)
  {
    Eigen::Isometry3d T_ob_last = Eigen::Isometry3d::Identity();
    T_ob_last.translation() = session_kf.back().p.cast<double>();
    T_ob_last.linear() = session_kf.back().q.cast<double>().toRotationMatrix();
    Eigen::Isometry3d T_ob_current = Eigen::Isometry3d::Identity();
    T_ob_current.matrix() = T_odom_body.cast<double>();

    Eigen::Isometry3d rel = T_ob_last.inverse() * T_ob_current;
    auto *e = new g2o::EdgeSE3();
    e->setMeasurement(rel);
    e->setInformation(odom_info);
    e->vertices()[0] = optimizer.vertex(num_sk - 1);
    e->vertices()[1] = optimizer.vertex(num_sk);
    optimizer.addEdge(e);
  }

  // ── Prior map anchor edges ──
  // Ground the trajectory to known map positions.
  // Measurements computed from CURRENT projection (before correction).
  int next_id = num_sk + 1;
  int num_anchors = 0;
  int N_sc = static_cast<int>(sc_snap.size());

  Eigen::Matrix<double, 6, 6> anchor_info = Eigen::Matrix<double, 6, 6>::Identity();
  anchor_info.block<3, 3>(0, 0) *= 5.0;  // rotation: loose
  anchor_info.block<3, 3>(3, 3) *= 50.0; // translation: moderate

  for (int i = 0; i < num_sk; i += 3)
  {
    Eigen::Vector3d sk_pos = sk_map[i].translation();

    double best_dist = 1e9;
    int best_sc = -1;
    for (int j = 0; j < N_sc; j++)
    {
      double d = (sc_snap[j].position.cast<double>() - sk_pos).norm();
      if (d < best_dist)
      {
        best_dist = d;
        best_sc = j;
      }
    }

    if (best_sc >= 0 && best_dist < 10.0)
    {
      auto *v = new g2o::VertexSE3();
      v->setId(next_id);
      Eigen::Isometry3d sc_pose = Eigen::Isometry3d::Identity();
      sc_pose.translation() = sc_snap[best_sc].position.cast<double>();
      v->setEstimate(sc_pose);
      v->setFixed(true);
      optimizer.addVertex(v);

      // Measurement: expected relative transform (from current projection)
      Eigen::Isometry3d rel = sc_pose.inverse() * sk_map[i];
      auto *e = new g2o::EdgeSE3();
      e->setMeasurement(rel);
      e->setInformation(anchor_info);
      e->vertices()[0] = optimizer.vertex(next_id);
      e->vertices()[1] = optimizer.vertex(i);
      optimizer.addEdge(e);

      next_id++;
      num_anchors++;
    }
  }

  // ── Loop closure edge ──
  // This is the ONLY edge that uses the GICP result.
  // It says: "current pose should be at T_map_body_gicp relative to SC kf"
  // If GICP matched at wrong place, this fights the odom chain + anchors → high chi2.
  Eigen::Isometry3d sc_kf_pose = Eigen::Isometry3d::Identity();
  sc_kf_pose.translation() = sc_snap[loop_kf_idx].position.cast<double>();

  auto *v_loop = new g2o::VertexSE3();
  v_loop->setId(next_id);
  v_loop->setEstimate(sc_kf_pose);
  v_loop->setFixed(true);
  optimizer.addVertex(v_loop);

  // Loop measurement: from SC kf to GICP result (proposed position)
  Eigen::Isometry3d loop_meas = sc_kf_pose.inverse() * current_gicp;

  Eigen::Matrix<double, 6, 6> loop_info = Eigen::Matrix<double, 6, 6>::Identity();
  loop_info.block<3, 3>(0, 0) *= 10.0;  // rotation
  loop_info.block<3, 3>(3, 3) *= 100.0; // translation

  auto *loop_edge = new g2o::EdgeSE3();
  loop_edge->setMeasurement(loop_meas);
  loop_edge->setInformation(loop_info);
  loop_edge->vertices()[0] = optimizer.vertex(next_id);
  loop_edge->vertices()[1] = optimizer.vertex(num_sk);
  optimizer.addEdge(loop_edge);

  // ── Optimize ──
  optimizer.initializeOptimization();
  optimizer.optimize(this->g2o_iterations_);

  // ── Check chi2 ──
  out_chi2 = loop_edge->chi2();

  if (this->debug_)
    RCLCPP_INFO(this->get_logger(),
                "[bayes] g2o: loop_chi2=%.4f anchors=%d vertices=%d edges=%d",
                out_chi2, num_anchors,
                static_cast<int>(optimizer.vertices().size()),
                static_cast<int>(optimizer.edges().size()));

  out_num_anchors = num_anchors;
  return out_chi2 < this->g2o_chi2_threshold_;
}

void dlio::OdomNode::publishOccupancyGrid()
{
  if (!this->occupancy_grid_gen_ || !this->dlio_initialized)
    return;

  pcl::PointCloud<PointType>::ConstPtr scan;
  Eigen::Matrix4f T_current;
  double scan_time;
  {
    std::lock_guard<std::mutex> lock(this->latest_scan_mtx_);
    if (!this->latest_scan_ || this->latest_scan_->empty())
      return;
    scan_time = this->latest_scan_time_;
    // Skip if we already processed this scan
    if (scan_time <= this->og_last_scan_time_)
      return;
    scan = this->latest_scan_;
    T_current = this->latest_scan_T_;
  }
  this->og_last_scan_time_ = scan_time;

  // Transform sensor-frame scan to map frame (via odom → map correction)
  Eigen::Matrix4f T_lidar_odom = T_current * this->extrinsics.baselink2lidar_T;
  Eigen::Matrix4f T_lidar_map = this->T_map_odom_ * T_lidar_odom;
  pcl::PointCloud<PointType>::Ptr scan_world(new pcl::PointCloud<PointType>);
  pcl::transformPointCloud(*scan, *scan_world, T_lidar_map);

  // Sensor origin in map frame
  Eigen::Vector3f sensor_origin = T_lidar_map.block<3, 1>(0, 3);

  // Use the scan timestamp (matches TF) for correct RViz rendering
  rclcpp::Time scan_stamp(static_cast<int64_t>(scan_time * 1e9),
                          this->get_clock()->get_clock_type());

  // Extract robot yaw and pitch from map-frame transform
  Eigen::Matrix3f R_map = T_lidar_map.block<3, 3>(0, 0);
  float robot_yaw = std::atan2(R_map(1, 0), R_map(0, 0));
  float robot_pitch = std::asin(-R_map(2, 0));

  this->occupancy_grid_gen_->update(scan_world, sensor_origin, scan_time,
                                    robot_yaw, robot_pitch);
  auto og_msg = this->occupancy_grid_gen_->getOccupancyGrid(scan_stamp);

  // Debug: count non-unknown cells
  int free_cells = 0, occ_cells = 0;
  for (auto v : og_msg.data)
  {
    if (v >= 0 && v < 50)
      free_cells++;
    else if (v >= 50)
      occ_cells++;
  }
  if (this->debug_)
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "[ogm] pts=%zu origin=(%.1f,%.1f) sensor=(%.1f,%.1f,%.1f) free=%d occ=%d frame=%s",
                         scan_world->size(), og_msg.info.origin.position.x, og_msg.info.origin.position.y,
                         sensor_origin.x(), sensor_origin.y(), sensor_origin.z(),
                         free_cells, occ_cells, og_msg.header.frame_id.c_str());

  this->occupancy_grid_pub_->publish(std::move(og_msg));
}
