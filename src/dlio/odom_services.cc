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

void dlio::OdomNode::continuousLocalize()
{
  // Prerequisites
  if (!this->dlio_initialized || !this->relocalized_)
  {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "[bayes] skip: initialized=%d, relocalized=%d",
                         this->dlio_initialized.load(), this->relocalized_);
    return;
  }

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
    {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "[bayes] skip: prior_map=%d, kdtree=%d",
                           this->prior_map_cloud_ ? 1 : 0, this->prior_map_kdtree_ ? 1 : 0);
      return;
    }
    if (this->sc_database_.empty())
    {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "[bayes] skip: sc_database empty");
      return;
    }
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
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "[bayes] skip: no latest scan");
    return;
  }

  // Skip if scan is stale (no new data arriving)
  double now_sec = this->now().seconds();
  double scan_age = now_sec - scan_time;
  if (scan_time > 0.0 && scan_age > 3.0 * this->continuous_localize_interval_)
  {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "[bayes] skip: scan stale (%.1fs old)", scan_age);
    return;
  }

  const int N = static_cast<int>(sc_snap.size());

  // Initialize Bayesian posterior if needed (lazy init / size change)
  if (static_cast<int>(bayes_snap.size()) != N + 1)
  {
    bayes_snap.resize(N + 1);
    bayes_snap[0] = 1.0f; // virtual place = certain "new place"
    for (int i = 1; i <= N; i++)
      bayes_snap[i] = 0.0f;
    bayes_consec = 0;
    RCLCPP_INFO(this->get_logger(), "[bayes] initialized posterior: N=%d keyframes", N);
  }

  // ── 2. Compute SC likelihoods ──────────────────────────────────────────

  // Rotate scan to gravity-aligned frame for SC descriptor
  auto sc_scan = dlio::sc::prepareGravityAlignedScan(
      scan_body, this->kfdb_gravity_q_, this->extrinsics.baselink2lidar.R);

  auto sc_desc = dlio::sc::computeScanContext(sc_scan, this->sc_max_range_);

  // Raw likelihoods: invert SC distance (lower distance = better match = higher likelihood)
  // Two filters applied:
  //   1. SC distance threshold: reject candidates with dist > threshold (too dissimilar)
  //   2. Top-K: only keep the K best matches, zero out the rest
  std::vector<float> raw_likelihood(N);
  std::vector<float> sc_distances(N);
  float best_raw = 0.0f;
  int best_raw_idx = 0;
  float worst_raw = 1.0f;
  int n_dist_rejected = 0;
  for (int i = 0; i < N; i++)
  {
    auto [dist, shift] = dlio::sc::computeScanContextDistance(sc_desc, sc_snap[i].descriptor);
    sc_distances[i] = dist;

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
  };
  std::vector<BayesCandidate> top_candidates;
  top_candidates.reserve(N);
  for (int i = 1; i <= N; i++)
  {
    if (bayes_snap[i] > 0.0f)
      top_candidates.push_back({i - 1, bayes_snap[i]});
  }
  std::sort(top_candidates.begin(), top_candidates.end(),
            [](const BayesCandidate &a, const BayesCandidate &b)
            { return a.posterior > b.posterior; });

  int best_kf_idx = top_candidates.empty() ? 0 : top_candidates[0].kf_idx;
  float best_kf_posterior = top_candidates.empty() ? 0.0f : top_candidates[0].posterior;

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

  if (P_loop <= this->bayes_loop_threshold_)
  {
    // Virtual place wins — no correction, reset consecutive
    bayes_consec = 0;

    // Write back Bayesian state
    {
      std::lock_guard<std::mutex> wb(this->continuous_localize_mtx_);
      this->bayes_posterior_ = bayes_snap;
      this->bayes_consecutive_accepts_ = bayes_consec;
    }

    // Publish low confidence: P_loop scaled down (not in a recognized place)
    this->last_confidence_ = P_loop * 0.5f;
    if (this->confidence_pub_)
    {
      std_msgs::msg::Float32 msg;
      msg.data = this->last_confidence_;
      this->confidence_pub_->publish(msg);
    }
    return;
  }

  // ── GICP verification: Two-stage approach ──────────────────────────────
  //
  // Stage 1 (drift correction): Try GICP at current estimated position.
  //   Produces small corrections that fix accumulated odom drift.
  //
  // Stage 2 (relocalization): If Stage 1 fails AND estimated position is
  //   far from the Bayesian winner keyframe, try GICP at the keyframe's
  //   position. This handles random initialization / large position error.
  //   Relocalization requires more consecutive accepts (3 vs 2).

  Eigen::Matrix4f T_map_odom_cur = T_map_odom_snap;
  Eigen::Matrix4f T_body_lidar = this->extrinsics.baselink2lidar_T; // lidar→body

  Eigen::Matrix4f T_map_body_est = T_map_odom_cur * T_odom_body;
  Eigen::Matrix4f T_map_lidar_est = T_map_body_est * T_body_lidar; // lidar→body→odom→map
  Eigen::Vector3f est_pos = T_map_body_est.block<3, 1>(0, 3);

  // Number of keyframe candidates to try in Stage 2 (relocalization)
  int max_reloc_candidates = std::min(static_cast<int>(top_candidates.size()),
                                      std::max(this->bayes_sc_top_k_, 3));

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
    prior_kdtree_snap->radiusSearch(search_pt, 50.f * 50.f, nn_indices, nn_dists);

    if (nn_indices.size() < 200)
    {
      RCLCPP_WARN(this->get_logger(),
                  "[bayes] tryGICP FAIL: too few neighbors=%zu (need 200) at [%.1f,%.1f,%.1f], prior_map=%zu pts",
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
      RCLCPP_WARN(this->get_logger(),
                  "[bayes] tryGICP FAIL: not converged at [%.1f,%.1f,%.1f], scan=%zu pts, local_map=%zu pts",
                  center[0], center[1], center[2], scan_body->size(), local_map->size());
      return false;
    }

    if (result_fitness > this->continuous_localize_fitness_thresh_)
    {
      RCLCPP_WARN(this->get_logger(),
                  "[bayes] tryGICP FAIL: fitness=%.4f > thresh=%.4f at [%.1f,%.1f,%.1f], scan=%zu, local_map=%zu",
                  result_fitness, this->continuous_localize_fitness_thresh_,
                  center[0], center[1], center[2], scan_body->size(), local_map->size());
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
  gicp_ok = tryGICP(est_pos, T_map_lidar_est, T_map_lidar_gicp, fitness);

  if (gicp_ok)
  {
    RCLCPP_INFO(this->get_logger(), "[bayes] Stage1 GICP OK: fitness=%.4f at est=[%.1f,%.1f,%.1f]",
                fitness, est_pos[0], est_pos[1], est_pos[2]);
  }

  // ── Stage 2: GICP at top keyframe candidates (relocalization fallback) ──
  //   Try multiple candidates by posterior rank until one converges.
  //   Only active when enable_global_correction is true.
  if (!gicp_ok && this->enable_global_correction_.load())
  {
    for (int ci = 0; ci < max_reloc_candidates && !gicp_ok; ci++)
    {
      int kf_idx = top_candidates[ci].kf_idx;
      Eigen::Vector3f kf_pos = sc_snap[kf_idx].position;
      float est_kf_dist = (est_pos - kf_pos).norm();

      if (est_kf_dist <= 1.0f)
        continue; // too close to estimated pos, Stage 1 already tried this area

      // Init guess: body rotation (trust IMU) + keyframe position, then apply lidar extrinsic
      Eigen::Matrix4f kf_init = Eigen::Matrix4f::Identity();
      kf_init.block<3, 3>(0, 0) = T_map_body_est.block<3, 3>(0, 0);
      kf_init.block<3, 1>(0, 3) = kf_pos;
      kf_init = kf_init * T_body_lidar; // body→lidar extrinsic for lidar-frame scan

      gicp_ok = tryGICP(kf_pos, kf_init, T_map_lidar_gicp, fitness);

      if (gicp_ok)
      {
        is_reloc = true;
        best_kf_idx = kf_idx; // update to the candidate that actually worked
        RCLCPP_INFO(this->get_logger(),
                    "[bayes] Stage2 RELOC GICP OK: fitness=%.4f at candidate #%d kf%d=[%.1f,%.1f,%.1f] "
                    "(est was [%.1f,%.1f,%.1f], dist=%.1fm)",
                    fitness, ci, kf_idx, kf_pos[0], kf_pos[1], kf_pos[2],
                    est_pos[0], est_pos[1], est_pos[2], est_kf_dist);
      }
      else
      {
        RCLCPP_INFO(this->get_logger(),
                    "[bayes] Stage2 candidate #%d kf%d GICP failed at [%.1f,%.1f,%.1f]",
                    ci, kf_idx, kf_pos[0], kf_pos[1], kf_pos[2]);
      }
    }

    if (!gicp_ok)
    {
      RCLCPP_INFO(this->get_logger(),
                  "[bayes] GICP failed at est and all %d reloc candidates", max_reloc_candidates);
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

    RCLCPP_WARN(this->get_logger(),
                "[bayes] correction too large (dist=%.3fm, angle=%.1f deg), confidence=%.2f, rejected",
                correction_dist, correction_angle, confidence);
    return;
  }

  // GICP passed — increment consecutive counter
  bayes_consec++;

  RCLCPP_INFO(this->get_logger(),
              "[bayes] GICP PASS%s: kf=%d fitness=%.4f conf=%.2f dist=%.3fm angle=%.1fdeg | consecutive=%d/%d",
              is_reloc ? " (RELOC)" : "", best_kf_idx, fitness, confidence,
              correction_dist, correction_angle,
              bayes_consec, required_consecutive);

  // ── Delayed acceptance: apply only after N consecutive confirms ────────

  bool accepted = (bayes_consec >= required_consecutive);
  if (accepted)
  {
    bayes_consec = 0;

    // After relocalization, reset posterior so Bayesian filter starts fresh
    // from the corrected position (old posteriors are for the wrong location)
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
    RCLCPP_INFO(this->get_logger(),
                "[bayes] >>> ACCEPTED %s correction (dist=%.3fm, angle=%.1f deg, fitness=%.4f, confidence=%.2f, P_loop=%.3f)",
                is_reloc ? "RELOCALIZATION" : "map->odom",
                correction_dist, correction_angle, fitness, confidence, P_loop);
  }
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
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                       "[ogm] pts=%zu origin=(%.1f,%.1f) sensor=(%.1f,%.1f,%.1f) free=%d occ=%d frame=%s",
                       scan_world->size(), og_msg.info.origin.position.x, og_msg.info.origin.position.y,
                       sensor_origin.x(), sensor_origin.y(), sensor_origin.z(),
                       free_cells, occ_cells, og_msg.header.frame_id.c_str());

  this->occupancy_grid_pub_->publish(std::move(og_msg));
}
