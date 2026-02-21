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
#include "direct_lidar_inertial_odometry/srv/save_pcd.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>

// PCL
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <mutex>
#include <atomic>

class dlio::MapNode : public rclcpp::Node
{

public:
  explicit MapNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~MapNode();

  void start();
  void saveOnShutdown();

private:
  void getParams();

  void callbackKeyframe(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &keyframe);

  void savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
               std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res);

  void loadPriorMap();
  void autoSave();

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr keyframe_sub;
  rclcpp::CallbackGroup::SharedPtr keyframe_cb_group, save_pcd_cb_group;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub;

  rclcpp::Service<direct_lidar_inertial_odometry::srv::SavePCD>::SharedPtr save_pcd_srv;
  rclcpp::TimerBase::SharedPtr auto_save_timer;

  pcl::PointCloud<PointType>::Ptr dlio_map;
  std::mutex map_mtx_;                      // protects dlio_map from concurrent access
  std::atomic<bool> shutdown_saved_{false}; // prevent double saveOnShutdown
  pcl::VoxelGrid<PointType> voxelgrid;

  std::string odom_frame;
  std::string map_frame_;
  std::string map_mode_;
  std::string map_path_;
  bool use_corrected_;
  double map_voxel_size_;
  double auto_save_interval_;
  double publish_interval_;

  double leaf_size_;
};
