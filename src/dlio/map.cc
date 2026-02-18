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

#include "dlio/map.h"
#include "dlio/utils.h"

#include <csignal>
#include <fstream>
#include <filesystem>

// Global pointer for signal handler
static dlio::MapNode* g_map_node = nullptr;

static void shutdownSave(int sig) {
  if (g_map_node) {
    g_map_node->saveOnShutdown();
    g_map_node = nullptr;
  }
  // Re-raise to let default handler finish
  signal(sig, SIG_DFL);
  raise(sig);
}

dlio::MapNode::MapNode(const rclcpp::NodeOptions& options)
: Node("dlio_map_node", options) {

  this->getParams();

  this->keyframe_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto keyframe_sub_opt = rclcpp::SubscriptionOptions();
  keyframe_sub_opt.callback_group = this->keyframe_cb_group;
  this->keyframe_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("keyframes", 10,
      std::bind(&dlio::MapNode::callbackKeyframe, this, std::placeholders::_1), keyframe_sub_opt);

  this->map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("map", 100);

  this->save_pcd_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->save_pcd_srv = this->create_service<direct_lidar_inertial_odometry::srv::SavePCD>("save_pcd",
      std::bind(&dlio::MapNode::savePCD, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_services_default, this->save_pcd_cb_group);

  this->dlio_map = std::make_shared<pcl::PointCloud<PointType>>();

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  // Ensure parent directory exists for map save
  if (!this->map_path_.empty()) {
    std::filesystem::path filepath(this->map_path_);
    if (filepath.has_parent_path()) {
      std::filesystem::create_directories(filepath.parent_path());
    }
  }

  // Load prior map if file exists
  if (!this->map_path_.empty()) {
    std::ifstream f(this->map_path_);
    if (f.good()) {
      f.close();
      this->loadPriorMap();
    } else {
      RCLCPP_INFO(this->get_logger(), "[map] No existing map at '%s', starting fresh",
                  this->map_path_.c_str());
    }
  }

  // Auto-save timer (mapping mode only)
  if (this->map_mode_ == "mapping" && !this->map_path_.empty() && this->auto_save_interval_ > 0.) {
    this->auto_save_timer = this->create_wall_timer(
        std::chrono::duration<double>(this->auto_save_interval_),
        std::bind(&dlio::MapNode::autoSave, this));

    RCLCPP_INFO(this->get_logger(), "[map] Auto-save: every %.0fs to %s",
                this->auto_save_interval_, this->map_path_.c_str());
  }

  // Register signal handler for save-on-shutdown (mapping mode)
  if (this->map_mode_ == "mapping" && !this->map_path_.empty()) {
    g_map_node = this;
    signal(SIGINT, shutdownSave);
    signal(SIGTERM, shutdownSave);
  }

  RCLCPP_INFO(this->get_logger(), "[map] Initialized (mode=%s, leaf_size=%.2f, map=%zu pts)",
              this->map_mode_.c_str(), this->leaf_size_, this->dlio_map->points.size());

}

dlio::MapNode::~MapNode() {
  // Fallback: save on destructor if signal handler didn't fire
  this->saveOnShutdown();
}

void dlio::MapNode::getParams() {

  this->declare_parameter<std::string>("odom/odom_frame", "odom");
  this->declare_parameter<double>("map/sparse/leafSize", 0.5);
  this->declare_parameter<std::string>("map/mode", "localization");
  this->declare_parameter<std::string>("map/path", "");
  this->declare_parameter<bool>("map/use_corrected", true);
  this->declare_parameter<double>("map/voxel_size", 0.25);
  this->declare_parameter<double>("map/auto_save_interval", 30.0);

  this->get_parameter("odom/odom_frame", this->odom_frame);
  this->get_parameter("map/sparse/leafSize", this->leaf_size_);
  this->get_parameter("map/mode", this->map_mode_);
  this->get_parameter("map/path", this->map_path_);
  this->get_parameter("map/use_corrected", this->use_corrected_);
  if (this->map_path_.empty())
  {
    const char *home = std::getenv("HOME");
    if (home)
    {
      this->map_path_ = std::string(home) + "/.ros/dlio_map.pcd";
      RCLCPP_INFO(this->get_logger(), "[map] map/path not set, defaulting to: %s", this->map_path_.c_str());
    }
  }
  this->get_parameter("map/voxel_size", this->map_voxel_size_);
  this->get_parameter("map/auto_save_interval", this->auto_save_interval_);
}

void dlio::MapNode::loadPriorMap()
{
  // Resolve load path: try corrected file first if use_corrected is enabled
  std::string load_path = this->map_path_;
  if (this->use_corrected_)
  {
    std::filesystem::path p(this->map_path_);
    std::string corrected = p.parent_path().string() + "/" + p.stem().string() + "_corrected.pcd";
    if (std::filesystem::exists(corrected))
    {
      load_path = corrected;
      RCLCPP_INFO(this->get_logger(), "[map] Using corrected map: %s", corrected.c_str());
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "[map] Corrected map not found (%s), falling back to raw", corrected.c_str());
    }
  }

  pcl::PointCloud<PointType>::Ptr prior_cloud = std::make_shared<pcl::PointCloud<PointType>>();
  if (pcl::io::loadPCDFile(load_path, *prior_cloud) == -1) {
    RCLCPP_ERROR(this->get_logger(), "[map] Failed to load prior map: %s", load_path.c_str());
    return;
  }

  // Voxel filter
  this->voxelgrid.setLeafSize(this->leaf_size_, this->leaf_size_, this->leaf_size_);
  this->voxelgrid.setInputCloud(prior_cloud);
  this->voxelgrid.filter(*prior_cloud);

  *this->dlio_map += *prior_cloud;

  RCLCPP_INFO(this->get_logger(), "[map] Loaded prior map: %zu points from %s",
              prior_cloud->points.size(), load_path.c_str());

  // Publish loaded map immediately for RViz visualization
  sensor_msgs::msg::PointCloud2 map_ros;
  pcl::toROSMsg(*this->dlio_map, map_ros);
  map_ros.header.stamp = this->now();
  map_ros.header.frame_id = this->odom_frame;
  this->map_pub->publish(map_ros);
  RCLCPP_INFO(this->get_logger(), "[map] Published prior map (%zu pts) for visualization",
              this->dlio_map->points.size());
}

void dlio::MapNode::autoSave()
{
  if (!this->dlio_map || this->dlio_map->points.empty()) return;

  pcl::PointCloud<PointType>::Ptr m = std::make_shared<pcl::PointCloud<PointType>>(*this->dlio_map);

  // Voxel filter before save
  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(this->map_voxel_size_, this->map_voxel_size_, this->map_voxel_size_);
  vg.setInputCloud(m);
  vg.filter(*m);

  int ret = pcl::io::savePCDFileBinary(this->map_path_, *m);
  if (ret == 0) {
    RCLCPP_INFO(this->get_logger(), "[map] Saved: %zu pts to %s",
                m->points.size(), this->map_path_.c_str());
  } else {
    RCLCPP_ERROR(this->get_logger(), "[map] Save failed: %s", this->map_path_.c_str());
  }
}

void dlio::MapNode::saveOnShutdown()
{
  if (this->map_mode_ != "mapping" || this->map_path_.empty()) return;
  if (!this->dlio_map || this->dlio_map->points.empty()) return;

  // Ensure parent directory exists
  std::filesystem::path filepath(this->map_path_);
  if (filepath.has_parent_path()) {
    std::filesystem::create_directories(filepath.parent_path());
  }

  pcl::PointCloud<PointType>::Ptr m = std::make_shared<pcl::PointCloud<PointType>>(*this->dlio_map);

  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(this->map_voxel_size_, this->map_voxel_size_, this->map_voxel_size_);
  vg.setInputCloud(m);
  vg.filter(*m);

  int ret = pcl::io::savePCDFileBinary(this->map_path_, *m);
  if (ret == 0) {
    std::cout << "[map] Shutdown save: " << m->points.size() << " pts to " << this->map_path_ << std::endl;
  } else {
    std::cerr << "[map] Shutdown save FAILED: " << this->map_path_ << std::endl;
  }
}

void dlio::MapNode::start() {
}

void dlio::MapNode::callbackKeyframe(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keyframe) {

  // convert scan to pcl format
  pcl::PointCloud<PointType>::Ptr keyframe_pcl = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*keyframe, *keyframe_pcl);

  // voxel filter
  this->voxelgrid.setLeafSize(this->leaf_size_, this->leaf_size_, this->leaf_size_);
  this->voxelgrid.setInputCloud(keyframe_pcl);
  this->voxelgrid.filter(*keyframe_pcl);

  // save filtered keyframe to map for rviz
  *this->dlio_map += *keyframe_pcl;

  // publish full map
  if (this->dlio_map->points.size() == this->dlio_map->width * this->dlio_map->height) {
    sensor_msgs::msg::PointCloud2 map_ros;
    pcl::toROSMsg(*this->dlio_map, map_ros);
    map_ros.header.stamp = this->now();
    map_ros.header.frame_id = this->odom_frame;
    this->map_pub->publish(map_ros);
  }

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
    "[map] Received keyframe: %zu pts, total map: %zu pts",
    keyframe_pcl->points.size(), this->dlio_map->points.size());
}

void dlio::MapNode::savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
                            std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res) {

  pcl::PointCloud<PointType>::Ptr m = std::make_shared<pcl::PointCloud<PointType>>(*this->dlio_map);

  if (!m || m->points.empty()) {
    std::cout << "[map] SavePCD: map is empty, nothing to save" << std::endl;
    res->success = false;
    return;
  }

  float leaf_size = req->leaf_size;
  std::string p = req->save_path;

  // Derive filename from map_path_ stem (e.g. "test_gs.pcd" -> "test_gs.pcd")
  std::filesystem::path map_fp(this->map_path_);
  std::string filename = map_fp.filename().string();
  if (filename.empty()) filename = "dlio_map.pcd";
  std::string save_file = p + "/" + filename;

  std::cout << std::setprecision(2) << "Saving map to " << save_file
    << " with leaf size " << to_string_with_precision(leaf_size, 2) << "... "; std::cout.flush();

  // voxelize map
  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.setInputCloud(m);
  vg.filter(*m);

  // save map
  int ret = pcl::io::savePCDFileBinary(save_file, *m);
  res->success = ret == 0;

  if (res->success) {
    std::cout << "done" << std::endl;
  } else {
    std::cout << "failed" << std::endl;
  }
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(dlio::MapNode)
