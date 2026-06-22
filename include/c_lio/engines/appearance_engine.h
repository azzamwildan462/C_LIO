/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * Yamato Infrastructure Robotics,                         *
 * Japan                                                   *
 *                                                         *
 * Authors: Azzam Wildan M                                 *
 *                                                         *
 ***********************************************************/

#pragma once

#include "c_lio/c_lio.h"
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>
#include <utility>

namespace c_lio
{

  enum class AppearanceMethod
  {
    SC_PLUS_PLUS,      // Scan Context++ (CPU, header-only)
    SC_PLUS_PLUS_CUDA, // SC++ with GPU ring/sector computation (future)
    STD,               // Stable Triangle Descriptor (future)
    STD_CUDA           // STD with GPU extraction + matching (future)
  };

  AppearanceMethod parseAppearanceMethod(const std::string &s);
  std::string appearanceMethodToString(AppearanceMethod m);

  struct AppearanceParams
  {
    float max_range = 40.0f;
    float ground_height_threshold = 0.3f; // ground removal for gravity-aligned scan
    int search_window = 7;                // SC++ sector key search window
  };

  // Method-agnostic descriptor (holds data for whichever method is active)
  struct AppearanceDescriptor
  {
    // SC++ data
    Eigen::MatrixXf sc_descriptor; // NR x NS (40 x 120)
    Eigen::VectorXf ring_key;      // NR (40)
    Eigen::VectorXf sector_key;    // NS (120)

    // (STD data will be added here)
  };

  // Database entry: descriptor + pose + GPS metadata
  struct AppearanceEntry
  {
    AppearanceDescriptor descriptor;
    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;
    double gps_latitude = 0.0;
    double gps_longitude = 0.0;
    double gps_altitude = 0.0;
    bool gps_valid = false;
    float gps_horizontal_accuracy = 0.f;
    int8_t gps_status = -1; // sensor_msgs::msg::NavSatStatus::status: -1=NO_FIX, 0=FIX, 1=SBAS, 2=GBAS
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  class AppearanceEngine
  {
  public:
    AppearanceEngine() = default;

    void init(AppearanceMethod method, const AppearanceParams &params,
              rclcpp::Logger logger = rclcpp::get_logger("appearance_engine"));

    // Compute descriptor from a point cloud (should be gravity-aligned for SC++)
    AppearanceDescriptor computeDescriptor(pcl::PointCloud<PointType>::ConstPtr cloud);

    // Compare two descriptors; returns (distance, alignment_shift)
    std::pair<float, int> compareDescriptors(
        const AppearanceDescriptor &a, const AppearanceDescriptor &b);

    // Find top-K candidates from database sorted by distance (ascending)
    // Returns vector of (db_index, distance). max_dist <= 0 disables threshold.
    std::vector<std::pair<int, float>> findCandidates(
        const AppearanceDescriptor &query,
        const std::vector<AppearanceEntry> &database,
        int top_k, float max_dist = 0.0f);

    // Prepare gravity-aligned scan for descriptor computation
    pcl::PointCloud<PointType>::Ptr prepareGravityAlignedScan(
        pcl::PointCloud<PointType>::ConstPtr scan,
        const Eigen::Quaternionf &gravity_q,
        const Eigen::Matrix3f &R_body_to_lidar);

    // Convert method-specific alignment shift to yaw angle (radians)
    float shiftToYaw(int shift) const;

    // Number of sectors (for external yaw computation if needed)
    int numSectors() const;

    // Accessors
    AppearanceMethod method() const { return method_; }
    float maxRange() const { return params_.max_range; }
    float groundHeightThreshold() const { return params_.ground_height_threshold; }
    int searchWindow() const { return params_.search_window; }
    bool isCuda() const
    {
      return method_ == AppearanceMethod::SC_PLUS_PLUS_CUDA ||
             method_ == AppearanceMethod::STD_CUDA;
    }

  private:
    AppearanceMethod method_ = AppearanceMethod::SC_PLUS_PLUS;
    AppearanceParams params_;
    bool initialized_ = false;
    rclcpp::Logger logger_ = rclcpp::get_logger("appearance_engine");
  };

} // namespace c_lio
