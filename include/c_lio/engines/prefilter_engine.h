#pragma once

#include "c_lio/c_lio.h"
#include <pcl/point_cloud.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <string>

namespace c_lio {

enum class PrefilterMethod {
  NONE,
  SOR,       // Statistical Outlier Removal (PCL)
  SOR_CUDA,  // GPU SOR (future)
  ROR,       // Radius Outlier Removal (PCL)
  DROR       // Dynamic Radius Outlier Removal (distance-adaptive)
};

PrefilterMethod parsePrefilterMethod(const std::string& s);
std::string prefilterMethodToString(PrefilterMethod m);

struct PrefilterParams {
  // SOR
  int mean_k = 10;            // number of neighbors for mean distance estimation
  double stddev_mul = 1.0;    // standard deviation multiplier threshold

  // ROR
  double radius = 0.5;        // search radius (meters)
  int min_neighbors = 3;      // minimum neighbors within radius

  // DROR
  double dror_min_radius = 0.1;   // minimum radius near sensor
  double dror_max_radius = 2.0;   // maximum radius far from sensor
  double dror_range_scale = 0.01; // radius = max(min_radius, range * scale)
  int dror_min_neighbors = 3;
};

class PrefilterEngine {
public:
  PrefilterEngine() = default;

  void init(PrefilterMethod method, const PrefilterParams& params);

  // Filter point cloud in-place (returns filtered cloud)
  pcl::PointCloud<PointType>::Ptr filter(pcl::PointCloud<PointType>::ConstPtr cloud);

  PrefilterMethod method() const { return method_; }
  bool isEnabled() const { return method_ != PrefilterMethod::NONE; }

private:
  pcl::PointCloud<PointType>::Ptr filterSOR(pcl::PointCloud<PointType>::ConstPtr cloud);
  pcl::PointCloud<PointType>::Ptr filterROR(pcl::PointCloud<PointType>::ConstPtr cloud);
  pcl::PointCloud<PointType>::Ptr filterDROR(pcl::PointCloud<PointType>::ConstPtr cloud);

  PrefilterMethod method_ = PrefilterMethod::NONE;
  PrefilterParams params_;
};

} // namespace c_lio
