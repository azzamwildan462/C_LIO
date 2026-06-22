#include "c_lio/engines/prefilter_engine.h"
#include <cmath>
#include <stdexcept>

namespace c_lio {

PrefilterMethod parsePrefilterMethod(const std::string& s) {
  if (s == "none") return PrefilterMethod::NONE;
  if (s == "sor") return PrefilterMethod::SOR;
  if (s == "sor_cuda") return PrefilterMethod::SOR_CUDA;
  if (s == "ror") return PrefilterMethod::ROR;
  if (s == "dror") return PrefilterMethod::DROR;
  throw std::runtime_error("Unknown prefilter method: " + s);
}

std::string prefilterMethodToString(PrefilterMethod m) {
  switch (m) {
    case PrefilterMethod::NONE: return "none";
    case PrefilterMethod::SOR: return "sor";
    case PrefilterMethod::SOR_CUDA: return "sor_cuda";
    case PrefilterMethod::ROR: return "ror";
    case PrefilterMethod::DROR: return "dror";
  }
  return "none";
}

void PrefilterEngine::init(PrefilterMethod method, const PrefilterParams& params) {
  method_ = method;
  params_ = params;
}

pcl::PointCloud<PointType>::Ptr PrefilterEngine::filter(pcl::PointCloud<PointType>::ConstPtr cloud) {
  if (!cloud || cloud->empty()) return std::make_shared<pcl::PointCloud<PointType>>();

  switch (method_) {
    case PrefilterMethod::NONE:
      return std::make_shared<pcl::PointCloud<PointType>>(*cloud);
    case PrefilterMethod::SOR:
    case PrefilterMethod::SOR_CUDA: // fallback to CPU until CUDA implemented
      return filterSOR(cloud);
    case PrefilterMethod::ROR:
      return filterROR(cloud);
    case PrefilterMethod::DROR:
      return filterDROR(cloud);
  }
  return std::make_shared<pcl::PointCloud<PointType>>(*cloud);
}

pcl::PointCloud<PointType>::Ptr PrefilterEngine::filterSOR(pcl::PointCloud<PointType>::ConstPtr cloud) {
  auto filtered = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::StatisticalOutlierRemoval<PointType> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(params_.mean_k);
  sor.setStddevMulThresh(params_.stddev_mul);
  sor.filter(*filtered);
  return filtered;
}

pcl::PointCloud<PointType>::Ptr PrefilterEngine::filterROR(pcl::PointCloud<PointType>::ConstPtr cloud) {
  auto filtered = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::RadiusOutlierRemoval<PointType> ror;
  ror.setInputCloud(cloud);
  ror.setRadiusSearch(params_.radius);
  ror.setMinNeighborsInRadius(params_.min_neighbors);
  ror.filter(*filtered);
  return filtered;
}

pcl::PointCloud<PointType>::Ptr PrefilterEngine::filterDROR(pcl::PointCloud<PointType>::ConstPtr cloud) {
  // Dynamic Radius Outlier Removal: radius scales with distance from sensor origin
  // Points farther from sensor have larger search radius (more tolerant)
  auto filtered = std::make_shared<pcl::PointCloud<PointType>>();
  filtered->reserve(cloud->size());

  // Build KdTree for neighbor search
  pcl::KdTreeFLANN<PointType> kdtree;
  kdtree.setInputCloud(cloud);

  std::vector<int> nn_indices;
  std::vector<float> nn_dists;

  for (size_t i = 0; i < cloud->size(); ++i) {
    const auto& pt = cloud->points[i];
    float range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);

    // Adaptive radius: farther points get larger radius
    double radius = std::max(params_.dror_min_radius, static_cast<double>(range) * params_.dror_range_scale);
    radius = std::min(radius, params_.dror_max_radius);

    int count = kdtree.radiusSearch(pt, radius, nn_indices, nn_dists);

    // Point is inlier if it has enough neighbors (including itself)
    if (count >= params_.dror_min_neighbors + 1) {
      filtered->push_back(pt);
    }
  }

  filtered->width = filtered->size();
  filtered->height = 1;
  filtered->is_dense = true;
  return filtered;
}

} // namespace c_lio
