#pragma once

#include "dlio/dlio.h"
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <memory>

namespace dlio {
namespace cuda {

class RobustIcpCudaImpl;

class RobustIcpCuda {
public:
  RobustIcpCuda();
  ~RobustIcpCuda();

  void setInputSource(pcl::PointCloud<PointType>::ConstPtr cloud);
  void setInputTarget(pcl::PointCloud<PointType>::ConstPtr cloud);
  void align(pcl::PointCloud<PointType>& output);
  void align(pcl::PointCloud<PointType>& output, const Eigen::Matrix4f& guess);
  Eigen::Matrix4f getFinalTransformation();
  double getFitnessScore(double max_range = 1.0);
  bool hasConverged();

  void setMaxIterations(int n) { max_iterations_ = n; }
  void setMaxCorrespondenceDistance(double d) { max_corr_dist_ = d; }
  void setKernelScale(double s) { kernel_scale_ = s; }
  void setConvergenceEpsilon(double e) { convergence_eps_ = e; }

private:
  int max_iterations_ = 64;
  double max_corr_dist_ = 1.5;
  double kernel_scale_ = 0.0;
  double convergence_eps_ = 0.001;
  Eigen::Matrix4f final_transformation_ = Eigen::Matrix4f::Identity();
  bool converged_ = false;

  pcl::PointCloud<PointType>::ConstPtr source_;
  pcl::PointCloud<PointType>::ConstPtr target_;

  std::unique_ptr<RobustIcpCudaImpl> impl_;
};

} // namespace cuda
} // namespace dlio
