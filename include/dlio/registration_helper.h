#ifndef DLIO_REGISTRATION_HELPER_H
#define DLIO_REGISTRATION_HELPER_H

#include "dlio/dlio.h"
#include "dlio/robust_icp.h"
#include <nano_gicp/nano_gicp.h>
#include <pclomp/voxel_grid_covariance_omp.h>
#include <pclomp/voxel_grid_covariance_omp_impl.hpp>
#include <pclomp/ndt_omp.h>
#include <pclomp/ndt_omp_impl.hpp>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>
#include <string>

namespace dlio
{

  struct RegistrationResult
  {
    bool converged = false;
    double fitness = std::numeric_limits<double>::max();
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  class RegistrationHelper
  {
  public:
    RegistrationHelper() = default;

    void setMethod(const std::string &method) { method_ = method; }
    void setMaxCorrespondenceDistance(double d) { max_corr_dist_ = d; }
    void setMaxIterations(int n) { max_iterations_ = n; }
    void setTransformationEpsilon(double e) { transformation_ep_ = e; }
    void setRotationEpsilon(double e) { rotation_ep_ = e; }
    void setGICPCorrespondenceRandomness(int k) { gicp_k_correspondences_ = k; }
    void setNDTResolution(double r) { ndt_resolution_ = r; }
    void setNDTNumThreads(int n) { ndt_num_threads_ = n; }
    void setRobustICPKernelScale(double s) { robust_icp_kernel_scale_ = s; }

    // Align source to target with optional initial guess
    RegistrationResult align(
        pcl::PointCloud<PointType>::ConstPtr source,
        pcl::PointCloud<PointType>::ConstPtr target,
        const Eigen::Matrix4f &initial_guess = Eigen::Matrix4f::Identity());

    std::string getMethod() const { return method_; }

  private:
    RegistrationResult alignGICP(
        pcl::PointCloud<PointType>::ConstPtr source,
        pcl::PointCloud<PointType>::ConstPtr target,
        const Eigen::Matrix4f &guess);

    RegistrationResult alignNDT(
        pcl::PointCloud<PointType>::ConstPtr source,
        pcl::PointCloud<PointType>::ConstPtr target,
        const Eigen::Matrix4f &guess);

    RegistrationResult alignRobustICP(
        pcl::PointCloud<PointType>::ConstPtr source,
        pcl::PointCloud<PointType>::ConstPtr target,
        const Eigen::Matrix4f &guess);

    std::string method_ = "gicp";
    double max_corr_dist_ = 1.5;
    int max_iterations_ = 64;
    double transformation_ep_ = 0.05;
    double rotation_ep_ = 0.05;
    int gicp_k_correspondences_ = 16;
    double ndt_resolution_ = 1.0;
    int ndt_num_threads_ = 4;
    double robust_icp_kernel_scale_ = 0.0;
  };

} // namespace dlio

#endif // DLIO_REGISTRATION_HELPER_H
