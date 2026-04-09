#pragma once

#include "dlio/dlio.h"
#include "dlio/algorithms/nano_gicp/nano_gicp.h"
#include "dlio/algorithms/robust_icp.h"
#if DLIO_HAS_CUDA
#include "dlio/cuda/robust_icp_cuda.h"
#include <ndt_cuda/ndt_cuda.h>
#endif
#include <pclomp/voxel_grid_covariance_omp.h>
#include <pclomp/voxel_grid_covariance_omp_impl.hpp>
#include <pclomp/ndt_omp.h>
#include <pclomp/ndt_omp_impl.hpp>

#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <memory>
#include <optional>

namespace dlio
{

  enum class RegistrationMethod
  {
    GICP,
    GICP_CUDA,
    NDT,
    NDT_CUDA,
    ROBUST_ICP,
    ROBUST_ICP_CUDA
  };

  RegistrationMethod parseRegistrationMethod(const std::string &s);
  std::string registrationMethodToString(RegistrationMethod m);

  struct RegistrationResult
  {
    bool converged = false;
    double fitness = std::numeric_limits<double>::max();
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct RegistrationParams
  {
    // GICP
    int gicp_k_correspondences = 8;
    double gicp_max_corr_dist = 0.5;
    int gicp_max_iter = 64;
    double gicp_transformation_ep = 0.001;
    double gicp_rotation_ep = 0.001;
    double gicp_init_lambda_factor = 1e-9;
    float gicp_gpu_voxel_size = 0.5f;
    // NDT
    float ndt_resolution = 1.0f;
    int ndt_num_threads = 4;
    std::string ndt_search_method = "DIRECT7";
    double ndt_step_size = 0.1;
    // Robust ICP
    double robust_icp_kernel_scale = 0.0;
  };

  class RegistrationEngine
  {
  public:
    RegistrationEngine() = default;

    void init(RegistrationMethod method, const RegistrationParams &params,
              rclcpp::Logger logger = rclcpp::get_logger("registration_engine"));

    // Input
    void setInputSource(pcl::PointCloud<PointType>::ConstPtr cloud);
    void setInputTarget(pcl::PointCloud<PointType>::ConstPtr cloud);
    void registerInputTarget(pcl::PointCloud<PointType>::ConstPtr cloud);

    // GICP-specific: set external kdtree/covariances for target
    void setTargetKdTree(std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> kdtree);
    void setTargetCovariances(std::shared_ptr<const nano_gicp::CovarianceList> covs);

    // Registration
    void align(pcl::PointCloud<PointType> &output);

    // Results
    Eigen::Matrix4f getFinalTransformation();
    double getFitnessScore(double max_range = 1.0);
    bool hasConverged();

    // Covariance queries (only meaningful for GICP/GICP_CUDA)
    bool needsCovariances() const;
    bool needsKdTree() const;
    void calculateSourceCovariances();
    float getSourceDensity();
    std::shared_ptr<const nano_gicp::CovarianceList> getSourceCovariances();

    // KdTree access (for GICP/robust_icp that build target kdtree)
    std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> getTargetKdTree();

    // One-shot align (convenience, matches RegistrationHelper API)
    RegistrationResult align(
        pcl::PointCloud<PointType>::ConstPtr source,
        pcl::PointCloud<PointType>::ConstPtr target,
        const Eigen::Matrix4f &initial_guess = Eigen::Matrix4f::Identity());

    // Adaptive params
    void setMaxCorrespondenceDistance(double d);

    // Accessors
    RegistrationMethod method() const { return method_; }
    bool isGicp() const { return method_ == RegistrationMethod::GICP || method_ == RegistrationMethod::GICP_CUDA; }
    bool isNdt() const { return method_ == RegistrationMethod::NDT || method_ == RegistrationMethod::NDT_CUDA; }
    bool isRobustIcp() const { return method_ == RegistrationMethod::ROBUST_ICP || method_ == RegistrationMethod::ROBUST_ICP_CUDA; }
    bool isCuda() const { return method_ == RegistrationMethod::GICP_CUDA || method_ == RegistrationMethod::NDT_CUDA || method_ == RegistrationMethod::ROBUST_ICP_CUDA; }

  private:
    RegistrationMethod method_ = RegistrationMethod::GICP;
    RegistrationMethod effective_method_ = RegistrationMethod::GICP; // actual method after fallback
    bool initialized_ = false;
    rclcpp::Logger logger_ = rclcpp::get_logger("registration_engine");

    nano_gicp::NanoGICP<PointType, PointType> gicp_;
    pclomp::NormalDistributionsTransform<PointType, PointType> ndt_;
#if DLIO_HAS_CUDA
    ndt_cuda::NormalDistributionsTransformCUDA<PointType, PointType> ndt_cuda_;
#endif
    dlio::RobustICP robust_icp_;
#if DLIO_HAS_CUDA
    dlio::cuda::RobustIcpCuda robust_icp_cuda_;
#endif
  };

} // namespace dlio
