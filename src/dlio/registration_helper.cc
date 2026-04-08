#include "dlio/registration_helper.h"

namespace dlio
{

  RegistrationResult RegistrationHelper::align(
      pcl::PointCloud<PointType>::ConstPtr source,
      pcl::PointCloud<PointType>::ConstPtr target,
      const Eigen::Matrix4f &initial_guess)
  {
    if (!source || source->empty() || !target || target->empty())
      return {};

    if (method_ == "ndt")
      return alignNDT(source, target, initial_guess);
    else if (method_ == "robust_icp")
      return alignRobustICP(source, target, initial_guess);
    else
      return alignGICP(source, target, initial_guess);
  }

  RegistrationResult RegistrationHelper::alignGICP(
      pcl::PointCloud<PointType>::ConstPtr source,
      pcl::PointCloud<PointType>::ConstPtr target,
      const Eigen::Matrix4f &guess)
  {
    nano_gicp::NanoGICP<PointType, PointType> gicp;
    gicp.setCorrespondenceRandomness(gicp_k_correspondences_);
    gicp.setMaxCorrespondenceDistance(max_corr_dist_);
    gicp.setMaximumIterations(max_iterations_);
    gicp.setTransformationEpsilon(transformation_ep_);
    gicp.setRotationEpsilon(rotation_ep_);

    gicp.setInputSource(source);
    gicp.calculateSourceCovariances();
    gicp.setInputTarget(target);
    gicp.calculateTargetCovariances();

    pcl::PointCloud<PointType> aligned;
    gicp.align(aligned, guess);

    RegistrationResult result;
    result.converged = gicp.hasConverged();
    result.transformation = gicp.getFinalTransformation();
    result.fitness = gicp.getFitnessScore(1.0);
    return result;
  }

  RegistrationResult RegistrationHelper::alignNDT(
      pcl::PointCloud<PointType>::ConstPtr source,
      pcl::PointCloud<PointType>::ConstPtr target,
      const Eigen::Matrix4f &guess)
  {
    pclomp::NormalDistributionsTransform<PointType, PointType> ndt;
    ndt.setResolution(ndt_resolution_);
    ndt.setNumThreads(ndt_num_threads_);
    ndt.setNeighborhoodSearchMethod(pclomp::DIRECT7);
    ndt.setMaximumIterations(max_iterations_);
    ndt.setTransformationEpsilon(transformation_ep_);

    ndt.setInputSource(source);
    ndt.setInputTarget(target);

    pcl::PointCloud<PointType> aligned;
    ndt.align(aligned, guess);

    RegistrationResult result;
    result.converged = ndt.hasConverged();
    result.transformation = ndt.getFinalTransformation();
    try
    {
      result.fitness = ndt.getFitnessScore(1.0);
    }
    catch (...)
    {
      result.fitness = result.converged ? 0.1 : 1.0;
    }
    return result;
  }

  RegistrationResult RegistrationHelper::alignRobustICP(
      pcl::PointCloud<PointType>::ConstPtr source,
      pcl::PointCloud<PointType>::ConstPtr target,
      const Eigen::Matrix4f &guess)
  {
    dlio::RobustICP icp;
    icp.setMaxIterations(max_iterations_);
    icp.setMaxCorrespondenceDistance(max_corr_dist_);
    icp.setConvergenceEpsilon(transformation_ep_);
    icp.setKernelScale(robust_icp_kernel_scale_);

    icp.setInputSource(source);
    icp.setInputTarget(target);

    pcl::PointCloud<PointType> aligned;
    icp.align(aligned, guess);

    RegistrationResult result;
    result.converged = icp.hasConverged();
    result.transformation = icp.getFinalTransformation();
    result.fitness = icp.getFitnessScore(1.0);
    return result;
  }

} // namespace dlio
