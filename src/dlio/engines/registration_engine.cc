#include "dlio/engines/registration_engine.h"
#include <pcl/common/transforms.h>
#include <stdexcept>

namespace dlio {

RegistrationMethod parseRegistrationMethod(const std::string& s) {
  if (s == "gicp") return RegistrationMethod::GICP;
  if (s == "gicp_cuda") return RegistrationMethod::GICP_CUDA;
  if (s == "ndt") return RegistrationMethod::NDT;
  if (s == "ndt_cuda") return RegistrationMethod::NDT_CUDA;
  if (s == "robust_icp") return RegistrationMethod::ROBUST_ICP;
  if (s == "robust_icp_cuda") return RegistrationMethod::ROBUST_ICP_CUDA;
  throw std::runtime_error("Unknown registration method: " + s);
}

std::string registrationMethodToString(RegistrationMethod m) {
  switch (m) {
    case RegistrationMethod::GICP: return "gicp";
    case RegistrationMethod::GICP_CUDA: return "gicp_cuda";
    case RegistrationMethod::NDT: return "ndt";
    case RegistrationMethod::NDT_CUDA: return "ndt_cuda";
    case RegistrationMethod::ROBUST_ICP: return "robust_icp";
    case RegistrationMethod::ROBUST_ICP_CUDA: return "robust_icp_cuda";
  }
  return "unknown";
}

void RegistrationEngine::init(RegistrationMethod method, const RegistrationParams& p,
                               rclcpp::Logger logger) {
  method_ = method;
  logger_ = logger;
  initialized_ = true;

  // Determine effective method (fallback when CUDA/NDT_CUDA not compiled in).
  // DLIO_HAS_CUDA gates native CUDA methods (GICP_CUDA, ROBUST_ICP_CUDA).
  // DLIO_HAS_NDT_CUDA gates only the external ndt_cuda_ros2 method.
  switch (method_) {
    case RegistrationMethod::GICP_CUDA:
#if DLIO_HAS_CUDA
      effective_method_ = RegistrationMethod::GICP_CUDA;
#else
      RCLCPP_WARN(logger_, "GICP_CUDA requested but CUDA not available, falling back to GICP");
      effective_method_ = RegistrationMethod::GICP;
#endif
      break;
    case RegistrationMethod::ROBUST_ICP_CUDA:
#if DLIO_HAS_CUDA
      effective_method_ = RegistrationMethod::ROBUST_ICP_CUDA;
#else
      RCLCPP_WARN(logger_, "ROBUST_ICP_CUDA requested but CUDA not available, falling back to ROBUST_ICP");
      effective_method_ = RegistrationMethod::ROBUST_ICP;
#endif
      break;
    case RegistrationMethod::NDT_CUDA:
#if DLIO_HAS_NDT_CUDA
      effective_method_ = RegistrationMethod::NDT_CUDA;
#else
      RCLCPP_WARN(logger_, "NDT_CUDA requested but ndt_cuda_ros2 not available, falling back to NDT");
      effective_method_ = RegistrationMethod::NDT;
#endif
      break;
    default:
      effective_method_ = method_;
      break;
  }

  // Initialize the effective method
  switch (effective_method_) {
    case RegistrationMethod::GICP:
    case RegistrationMethod::GICP_CUDA: {
      gicp_.setCorrespondenceRandomness(p.gicp_k_correspondences);
      gicp_.setMaxCorrespondenceDistance(p.gicp_max_corr_dist);
      gicp_.setMaximumIterations(p.gicp_max_iter);
      gicp_.setTransformationEpsilon(p.gicp_transformation_ep);
      gicp_.setRotationEpsilon(p.gicp_rotation_ep);
      gicp_.setInitialLambdaFactor(p.gicp_init_lambda_factor);
#if DLIO_HAS_CUDA
      if (effective_method_ == RegistrationMethod::GICP_CUDA)
        gicp_.setUseGpu(true, p.gicp_gpu_voxel_size);
#endif
      pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
      gicp_.setSearchMethodSource(temp, true);
      gicp_.setSearchMethodTarget(temp, true);
      break;
    }
    case RegistrationMethod::NDT: {
      auto search = pclomp::DIRECT7;
      if (p.ndt_search_method == "KDTREE") search = pclomp::KDTREE;
      else if (p.ndt_search_method == "DIRECT1") search = pclomp::DIRECT1;
      else if (p.ndt_search_method == "DIRECT26") search = pclomp::DIRECT26;
      ndt_.setResolution(p.ndt_resolution);
      ndt_.setNumThreads(p.ndt_num_threads);
      ndt_.setNeighborhoodSearchMethod(search);
      ndt_.setStepSize(p.ndt_step_size);
      ndt_.setMaximumIterations(p.gicp_max_iter);
      ndt_.setTransformationEpsilon(p.gicp_transformation_ep);
      break;
    }
#if DLIO_HAS_NDT_CUDA
    case RegistrationMethod::NDT_CUDA: {
      ndt_cuda_.setResolution(p.ndt_resolution);
      ndt_cuda_.setStepSize(p.ndt_step_size);
      ndt_cuda_.setMaximumIterations(p.gicp_max_iter);
      ndt_cuda_.setTransformationEpsilon(p.gicp_transformation_ep);
      break;
    }
#endif
    case RegistrationMethod::ROBUST_ICP: {
      robust_icp_.setMaxIterations(p.gicp_max_iter);
      robust_icp_.setMaxCorrespondenceDistance(p.gicp_max_corr_dist);
      robust_icp_.setConvergenceEpsilon(p.gicp_transformation_ep);
      robust_icp_.setKernelScale(p.robust_icp_kernel_scale);
      break;
    }
#if DLIO_HAS_CUDA
    case RegistrationMethod::ROBUST_ICP_CUDA: {
      robust_icp_cuda_.setMaxIterations(p.gicp_max_iter);
      robust_icp_cuda_.setMaxCorrespondenceDistance(p.gicp_max_corr_dist);
      robust_icp_cuda_.setConvergenceEpsilon(p.gicp_transformation_ep);
      robust_icp_cuda_.setKernelScale(p.robust_icp_kernel_scale);
      break;
    }
#endif
    default: break;
  }
}

// All dispatch below uses effective_method_ (CPU fallback when CUDA unavailable)

void RegistrationEngine::setInputSource(pcl::PointCloud<PointType>::ConstPtr cloud) {
  switch (effective_method_) {
    case RegistrationMethod::GICP:
    case RegistrationMethod::GICP_CUDA:
      gicp_.setInputSource(cloud);
      gicp_.calculateSourceCovariances();
      break;
    case RegistrationMethod::NDT:
      ndt_.setInputSource(cloud);
      break;
#if DLIO_HAS_NDT_CUDA
    case RegistrationMethod::NDT_CUDA:
      ndt_cuda_.setInputSource(cloud);
      break;
#endif
    case RegistrationMethod::ROBUST_ICP:
      robust_icp_.setInputSource(cloud);
      break;
#if DLIO_HAS_CUDA
    case RegistrationMethod::ROBUST_ICP_CUDA:
      robust_icp_cuda_.setInputSource(cloud);
      break;
#endif
    default: break;
  }
}

void RegistrationEngine::setInputTarget(pcl::PointCloud<PointType>::ConstPtr cloud) {
  switch (effective_method_) {
    case RegistrationMethod::GICP:
    case RegistrationMethod::GICP_CUDA:
      gicp_.setInputTarget(cloud);
      break;
    case RegistrationMethod::NDT:
      ndt_.setInputTarget(cloud);
      break;
#if DLIO_HAS_NDT_CUDA
    case RegistrationMethod::NDT_CUDA:
      ndt_cuda_.setInputTarget(cloud);
      break;
#endif
    case RegistrationMethod::ROBUST_ICP:
      robust_icp_.setInputTarget(cloud);
      break;
#if DLIO_HAS_CUDA
    case RegistrationMethod::ROBUST_ICP_CUDA:
      robust_icp_cuda_.setInputTarget(cloud);
      break;
#endif
    default: break;
  }
}

void RegistrationEngine::registerInputTarget(pcl::PointCloud<PointType>::ConstPtr cloud) {
  switch (effective_method_) {
    case RegistrationMethod::GICP:
    case RegistrationMethod::GICP_CUDA:
      gicp_.registerInputTarget(cloud);
      break;
    case RegistrationMethod::NDT:
      ndt_.setInputTarget(cloud);
      break;
#if DLIO_HAS_NDT_CUDA
    case RegistrationMethod::NDT_CUDA:
      ndt_cuda_.setInputTarget(cloud);
      break;
#endif
    case RegistrationMethod::ROBUST_ICP:
      robust_icp_.setInputTarget(cloud);
      break;
#if DLIO_HAS_CUDA
    case RegistrationMethod::ROBUST_ICP_CUDA:
      robust_icp_cuda_.setInputTarget(cloud);
      break;
#endif
    default: break;
  }
}

void RegistrationEngine::setTargetKdTree(std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> kdtree) {
  if (isGicp()) gicp_.target_kdtree_ = kdtree;
  else if (effective_method_ == RegistrationMethod::ROBUST_ICP) robust_icp_.setTargetKdTree(kdtree);
}

void RegistrationEngine::setTargetCovariances(std::shared_ptr<const nano_gicp::CovarianceList> covs) {
  if (isGicp()) gicp_.setTargetCovariances(covs);
}

void RegistrationEngine::align(pcl::PointCloud<PointType>& output) {
  switch (effective_method_) {
    case RegistrationMethod::GICP:
    case RegistrationMethod::GICP_CUDA:
      gicp_.align(output); break;
    case RegistrationMethod::NDT:
      ndt_.align(output); break;
#if DLIO_HAS_NDT_CUDA
    case RegistrationMethod::NDT_CUDA:
      ndt_cuda_.align(output); break;
#endif
    case RegistrationMethod::ROBUST_ICP:
      robust_icp_.align(output); break;
#if DLIO_HAS_CUDA
    case RegistrationMethod::ROBUST_ICP_CUDA:
      robust_icp_cuda_.align(output); break;
#endif
    default: break;
  }
}

Eigen::Matrix4f RegistrationEngine::getFinalTransformation() {
  switch (effective_method_) {
    case RegistrationMethod::GICP:
    case RegistrationMethod::GICP_CUDA:
      return gicp_.getFinalTransformation();
    case RegistrationMethod::NDT:
      return ndt_.getFinalTransformation();
#if DLIO_HAS_NDT_CUDA
    case RegistrationMethod::NDT_CUDA:
      return ndt_cuda_.getFinalTransformation();
#endif
    case RegistrationMethod::ROBUST_ICP:
      return robust_icp_.getFinalTransformation();
#if DLIO_HAS_CUDA
    case RegistrationMethod::ROBUST_ICP_CUDA:
      return robust_icp_cuda_.getFinalTransformation();
#endif
    default: break;
  }
  return Eigen::Matrix4f::Identity();
}

double RegistrationEngine::getFitnessScore(double max_range) {
  switch (effective_method_) {
    case RegistrationMethod::GICP:
    case RegistrationMethod::GICP_CUDA: {
      if (!gicp_.target_kdtree_ || !gicp_.getInputSource()) return std::numeric_limits<double>::max();
      double score = 0.0;
      int nr = 0;
      pcl::PointCloud<PointType> transformed;
      pcl::transformPointCloud(*gicp_.getInputSource(), transformed, gicp_.getFinalTransformation());
      std::vector<int> nn_idx(1);
      std::vector<float> nn_dist(1);
      for (const auto& pt : transformed.points) {
        if (!std::isfinite(pt.x)) continue;
        gicp_.target_kdtree_->nearestKSearch(pt, 1, nn_idx, nn_dist);
        if (nn_dist[0] <= max_range) { score += nn_dist[0]; nr++; }
      }
      return (nr > 0) ? (score / nr) : std::numeric_limits<double>::max();
    }
    case RegistrationMethod::NDT:
      try { return ndt_.getFitnessScore(max_range); }
      catch (...) { return ndt_.hasConverged() ? 0.1 : 1.0; }
#if DLIO_HAS_NDT_CUDA
    case RegistrationMethod::NDT_CUDA:
      try { return ndt_cuda_.getFitnessScore(max_range); }
      catch (...) { return ndt_cuda_.hasConverged() ? 0.1 : 1.0; }
#endif
    case RegistrationMethod::ROBUST_ICP:
      return robust_icp_.getFitnessScore(max_range);
#if DLIO_HAS_CUDA
    case RegistrationMethod::ROBUST_ICP_CUDA:
      return robust_icp_cuda_.getFitnessScore(max_range);
#endif
    default: break;
  }
  return std::numeric_limits<double>::max();
}

bool RegistrationEngine::hasConverged() {
  switch (effective_method_) {
    case RegistrationMethod::GICP:
    case RegistrationMethod::GICP_CUDA:
      return gicp_.hasConverged();
    case RegistrationMethod::NDT:
      return ndt_.hasConverged();
#if DLIO_HAS_NDT_CUDA
    case RegistrationMethod::NDT_CUDA:
      return ndt_cuda_.hasConverged();
#endif
    case RegistrationMethod::ROBUST_ICP:
      return robust_icp_.hasConverged();
#if DLIO_HAS_CUDA
    case RegistrationMethod::ROBUST_ICP_CUDA:
      return robust_icp_cuda_.hasConverged();
#endif
    default: break;
  }
  return false;
}

bool RegistrationEngine::needsCovariances() const { return isGicp(); }
bool RegistrationEngine::needsKdTree() const { return isGicp() || isRobustIcp(); }

void RegistrationEngine::calculateSourceCovariances() {
  if (isGicp()) gicp_.calculateSourceCovariances();
}

float RegistrationEngine::getSourceDensity() {
  if (isGicp()) return gicp_.source_density_;
  return 0.0f;
}

std::shared_ptr<const nano_gicp::CovarianceList> RegistrationEngine::getSourceCovariances() {
  if (isGicp()) return gicp_.getSourceCovariances();
  return nullptr;
}

std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> RegistrationEngine::getTargetKdTree() {
  if (isGicp()) return gicp_.target_kdtree_;
  if (effective_method_ == RegistrationMethod::ROBUST_ICP) return robust_icp_.target_kdtree_;
  return nullptr;
}

RegistrationResult RegistrationEngine::align(
    pcl::PointCloud<PointType>::ConstPtr source,
    pcl::PointCloud<PointType>::ConstPtr target,
    const Eigen::Matrix4f& initial_guess) {
  // The backend align(output) overloads start from IDENTITY (they don't take a
  // guess), so we MUST apply initial_guess ourselves — otherwise relocalization
  // ignores the seed and always converges to the same attractor. Pre-transform
  // the source into the guessed frame, align (a small local refinement from
  // there), then fold the guess back into the result.
  pcl::PointCloud<PointType>::Ptr source_guessed = std::make_shared<pcl::PointCloud<PointType>>();
  if (initial_guess.isApprox(Eigen::Matrix4f::Identity()))
    *source_guessed = *source;
  else
    pcl::transformPointCloud(*source, *source_guessed, initial_guess);

  setInputSource(source_guessed);
  setInputTarget(target);
  pcl::PointCloud<PointType> aligned;
  align(aligned);

  RegistrationResult result;
  result.converged = hasConverged();
  result.fitness = getFitnessScore(1.0);
  // getFinalTransformation() maps source_guessed → target; compose the guess
  // back so transformation maps the ORIGINAL source → target.
  result.transformation = getFinalTransformation() * initial_guess;
  return result;
}

void RegistrationEngine::setMaxCorrespondenceDistance(double d) {
  switch (effective_method_) {
    case RegistrationMethod::GICP:
    case RegistrationMethod::GICP_CUDA:
      gicp_.setMaxCorrespondenceDistance(d); break;
    case RegistrationMethod::ROBUST_ICP:
      robust_icp_.setMaxCorrespondenceDistance(d); break;
#if DLIO_HAS_CUDA
    case RegistrationMethod::ROBUST_ICP_CUDA:
      robust_icp_cuda_.setMaxCorrespondenceDistance(d); break;
#endif
    default: break;
  }
}

} // namespace dlio
