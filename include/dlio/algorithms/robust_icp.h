#ifndef DLIO_ROBUST_ICP_H
#define DLIO_ROBUST_ICP_H

#include "dlio/dlio.h"
#include <pcl/point_cloud.h>
#include "dlio/algorithms/nano_gicp/nanoflann_adaptor.h"
#include <Eigen/Dense>

namespace dlio
{

  class RobustICP
  {
  public:
    RobustICP();

    void setMaxIterations(int n) { max_iterations_ = n; }
    void setMaxCorrespondenceDistance(double d) { max_corr_dist_ = d; }
    void setKernelScale(double s) { kernel_scale_ = s; }
    void setConvergenceEpsilon(double e) { convergence_eps_ = e; }

    void setInputSource(pcl::PointCloud<PointType>::ConstPtr cloud);
    void setInputTarget(pcl::PointCloud<PointType>::ConstPtr cloud);
    void setTargetKdTree(std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> tree);

    void align(pcl::PointCloud<PointType> &output);
    void align(pcl::PointCloud<PointType> &output, const Eigen::Matrix4f &guess);

    Eigen::Matrix4f getFinalTransformation() const { return final_transformation_; }
    double getFitnessScore(double max_range) const;
    bool hasConverged() const { return converged_; }

    pcl::PointCloud<PointType>::ConstPtr getInputSource() const { return source_; }
    pcl::PointCloud<PointType>::ConstPtr getInputTarget() const { return target_; }
    std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> target_kdtree_;

  private:
    // SE3 exp map: twist vector (6D) → 4x4 transformation matrix
    static Eigen::Matrix4f expSE3(const Eigen::Matrix<float, 6, 1> &twist);

    // Skew-symmetric matrix from 3D vector
    static Eigen::Matrix3f hat(const Eigen::Vector3f &v);

    pcl::PointCloud<PointType>::ConstPtr source_;
    pcl::PointCloud<PointType>::ConstPtr target_;

    Eigen::Matrix4f final_transformation_;
    bool converged_ = false;

    int max_iterations_ = 64;
    double max_corr_dist_ = 1.5;
    double kernel_scale_ = 0.0; // 0 = auto (use max_corr_dist)
    double convergence_eps_ = 0.001;
  };

} // namespace dlio

#endif // DLIO_ROBUST_ICP_H
