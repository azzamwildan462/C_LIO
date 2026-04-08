#include "dlio/robust_icp.h"
#include <pcl/common/transforms.h>
#include <cmath>

namespace dlio
{

  RobustICP::RobustICP()
  {
    final_transformation_ = Eigen::Matrix4f::Identity();
  }

  Eigen::Matrix3f RobustICP::hat(const Eigen::Vector3f &v)
  {
    Eigen::Matrix3f m;
    m << 0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0;
    return m;
  }

  Eigen::Matrix4f RobustICP::expSE3(const Eigen::Matrix<float, 6, 1> &twist)
  {
    Eigen::Vector3f t = twist.head<3>();
    Eigen::Vector3f w = twist.tail<3>();
    float theta = w.norm();

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

    if (theta < 1e-6f)
    {
      // Small angle: first-order approximation
      T.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() + hat(w);
      T.block<3, 1>(0, 3) = t;
    }
    else
    {
      // Rodrigues rotation
      Eigen::Matrix3f W = hat(w / theta);
      float s = std::sin(theta);
      float c = std::cos(theta);
      Eigen::Matrix3f R = Eigen::Matrix3f::Identity() + s * W + (1.0f - c) * W * W;

      // V matrix for translation
      Eigen::Matrix3f V = Eigen::Matrix3f::Identity() +
                          ((1.0f - c) / theta) * W +
                          ((theta - s) / theta) * W * W;

      T.block<3, 3>(0, 0) = R;
      T.block<3, 1>(0, 3) = V * t;
    }

    return T;
  }

  void RobustICP::setInputSource(pcl::PointCloud<PointType>::ConstPtr cloud)
  {
    source_ = cloud;
  }

  void RobustICP::setInputTarget(pcl::PointCloud<PointType>::ConstPtr cloud)
  {
    target_ = cloud;
    auto tree = std::make_shared<nanoflann::KdTreeFLANN<PointType>>();
    tree->setInputCloud(cloud);
    target_kdtree_ = tree;
  }

  void RobustICP::setTargetKdTree(std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> tree)
  {
    target_kdtree_ = tree;
  }

  void RobustICP::align(pcl::PointCloud<PointType> &output)
  {
    converged_ = false;
    final_transformation_ = Eigen::Matrix4f::Identity();

    if (!source_ || source_->empty() || !target_ || target_->empty() || !target_kdtree_)
      return;

    double sigma = (kernel_scale_ > 0.0) ? kernel_scale_ : max_corr_dist_;
    double sigma2 = sigma * sigma;
    double max_dist2 = max_corr_dist_ * max_corr_dist_;

    // Copy source points (will be transformed in-place)
    std::vector<Eigen::Vector3f> src_pts(source_->size());
    for (size_t i = 0; i < source_->size(); ++i)
    {
      src_pts[i] = source_->points[i].getVector3fMap();
    }

    Eigen::Matrix4f T_accum = Eigen::Matrix4f::Identity();

    for (int iter = 0; iter < max_iterations_; ++iter)
    {
      // Normal equations
      Eigen::Matrix<float, 6, 6> JTJ = Eigen::Matrix<float, 6, 6>::Zero();
      Eigen::Matrix<float, 6, 1> JTr = Eigen::Matrix<float, 6, 1>::Zero();

      std::vector<int> nn_idx(1);
      std::vector<float> nn_dist2(1);
      int n_corr = 0;

      for (size_t i = 0; i < src_pts.size(); ++i)
      {
        if (!std::isfinite(src_pts[i][0]))
          continue;

        PointType query;
        query.x = src_pts[i][0];
        query.y = src_pts[i][1];
        query.z = src_pts[i][2];

        target_kdtree_->nearestKSearch(query, 1, nn_idx, nn_dist2);

        if (nn_dist2[0] > max_dist2)
          continue;

        const auto &tgt_pt = target_->points[nn_idx[0]];
        Eigen::Vector3f target_pt(tgt_pt.x, tgt_pt.y, tgt_pt.z);
        Eigen::Vector3f residual = src_pts[i] - target_pt;

        // Point-to-point Jacobian: J = [I | -hat(source)]
        Eigen::Matrix<float, 3, 6> J;
        J.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
        J.block<3, 3>(0, 3) = -hat(src_pts[i]);

        // Geman-McClure robust weight
        float r2 = residual.squaredNorm();
        float w = static_cast<float>(sigma2 / ((sigma2 + r2) * (sigma2 + r2)));

        // Accumulate weighted normal equations
        JTJ += J.transpose() * w * J;
        JTr += J.transpose() * w * residual;

        n_corr++;
      }

      if (n_corr < 10)
        break;

      // Solve for 6D twist
      Eigen::Matrix<float, 6, 1> dx = JTJ.ldlt().solve(-JTr);

      // Apply incremental transform
      Eigen::Matrix4f T_inc = expSE3(dx);

      // Update source points
      for (auto &p : src_pts)
      {
        Eigen::Vector3f tp = T_inc.block<3, 3>(0, 0) * p + T_inc.block<3, 1>(0, 3);
        p = tp;
      }

      T_accum = T_inc * T_accum;

      // Convergence check
      if (dx.norm() < convergence_eps_)
      {
        converged_ = true;
        break;
      }
    }

    final_transformation_ = T_accum;

    // Output = transformed source
    pcl::transformPointCloud(*source_, output, final_transformation_);
  }

  void RobustICP::align(pcl::PointCloud<PointType> &output, const Eigen::Matrix4f &guess)
  {
    // Transform source by initial guess, then run normal align
    auto transformed_source = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud(*source_, *transformed_source, guess);
    auto original_source = source_;
    source_ = transformed_source;
    align(output);
    // Final transformation includes the initial guess
    final_transformation_ = final_transformation_ * guess;
    source_ = original_source;
    // Re-transform output with correct final
    pcl::transformPointCloud(*original_source, output, final_transformation_);
  }

  double RobustICP::getFitnessScore(double max_range) const
  {
    if (!source_ || !target_ || !target_kdtree_)
      return std::numeric_limits<double>::max();

    pcl::PointCloud<PointType> transformed;
    pcl::transformPointCloud(*source_, transformed, final_transformation_);

    double score = 0.0;
    int nr = 0;
    std::vector<int> nn_idx(1);
    std::vector<float> nn_dist(1);

    for (const auto &pt : transformed.points)
    {
      if (!std::isfinite(pt.x))
        continue;
      target_kdtree_->nearestKSearch(pt, 1, nn_idx, nn_dist);
      if (nn_dist[0] <= max_range)
      {
        score += nn_dist[0];
        nr++;
      }
    }

    return (nr > 0) ? (score / nr) : std::numeric_limits<double>::max();
  }

} // namespace dlio
