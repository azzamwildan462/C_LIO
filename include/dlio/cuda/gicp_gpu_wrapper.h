#ifndef DLIO_CUDA_GICP_GPU_WRAPPER_H
#define DLIO_CUDA_GICP_GPU_WRAPPER_H

// C++ header (no CUDA) — can be included from .cc files
// Provides opaque handle to GPU resources for GICP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <memory>

namespace dlio
{
  namespace cuda
  {

    // Forward declare (implemented in .cu)
    class GicpGpuImpl;

    // C++ wrapper around GPU GICP resources.
    // Owns: voxel hash map, device buffers for points/covariances/correspondences.
    class GicpGpuHandle
    {
    public:
      GicpGpuHandle();
      ~GicpGpuHandle();

      // Upload target cloud + build voxel hash map
      // points: Nx4 (x,y,z,intensity), voxel_size in meters
      void setTarget(const std::vector<Eigen::Vector4f> &points, float voxel_size);

      // Upload source cloud
      void setSource(const std::vector<Eigen::Vector4f> &points);

      // Upload covariances (4x4 matrices, row-major doubles → converted to float)
      void setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &covs);
      void setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &covs);

      // Run GPU correspondence search + Mahalanobis computation
      // trans: current 4x4 transform estimate (row-major float)
      // max_corr_dist: correspondence distance threshold
      // Returns: number of valid correspondences
      int updateCorrespondences(const Eigen::Isometry3d &trans, double max_corr_dist);

      // Download results to host vectors
      void getCorrespondences(std::vector<int> &correspondences) const;
      void getSqDistances(std::vector<float> &sq_distances) const;
      void getMahalanobis(std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &mahalanobis) const;

      // GPU linearize: compute H (6x6), b (6x1), and error sum.
      // Uses correspondences + mahalanobis from last updateCorrespondences() call.
      // H/b can be nullptr to compute error only.
      // Returns sum of weighted errors.
      double linearize(const Eigen::Isometry3d &trans,
                       Eigen::Matrix<double, 6, 6> *H,
                       Eigen::Matrix<double, 6, 1> *b);

      // GPU compute_error only (no H/b)
      double computeError(const Eigen::Isometry3d &trans);

      size_t sourceSize() const;
      size_t targetSize() const;

      bool isReady() const;
      void clear();

    private:
      std::unique_ptr<GicpGpuImpl> impl_;
    };

  } // namespace cuda
} // namespace dlio

#endif // DLIO_CUDA_GICP_GPU_WRAPPER_H
