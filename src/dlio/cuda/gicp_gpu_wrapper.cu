#include "dlio/cuda/gicp_gpu_wrapper.h"
#include "dlio/cuda/nn_search.cuh"
#include "dlio/cuda/linearize.cuh"
#include "dlio/cuda/point_cloud_cuda.cuh"
#include "dlio/cuda/common.cuh"

#include <cstring>

namespace dlio
{
  namespace cuda
  {

    class GicpGpuImpl
    {
    public:
      GicpGpuImpl() = default;
      ~GicpGpuImpl() { clear(); }

      void setTarget(const std::vector<Eigen::Vector4f> &points, float voxel_size)
      {
        n_target_ = points.size();
        if (n_target_ == 0)
          return;

        // Upload target points as float4
        std::vector<float4> h_pts(n_target_);
        for (size_t i = 0; i < n_target_; ++i)
        {
          h_pts[i] = make_float4(points[i].x(), points[i].y(), points[i].z(), points[i].w());
        }
        d_target_.upload(h_pts.data(), n_target_);

        // Build voxel hash map
        voxel_map_.build(d_target_.data(), n_target_, voxel_size);
        voxel_size_ = voxel_size;
      }

      void setSource(const std::vector<Eigen::Vector4f> &points)
      {
        n_source_ = points.size();
        if (n_source_ == 0)
          return;

        std::vector<float4> h_pts(n_source_);
        for (size_t i = 0; i < n_source_; ++i)
        {
          h_pts[i] = make_float4(points[i].x(), points[i].y(), points[i].z(), points[i].w());
        }
        d_source_.upload(h_pts.data(), n_source_);

        // Allocate output buffers
        d_correspondences_.allocate(n_source_);
        d_sq_distances_.allocate(n_source_);
        d_mahalanobis_.allocate(n_source_ * 16);
      }

      void setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &covs)
      {
        size_t n = covs.size();
        std::vector<float> h_covs(n * 16);
        for (size_t i = 0; i < n; ++i)
        {
          // Eigen is column-major, we need row-major float
          for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
              h_covs[i * 16 + r * 4 + c] = static_cast<float>(covs[i](r, c));
        }
        d_source_covs_.upload(h_covs.data(), n * 16);
      }

      void setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &covs)
      {
        size_t n = covs.size();
        std::vector<float> h_covs(n * 16);
        for (size_t i = 0; i < n; ++i)
        {
          for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
              h_covs[i * 16 + r * 4 + c] = static_cast<float>(covs[i](r, c));
        }
        d_target_covs_.upload(h_covs.data(), n * 16);
      }

      int updateCorrespondences(const Eigen::Isometry3d &trans, double max_corr_dist)
      {
        if (n_source_ == 0 || n_target_ == 0)
          return 0;

        // Upload transform as row-major float 4x4
        Eigen::Matrix4f T_f = trans.matrix().cast<float>();
        float h_T[16];
        for (int r = 0; r < 4; ++r)
          for (int c = 0; c < 4; ++c)
            h_T[r * 4 + c] = T_f(r, c);
        d_T_.upload(h_T, 16);

        // Ensure temp buffers are allocated (reused across iterations)
        d_transformed_.allocate(n_source_);
        d_nn_idx_.allocate(n_source_);
        d_nn_dist_.allocate(n_source_);

        return findCorrespondencesAndMahalanobis(
            d_source_.data(), n_source_,
            d_source_covs_.data(),
            d_target_covs_.data(),
            d_T_.data(),
            voxel_map_,
            static_cast<float>(max_corr_dist),
            d_correspondences_.data(),
            d_sq_distances_.data(),
            d_mahalanobis_.data(),
            d_transformed_.data(),
            d_nn_idx_.data(),
            d_nn_dist_.data());
      }

      void getCorrespondences(std::vector<int> &out) const
      {
        out.resize(n_source_);
        if (n_source_ > 0)
          d_correspondences_.download(out.data(), n_source_);
      }

      void getSqDistances(std::vector<float> &out) const
      {
        out.resize(n_source_);
        if (n_source_ > 0)
          d_sq_distances_.download(out.data(), n_source_);
      }

      void getMahalanobis(std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &out) const
      {
        out.resize(n_source_);
        if (n_source_ == 0)
          return;

        std::vector<float> h_mah(n_source_ * 16);
        d_mahalanobis_.download(h_mah.data(), n_source_ * 16);

        for (size_t i = 0; i < n_source_; ++i)
        {
          for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
              out[i](r, c) = static_cast<double>(h_mah[i * 16 + r * 4 + c]);
        }
      }

      double linearize(const Eigen::Isometry3d &trans,
                       Eigen::Matrix<double, 6, 6> *H,
                       Eigen::Matrix<double, 6, 1> *b)
      {
        if (n_source_ == 0)
          return 0.0;

        // Upload transform
        Eigen::Matrix4f T_f = trans.matrix().cast<float>();
        float h_T[16];
        for (int r = 0; r < 4; ++r)
          for (int c = 0; c < 4; ++c)
            h_T[r * 4 + c] = T_f(r, c);
        d_T_.upload(h_T, 16);

        // Ensure workspace: N * 43 + 43 (per-point + reduction output)
        size_t ws_size = n_source_ * 43 + 43;
        d_linearize_workspace_.allocate(ws_size);

        double h_H[36], h_b[6];
        double *pH = (H != nullptr) ? h_H : nullptr;
        double *pb = (b != nullptr) ? h_b : nullptr;

        double err = linearizeGpu(
            d_source_.data(), n_source_,
            d_target_.data(),
            d_correspondences_.data(),
            d_mahalanobis_.data(),
            d_T_.data(),
            pH, pb,
            d_linearize_workspace_.data());

        if (H && pH)
        {
          for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c)
              (*H)(r, c) = pH[r * 6 + c];
        }
        if (b && pb)
        {
          for (int j = 0; j < 6; ++j)
            (*b)(j) = pb[j];
        }
        return err;
      }

      double computeError(const Eigen::Isometry3d &trans)
      {
        return linearize(trans, nullptr, nullptr);
      }

      size_t sourceSize() const { return n_source_; }
      size_t targetSize() const { return n_target_; }
      bool isReady() const { return n_source_ > 0 && n_target_ > 0 && !d_source_covs_.empty() && !d_target_covs_.empty(); }

      void clear()
      {
        voxel_map_.clear();
        d_source_.free();
        d_target_.free();
        d_source_covs_.free();
        d_target_covs_.free();
        d_correspondences_.free();
        d_sq_distances_.free();
        d_mahalanobis_.free();
        d_T_.free();
        d_transformed_.free();
        d_nn_idx_.free();
        d_nn_dist_.free();
        d_linearize_workspace_.free();
        n_source_ = 0;
        n_target_ = 0;
      }

    private:
      // Use DeviceBuffer<float4> for points (similar to CudaPointCloud but raw)
      DeviceBuffer<float4> d_source_;
      DeviceBuffer<float4> d_target_;
      DeviceBuffer<float> d_source_covs_; // N * 16
      DeviceBuffer<float> d_target_covs_; // M * 16
      DeviceBuffer<int> d_correspondences_;
      DeviceBuffer<float> d_sq_distances_;
      DeviceBuffer<float> d_mahalanobis_; // N * 16
      DeviceBuffer<float> d_T_;           // 16

      VoxelHashMapGPU voxel_map_;
      float voxel_size_ = 0.f;
      size_t n_source_ = 0;
      size_t n_target_ = 0;

      // Persistent temp buffers (reused across iterations to avoid repeated alloc/free)
      DeviceBuffer<float4> d_transformed_;
      DeviceBuffer<int> d_nn_idx_;
      DeviceBuffer<float> d_nn_dist_;
      DeviceBuffer<double> d_linearize_workspace_;
    };

    // ============================================================
    // GicpGpuHandle (pimpl forwarding)
    // ============================================================

    GicpGpuHandle::GicpGpuHandle() : impl_(std::make_unique<GicpGpuImpl>()) {}
    GicpGpuHandle::~GicpGpuHandle() = default;

    void GicpGpuHandle::setTarget(const std::vector<Eigen::Vector4f> &points, float voxel_size)
    {
      impl_->setTarget(points, voxel_size);
    }
    void GicpGpuHandle::setSource(const std::vector<Eigen::Vector4f> &points)
    {
      impl_->setSource(points);
    }
    void GicpGpuHandle::setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &covs)
    {
      impl_->setSourceCovariances(covs);
    }
    void GicpGpuHandle::setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &covs)
    {
      impl_->setTargetCovariances(covs);
    }
    int GicpGpuHandle::updateCorrespondences(const Eigen::Isometry3d &trans, double max_corr_dist)
    {
      return impl_->updateCorrespondences(trans, max_corr_dist);
    }
    void GicpGpuHandle::getCorrespondences(std::vector<int> &correspondences) const
    {
      impl_->getCorrespondences(correspondences);
    }
    void GicpGpuHandle::getSqDistances(std::vector<float> &sq_distances) const
    {
      impl_->getSqDistances(sq_distances);
    }
    void GicpGpuHandle::getMahalanobis(std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &mahalanobis) const
    {
      impl_->getMahalanobis(mahalanobis);
    }
    double GicpGpuHandle::linearize(const Eigen::Isometry3d &trans,
                                    Eigen::Matrix<double, 6, 6> *H,
                                    Eigen::Matrix<double, 6, 1> *b)
    {
      return impl_->linearize(trans, H, b);
    }
    double GicpGpuHandle::computeError(const Eigen::Isometry3d &trans)
    {
      return impl_->computeError(trans);
    }
    size_t GicpGpuHandle::sourceSize() const { return impl_->sourceSize(); }
    size_t GicpGpuHandle::targetSize() const { return impl_->targetSize(); }
    bool GicpGpuHandle::isReady() const { return impl_->isReady(); }
    void GicpGpuHandle::clear() { impl_->clear(); }

  } // namespace cuda
} // namespace dlio
