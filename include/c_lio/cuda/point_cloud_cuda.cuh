#ifndef C_LIO_CUDA_POINT_CLOUD_CUDA_CUH
#define C_LIO_CUDA_POINT_CLOUD_CUDA_CUH

#include "c_lio/cuda/common.cuh"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <vector>

namespace c_lio
{
  namespace cuda
  {

    // GPU point cloud: stores points as float4 (x, y, z, intensity) on device
    class CudaPointCloud
    {
    public:
      CudaPointCloud() = default;

      // Upload from PCL cloud
      template <typename PointT>
      void upload(const pcl::PointCloud<PointT> &cloud)
      {
        size_t n = cloud.size();
        if (n == 0)
        {
          points_.free();
          return;
        }

        std::vector<float4> host(n);
        for (size_t i = 0; i < n; ++i)
        {
          host[i] = make_float4(cloud.points[i].x, cloud.points[i].y,
                                cloud.points[i].z, cloud.points[i].intensity);
        }
        points_.upload(host.data(), n);
      }

      // Download to PCL cloud
      template <typename PointT>
      void download(pcl::PointCloud<PointT> &cloud) const
      {
        size_t n = points_.size();
        cloud.resize(n);
        if (n == 0)
          return;

        std::vector<float4> host(n);
        points_.download(host.data(), n);
        for (size_t i = 0; i < n; ++i)
        {
          cloud.points[i].x = host[i].x;
          cloud.points[i].y = host[i].y;
          cloud.points[i].z = host[i].z;
          cloud.points[i].intensity = host[i].w;
        }
        cloud.width = n;
        cloud.height = 1;
        cloud.is_dense = true;
      }

      float4 *data() { return points_.data(); }
      const float4 *data() const { return points_.data(); }
      size_t size() const { return points_.size(); }
      bool empty() const { return points_.empty(); }

      void allocate(size_t n) { points_.allocate(n); }

    private:
      DeviceBuffer<float4> points_;
    };

  } // namespace cuda
} // namespace c_lio

#endif // C_LIO_CUDA_POINT_CLOUD_CUDA_CUH
