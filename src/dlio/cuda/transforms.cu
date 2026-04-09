#include "dlio/cuda/transforms.cuh"
#include "dlio/cuda/common.cuh"

namespace dlio
{
  namespace cuda
  {

    __global__ void transformPointCloudKernel(const float4 *__restrict__ in,
                                              float4 *__restrict__ out,
                                              size_t n,
                                              const float *__restrict__ T)
    {
      size_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n)
        return;

      float4 p = in[i];

      // T is row-major 4x4:
      //   T[0]  T[1]  T[2]  T[3]
      //   T[4]  T[5]  T[6]  T[7]
      //   T[8]  T[9]  T[10] T[11]
      //   T[12] T[13] T[14] T[15]
      float x = T[0] * p.x + T[1] * p.y + T[2] * p.z + T[3];
      float y = T[4] * p.x + T[5] * p.y + T[6] * p.z + T[7];
      float z = T[8] * p.x + T[9] * p.y + T[10] * p.z + T[11];

      out[i] = make_float4(x, y, z, p.w); // preserve intensity in w
    }

    void transformPointCloud(const float4 *d_in, float4 *d_out, size_t n,
                             const float *d_T_4x4, cudaStream_t stream)
    {
      if (n == 0)
        return;
      constexpr int kBlockSize = 256;
      int grid = (n + kBlockSize - 1) / kBlockSize;
      transformPointCloudKernel<<<grid, kBlockSize, 0, stream>>>(d_in, d_out, n, d_T_4x4);
    }

    void transformPointCloudSync(float4 *d_points, size_t n,
                                 const float T_4x4[16])
    {
      if (n == 0)
        return;

      DeviceBuffer<float> d_T;
      d_T.upload(T_4x4, 16);

      transformPointCloud(d_points, d_points, n, d_T.data());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

  } // namespace cuda
} // namespace dlio
