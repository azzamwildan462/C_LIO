#include "dlio/cuda/linearize.cuh"
#include "dlio/cuda/common.cuh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cstring>
#include <vector>

namespace dlio
{
  namespace cuda
  {

    // Per-point: compute error, Jacobian, H_i (6x6), b_i (6x1), error_i (1)
    // Output: d_out[i * 43 + 0..35] = H_i (row-major 6x6)
    //         d_out[i * 43 + 36..41] = b_i (6x1)
    //         d_out[i * 43 + 42]     = error_i
    static constexpr int WORK_PER_POINT = 43; // 36 + 6 + 1

    __global__ void linearizeKernel(
        const float4 *__restrict__ source,
        const float4 *__restrict__ target,
        const int *__restrict__ correspondences,
        const float *__restrict__ mahalanobis, // N * 16, row-major 4x4 float
        const float *__restrict__ T,           // 16, row-major 4x4 float
        size_t n_source,
        double *__restrict__ out, // N * 43
        bool compute_hb)
    {
      size_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n_source)
        return;

      double *my_out = out + i * WORK_PER_POINT;

      // Zero output
      for (int j = 0; j < WORK_PER_POINT; ++j)
        my_out[j] = 0.0;

      int tidx = correspondences[i];
      if (tidx < 0)
        return;

      // Source point
      float4 sp = source[i];
      double ax = sp.x, ay = sp.y, az = sp.z;

      // Target point
      float4 tp = target[tidx];
      double bx = tp.x, by = tp.y, bz = tp.z;

      // Transform source: transed = T * source
      double tx = T[0] * ax + T[1] * ay + T[2] * az + T[3];
      double ty = T[4] * ax + T[5] * ay + T[6] * az + T[7];
      double tz = T[8] * ax + T[9] * ay + T[10] * az + T[11];

      // Error vector (4D, w=0)
      double ex = bx - tx;
      double ey = by - ty;
      double ez = bz - tz;

      // Mahalanobis matrix M (4x4, row-major float → double)
      const float *Mi = &mahalanobis[i * 16];
      // Only use 3x3 upper-left block (M[3][3]=0 by convention)
      double m00 = Mi[0], m01 = Mi[1], m02 = Mi[2];
      double m10 = Mi[4], m11 = Mi[5], m12 = Mi[6];
      double m20 = Mi[8], m21 = Mi[9], m22 = Mi[10];

      // Weighted error: e^T * M * e (3x3 block)
      double Me_x = m00 * ex + m01 * ey + m02 * ez;
      double Me_y = m10 * ex + m11 * ey + m12 * ez;
      double Me_z = m20 * ex + m21 * ey + m22 * ez;
      double error_val = ex * Me_x + ey * Me_y + ez * Me_z;
      my_out[42] = error_val;

      if (!compute_hb)
        return;

      // Jacobian dtdx0 (3x6):
      // dtdx0 = [skew(transed) | -I]
      // skew(v) = [  0  -vz  vy ]
      //           [  vz  0  -vx ]
      //           [ -vy  vx  0  ]
      // So dtdx0:
      //   row0: [  0   -tz   ty  | -1   0   0 ]
      //   row1: [  tz   0   -tx  |  0  -1   0 ]
      //   row2: [ -ty   tx   0   |  0   0  -1 ]
      double J[3][6] = {
          {0.0, -tz, ty, -1.0, 0.0, 0.0},
          {tz, 0.0, -tx, 0.0, -1.0, 0.0},
          {-ty, tx, 0.0, 0.0, 0.0, -1.0}};

      // H_i = J^T * M * J  (6x6)
      // b_i = J^T * M * e  (6x1)

      // Compute M * J (3x6)
      double MJ[3][6];
      for (int c = 0; c < 6; ++c)
      {
        MJ[0][c] = m00 * J[0][c] + m01 * J[1][c] + m02 * J[2][c];
        MJ[1][c] = m10 * J[0][c] + m11 * J[1][c] + m12 * J[2][c];
        MJ[2][c] = m20 * J[0][c] + m21 * J[1][c] + m22 * J[2][c];
      }

      // H_i = J^T * (M*J) → 6x6
      for (int r = 0; r < 6; ++r)
      {
        for (int c = 0; c < 6; ++c)
        {
          double val = 0.0;
          for (int k = 0; k < 3; ++k)
          {
            val += J[k][r] * MJ[k][c];
          }
          my_out[r * 6 + c] = val;
        }
      }

      // b_i = J^T * M * e → 6x1
      for (int r = 0; r < 6; ++r)
      {
        double val = 0.0;
        for (int k = 0; k < 3; ++k)
        {
          val += J[k][r] * (k == 0 ? Me_x : (k == 1 ? Me_y : Me_z));
        }
        my_out[36 + r] = val;
      }
    }

    // Reduction kernel: sum N arrays of 43 doubles into one
    // Uses block-level shared memory reduction
    __global__ void reduceKernel(const double *__restrict__ in, double *__restrict__ out,
                                 size_t n_points, int stride)
    {
      // Each thread handles one component (0..42)
      int comp = threadIdx.x; // 0..42
      if (comp >= WORK_PER_POINT)
        return;

      double sum = 0.0;
      for (size_t i = blockIdx.x; i < n_points; i += gridDim.x)
      {
        sum += in[i * stride + comp];
      }

      // Atomic add to output (43 components)
      atomicAdd(&out[comp], sum);
    }

    // Better reduction: each block reduces a chunk of points
    __global__ void reduceChunkKernel(const double *__restrict__ in, double *__restrict__ out,
                                      size_t n_points)
    {
      extern __shared__ double sdata[];

      int comp = threadIdx.x % WORK_PER_POINT;
      int lane = threadIdx.x / WORK_PER_POINT;
      int lanes_per_block = blockDim.x / WORK_PER_POINT;

      double sum = 0.0;
      for (size_t i = blockIdx.x * lanes_per_block + lane; i < n_points; i += gridDim.x * lanes_per_block)
      {
        sum += in[i * WORK_PER_POINT + comp];
      }

      sdata[threadIdx.x] = sum;
      __syncthreads();

      // Reduce within shared memory
      for (int s = lanes_per_block / 2; s > 0; s >>= 1)
      {
        if (lane < s)
        {
          sdata[lane * WORK_PER_POINT + comp] += sdata[(lane + s) * WORK_PER_POINT + comp];
        }
        __syncthreads();
      }

      if (lane == 0)
      {
        atomicAdd(&out[comp], sdata[comp]);
      }
    }

    double linearizeGpu(
        const float4 *d_source, size_t n_source,
        const float4 *d_target,
        const int *d_correspondences,
        const float *d_mahalanobis,
        const float *d_T_4x4,
        double *h_H,
        double *h_b,
        double *d_workspace,
        cudaStream_t stream)
    {
      if (n_source == 0)
        return 0.0;

      bool compute_hb = (h_H != nullptr && h_b != nullptr);

      // Step 1: per-point linearize
      constexpr int kBlock = 256;
      int grid = (n_source + kBlock - 1) / kBlock;
      linearizeKernel<<<grid, kBlock, 0, stream>>>(
          d_source, d_target, d_correspondences, d_mahalanobis, d_T_4x4,
          n_source, d_workspace, compute_hb);

      // Step 2: reduce 43 components across N points
      // Use a small output buffer (43 doubles) on device
      // d_workspace layout: [N * 43 per-point data] [43 reduction output]
      double *d_reduced = d_workspace + n_source * WORK_PER_POINT;
      CUDA_CHECK(cudaMemsetAsync(d_reduced, 0, WORK_PER_POINT * sizeof(double), stream));

      // Launch reduction: 256 threads per block, each block handles multiple points
      int reduce_blocks = std::min((int)((n_source + 5) / 6), 256);
      int threads_per_block = 256; // must be multiple of WORK_PER_POINT(43)... not ideal
      // Simpler approach: use a serial-ish reduction for 43 components
      // Each block iterates over all points for its assigned components
      reduceKernel<<<WORK_PER_POINT, 1, 0, stream>>>(d_workspace, d_reduced, n_source, WORK_PER_POINT);

      // Download result
      double h_reduced[WORK_PER_POINT];
      CUDA_CHECK(cudaMemcpyAsync(h_reduced, d_reduced, WORK_PER_POINT * sizeof(double),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // Unpack
      if (compute_hb)
      {
        for (int r = 0; r < 6; ++r)
          for (int c = 0; c < 6; ++c)
            h_H[r * 6 + c] = h_reduced[r * 6 + c];
        for (int j = 0; j < 6; ++j)
          h_b[j] = h_reduced[36 + j];
      }

      return h_reduced[42];
    }

    double computeErrorGpu(
        const float4 *d_source, size_t n_source,
        const float4 *d_target,
        const int *d_correspondences,
        const float *d_mahalanobis,
        const float *d_T_4x4,
        double *d_workspace,
        cudaStream_t stream)
    {
      return linearizeGpu(d_source, n_source, d_target, d_correspondences,
                          d_mahalanobis, d_T_4x4, nullptr, nullptr, d_workspace, stream);
    }

  } // namespace cuda
} // namespace dlio
