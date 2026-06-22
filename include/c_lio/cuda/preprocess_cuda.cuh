/***********************************************************
 *                                                         *
 * GPU Preprocessing: Deskewing + Voxel Grid Filter        *
 *                                                         *
 * Deskew: per-point SE(3) transform from IMU integration  *
 * Voxel:  hash-based voxel grid downsample                *
 *                                                         *
 * Authors: Azzam Wildan M                                 *
 *                                                         *
 ***********************************************************/

#ifndef C_LIO_CUDA_PREPROCESS_CUDA_CUH
#define C_LIO_CUDA_PREPROCESS_CUDA_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace c_lio
{
  namespace cuda
  {

    // ── GPU Deskewing ──────────────────────────────────────────────────────
    //
    // Each point belongs to a timestamp group. Each group has its own SE(3)
    // transform (computed on CPU from IMU integration). The GPU applies
    // the correct transform per point in parallel.
    //
    // d_in/d_out:       [N] float4 (x,y,z,intensity)
    // d_transforms:     [num_frames * 16] row-major 4x4 matrices, flat
    // d_frame_indices:  [N] per-point frame index (0..num_frames-1)
    // Can operate in-place (d_out == d_in).

    void deskewPointCloud(const float4 *d_in, float4 *d_out, size_t n,
                          const float *d_transforms, const int *d_frame_indices,
                          int num_frames, cudaStream_t stream = 0);

    // ── GPU Voxel Grid Filter ──────────────────────────────────────────────
    //
    // Hash-based voxel grid downsample. For each occupied voxel, keeps the
    // point closest to the voxel centroid (or the first point, depending on
    // mode). Returns compacted output.
    //
    // d_in:        [N] float4 input points
    // d_out:       [N] float4 output buffer (at most N points)
    // d_out_count: [1] output count (device int, caller allocates)
    // voxel_size:  grid resolution in meters
    // Returns: number of output points (also written to d_out_count)

    size_t voxelGridFilter(const float4 *d_in, size_t n,
                           float4 *d_out, int *d_out_count,
                           float voxel_size, cudaStream_t stream = 0);

  } // namespace cuda
} // namespace c_lio

#endif // C_LIO_CUDA_PREPROCESS_CUDA_CUH
