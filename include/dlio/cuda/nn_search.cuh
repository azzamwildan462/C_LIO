#ifndef DLIO_CUDA_NN_SEARCH_CUH
#define DLIO_CUDA_NN_SEARCH_CUH

#include "dlio/cuda/common.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace dlio
{
  namespace cuda
  {

    // Voxel hash map for GPU nearest neighbor search.
    // Target points are inserted into a hash map keyed by voxel coordinates.
    // Query points find their nearest neighbor by checking the 27 neighboring voxels.
    class VoxelHashMapGPU
    {
    public:
      VoxelHashMapGPU() = default;
      ~VoxelHashMapGPU();

      // Build hash map from target points on device.
      // voxel_size: side length of each voxel (meters)
      void build(const float4 *d_target, size_t n_target, float voxel_size);

      // Find nearest neighbor for each query point.
      // d_query:   [n_query] device array of query points (float4: x,y,z,w)
      // d_indices: [n_query] output — index of closest target point (-1 if none found)
      // d_sq_dist: [n_query] output — squared distance to closest target point
      // max_dist:  max correspondence distance (points farther are set to -1)
      void search(const float4 *d_query, size_t n_query,
                  int *d_indices, float *d_sq_dist,
                  float max_dist, cudaStream_t stream = 0) const;

      void clear();

      size_t numPoints() const { return n_target_; }

      // Hash table entry with voxel coords for open addressing
      struct HashEntry
      {
        int32_t start; // index into d_sorted_points_
        int32_t count; // number of points in this voxel
        int vx, vy, vz; // voxel coordinates for collision verification
      };

      static constexpr size_t HASH_TABLE_SIZE = 1 << 20; // 1M buckets
      static constexpr int MAX_PROBE = 32; // max linear probing steps

      float voxel_size_ = 0.f;
      size_t n_target_ = 0;

      // Device arrays
      float4 *d_sorted_points_ = nullptr;  // target points sorted by voxel
      int32_t *d_voxel_keys_ = nullptr;    // hash key per point (for sorting)
      HashEntry *d_hash_table_ = nullptr;  // hash table (fixed size)
      int32_t *d_point_indices_ = nullptr; // original index per sorted point
    };

    // Find correspondences + compute Mahalanobis matrices in one kernel.
    // This combines update_correspondences() functionality:
    //   1. Transform source by T
    //   2. NN search against target via voxel hash
    //   3. Compute Mahalanobis inverse: (cov_B + T * cov_A * T^T)^{-1}
    //
    // d_source:        [N] source points (float4)
    // d_source_covs:   [N * 16] source covariance matrices (4x4, row-major, double→float)
    // d_target_covs:   [M * 16] target covariance matrices
    // d_T_4x4:         [16] transform matrix (row-major float)
    // voxel_map:        built voxel hash map of target
    // max_corr_dist:    correspondence distance threshold
    //
    // Outputs:
    // d_correspondences: [N] matched target index (-1 if no match)
    // d_sq_distances:    [N] squared distance
    // d_mahalanobis:     [N * 16] Mahalanobis inverse matrices (4x4 row-major float)
    // Returns number of valid correspondences (host value).
    // Temp buffers must be pre-allocated by caller (size >= n_source).
    // d_transformed: [N] float4 workspace
    // d_nn_idx:      [N] int workspace
    // d_nn_dist:     [N] float workspace
    int findCorrespondencesAndMahalanobis(
        const float4 *d_source, size_t n_source,
        const float *d_source_covs,
        const float *d_target_covs,
        const float *d_T_4x4,
        const VoxelHashMapGPU &voxel_map,
        float max_corr_dist,
        int *d_correspondences,
        float *d_sq_distances,
        float *d_mahalanobis,
        float4 *d_transformed,
        int *d_nn_idx,
        float *d_nn_dist,
        cudaStream_t stream = 0);

  } // namespace cuda
} // namespace dlio

#endif // DLIO_CUDA_NN_SEARCH_CUH
