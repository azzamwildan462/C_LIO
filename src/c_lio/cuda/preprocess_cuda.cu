/***********************************************************
 *                                                         *
 * GPU Preprocessing: Deskewing + Voxel Grid Filter        *
 *                                                         *
 * Authors: Azzam Wildan M                                 *
 *                                                         *
 ***********************************************************/

#include "c_lio/cuda/preprocess_cuda.cuh"
#include "c_lio/cuda/common.cuh"

namespace c_lio
{
  namespace cuda
  {

    // ═══════════════════════════════════════════════════════════════════════
    // GPU Deskewing
    // ═══════════════════════════════════════════════════════════════════════

    __global__ void deskewKernel(const float4 *__restrict__ in,
                                 float4 *__restrict__ out,
                                 size_t n,
                                 const float *__restrict__ transforms,
                                 const int *__restrict__ frame_indices)
    {
      size_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n)
        return;

      float4 p = in[i];
      int fi = frame_indices[i];

      // Each frame is a row-major 4x4 matrix: transforms[fi*16 .. fi*16+15]
      const float *T = transforms + fi * 16;

      float x = T[0] * p.x + T[1] * p.y + T[2] * p.z + T[3];
      float y = T[4] * p.x + T[5] * p.y + T[6] * p.z + T[7];
      float z = T[8] * p.x + T[9] * p.y + T[10] * p.z + T[11];

      out[i] = make_float4(x, y, z, p.w); // preserve intensity
    }

    void deskewPointCloud(const float4 *d_in, float4 *d_out, size_t n,
                          const float *d_transforms, const int *d_frame_indices,
                          int num_frames, cudaStream_t stream)
    {
      if (n == 0)
        return;
      constexpr int kBlock = 256;
      int grid = (n + kBlock - 1) / kBlock;
      deskewKernel<<<grid, kBlock, 0, stream>>>(d_in, d_out, n,
                                                d_transforms, d_frame_indices);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GPU Voxel Grid Filter (centroid-based, matching PCL VoxelGrid)
    // ═══════════════════════════════════════════════════════════════════════
    //
    // Algorithm:
    //   1. Pack voxel (vx,vy,vz) into a single uint64_t key for race-free CAS
    //   2. Insert into hash table: atomicCAS on key, then atomicAdd on sums
    //   3. Emit centroids: scan table, output centroid = sum / count
    //
    // Hash table size: next power-of-2 >= 2*N, capped at 4M entries.

    static constexpr uint32_t VOXEL_HASH_MAX = 1u << 22; // 4M max buckets
    static constexpr uint64_t VOXEL_KEY_EMPTY = 0xFFFFFFFFFFFFFFFFull;
    static constexpr int VOXEL_MAX_PROBE = 64;

    // 21 bits per axis → ±1M voxels per axis (at 0.05m → ±50km range)
    static constexpr int VOXEL_COORD_OFFSET = (1 << 20); // 1048576

    struct VoxelEntry
    {
      unsigned long long key; // packed (vx,vy,vz) — use ULL for atomicCAS
      int count;
      float sum_x, sum_y, sum_z, sum_w;
    };

    __device__ __host__ inline unsigned long long packVoxelKey(int vx, int vy, int vz)
    {
      unsigned long long kx = static_cast<unsigned long long>(vx + VOXEL_COORD_OFFSET) & 0x1FFFFF;
      unsigned long long ky = static_cast<unsigned long long>(vy + VOXEL_COORD_OFFSET) & 0x1FFFFF;
      unsigned long long kz = static_cast<unsigned long long>(vz + VOXEL_COORD_OFFSET) & 0x1FFFFF;
      return (kx << 42) | (ky << 21) | kz;
    }

    __host__ __device__ inline uint32_t hashKey64(unsigned long long key, uint32_t mask)
    {
      // FNV-1a on the 8 bytes
      uint32_t h = 2166136261u;
      for (int b = 0; b < 8; b++)
      {
        h ^= static_cast<uint32_t>((key >> (b * 8)) & 0xFF);
        h *= 16777619u;
      }
      return h & mask;
    }

    // Init kernel
    __global__ void voxelInitKernel(VoxelEntry *__restrict__ table, uint32_t n)
    {
      size_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n)
        return;
      table[i].key = VOXEL_KEY_EMPTY;
      table[i].count = 0;
      table[i].sum_x = 0.f;
      table[i].sum_y = 0.f;
      table[i].sum_z = 0.f;
      table[i].sum_w = 0.f;
    }

    // Phase 1: Insert + accumulate centroid sums (race-free via 64-bit CAS on key)
    __global__ void voxelAccumKernel(const float4 *__restrict__ points,
                                     size_t n,
                                     VoxelEntry *__restrict__ table,
                                     uint32_t table_size,
                                     float inv_voxel)
    {
      size_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n)
        return;

      float4 p = points[i];
      int vx = static_cast<int>(floorf(p.x * inv_voxel));
      int vy = static_cast<int>(floorf(p.y * inv_voxel));
      int vz = static_cast<int>(floorf(p.z * inv_voxel));
      unsigned long long key = packVoxelKey(vx, vy, vz);

      uint32_t mask = table_size - 1;
      uint32_t h = hashKey64(key, mask);

      for (int probe = 0; probe < VOXEL_MAX_PROBE; probe++)
      {
        uint32_t slot = (h + probe) & mask;
        VoxelEntry *entry = &table[slot];

        // Single 64-bit CAS — no multi-field race
        unsigned long long old_key = atomicCAS(&entry->key, VOXEL_KEY_EMPTY, key);

        if (old_key == VOXEL_KEY_EMPTY || old_key == key)
        {
          // Claimed or same voxel — accumulate
          atomicAdd(&entry->sum_x, p.x);
          atomicAdd(&entry->sum_y, p.y);
          atomicAdd(&entry->sum_z, p.z);
          atomicAdd(&entry->sum_w, p.w);
          atomicAdd(&entry->count, 1);
          return;
        }
        // Different voxel — probe
      }
    }

    // Phase 2: Emit centroids from occupied entries
    __global__ void voxelEmitKernel(const VoxelEntry *__restrict__ table,
                                    uint32_t table_size,
                                    float4 *__restrict__ out,
                                    int *__restrict__ out_counter)
    {
      size_t slot = blockIdx.x * blockDim.x + threadIdx.x;
      if (slot >= table_size)
        return;

      const VoxelEntry &e = table[slot];
      if (e.key == VOXEL_KEY_EMPTY || e.count <= 0)
        return;

      float inv_n = 1.0f / static_cast<float>(e.count);
      int idx = atomicAdd(out_counter, 1);
      out[idx] = make_float4(e.sum_x * inv_n, e.sum_y * inv_n,
                              e.sum_z * inv_n, e.sum_w * inv_n);
    }

    // Host function
    size_t voxelGridFilter(const float4 *d_in, size_t n,
                           float4 *d_out, int *d_out_count,
                           float voxel_size, cudaStream_t stream)
    {
      if (n == 0)
        return 0;

      float inv_voxel = 1.0f / voxel_size;

      // Choose table size: next power of 2 >= 2*n, capped
      uint32_t table_size = 1;
      while (table_size < 2 * n && table_size < VOXEL_HASH_MAX)
        table_size <<= 1;
      if (table_size < 2 * n)
        table_size = VOXEL_HASH_MAX;

      // Init hash table
      DeviceBuffer<VoxelEntry> d_table(table_size);
      {
        constexpr int kBlock = 256;
        int grid = (table_size + kBlock - 1) / kBlock;
        voxelInitKernel<<<grid, kBlock, 0, stream>>>(d_table.data(), table_size);
      }

      // Phase 1: Accumulate
      {
        constexpr int kBlock = 256;
        int grid = (n + kBlock - 1) / kBlock;
        voxelAccumKernel<<<grid, kBlock, 0, stream>>>(
            d_in, n, d_table.data(), table_size, inv_voxel);
      }

      // Phase 2: Emit centroids
      DeviceBuffer<int> d_counter(1);
      CUDA_CHECK(cudaMemsetAsync(d_counter.data(), 0, sizeof(int), stream));

      {
        constexpr int kBlock = 256;
        int grid = (table_size + kBlock - 1) / kBlock;
        voxelEmitKernel<<<grid, kBlock, 0, stream>>>(
            d_table.data(), table_size, d_out, d_counter.data());
      }

      // Read output count
      int total_out = 0;
      CUDA_CHECK(cudaMemcpyAsync(&total_out, d_counter.data(), sizeof(int),
                                  cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      if (d_out_count)
      {
        CUDA_CHECK(cudaMemcpyAsync(d_out_count, &total_out, sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
      }

      return static_cast<size_t>(total_out);
    }

  } // namespace cuda
} // namespace c_lio
