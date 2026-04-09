#include "dlio/cuda/nn_search.cuh"
#include "dlio/cuda/transforms.cuh"
#include "dlio/cuda/common.cuh"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cstring>
#include <vector>
#include <cmath>

namespace dlio {
namespace cuda {

// ============================================================
// Hash helpers
// ============================================================

__host__ __device__ inline int3 pointToVoxel(float x, float y, float z, float inv_voxel) {
  return make_int3(static_cast<int>(floorf(x * inv_voxel)),
                   static_cast<int>(floorf(y * inv_voxel)),
                   static_cast<int>(floorf(z * inv_voxel)));
}

__host__ __device__ inline uint32_t hashVoxel(int3 v, uint32_t table_size) {
  uint32_t h = 2166136261u;
  h ^= static_cast<uint32_t>(v.x); h *= 16777619u;
  h ^= static_cast<uint32_t>(v.y); h *= 16777619u;
  h ^= static_cast<uint32_t>(v.z); h *= 16777619u;
  return h & (table_size - 1);
}

// Open addressing lookup: find hash entry matching voxel coords
__host__ __device__ inline const VoxelHashMapGPU::HashEntry* lookupEntry(
    const VoxelHashMapGPU::HashEntry* table, uint32_t table_size,
    int3 v, int max_probe) {
  uint32_t h = hashVoxel(v, table_size);
  for (int p = 0; p < max_probe; ++p) {
    uint32_t idx = (h + p) & (table_size - 1);
    const VoxelHashMapGPU::HashEntry& e = table[idx];
    if (e.count <= 0) return nullptr; // empty slot
    if (e.vx == v.x && e.vy == v.y && e.vz == v.z) return &table[idx]; // found
  }
  return nullptr;
}

// ============================================================
// Build hash map
// ============================================================

__global__ void computeVoxelKeysKernel(const float4* __restrict__ points,
                                        int32_t* __restrict__ keys,
                                        size_t n, float inv_voxel, uint32_t table_size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 pt = points[i];
  int3 v = pointToVoxel(pt.x, pt.y, pt.z, inv_voxel);
  keys[i] = static_cast<int32_t>(hashVoxel(v, table_size));
}

__global__ void gatherPointsKernel(const float4* __restrict__ src,
                                    const int32_t* __restrict__ idx,
                                    float4* __restrict__ dst, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = src[idx[i]];
}

void VoxelHashMapGPU::build(const float4* d_target, size_t n_target, float voxel_size) {
  clear();
  if (n_target == 0) return;

  voxel_size_ = voxel_size;
  n_target_ = n_target;
  float inv_voxel = 1.0f / voxel_size;

  // Allocate
  CUDA_CHECK(cudaMalloc(&d_sorted_points_, n_target * sizeof(float4)));
  CUDA_CHECK(cudaMalloc(&d_voxel_keys_, n_target * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_point_indices_, n_target * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_hash_table_, HASH_TABLE_SIZE * sizeof(HashEntry)));

  // Initialize hash table to empty (count=0)
  std::vector<HashEntry> h_empty(HASH_TABLE_SIZE);
  for (auto& e : h_empty) { e.start = -1; e.count = 0; e.vx = 0; e.vy = 0; e.vz = 0; }
  CUDA_CHECK(cudaMemcpy(d_hash_table_, h_empty.data(), HASH_TABLE_SIZE * sizeof(HashEntry), cudaMemcpyHostToDevice));

  // Compute voxel keys
  constexpr int kBlock = 256;
  int grid = (n_target + kBlock - 1) / kBlock;
  computeVoxelKeysKernel<<<grid, kBlock>>>(d_target, d_voxel_keys_, n_target, inv_voxel, HASH_TABLE_SIZE);

  // Create and sort index array
  std::vector<int32_t> h_indices(n_target);
  for (size_t i = 0; i < n_target; ++i) h_indices[i] = static_cast<int32_t>(i);
  CUDA_CHECK(cudaMemcpy(d_point_indices_, h_indices.data(), n_target * sizeof(int32_t), cudaMemcpyHostToDevice));

  thrust::device_ptr<int32_t> keys_ptr(d_voxel_keys_);
  thrust::device_ptr<int32_t> idx_ptr(d_point_indices_);
  thrust::sort_by_key(thrust::device, keys_ptr, keys_ptr + n_target, idx_ptr);

  // Gather sorted points
  float4* d_temp;
  CUDA_CHECK(cudaMalloc(&d_temp, n_target * sizeof(float4)));
  CUDA_CHECK(cudaMemcpy(d_temp, d_target, n_target * sizeof(float4), cudaMemcpyDeviceToDevice));
  gatherPointsKernel<<<grid, kBlock>>>(d_temp, d_point_indices_, d_sorted_points_, n_target);
  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Download to host for hash table construction (with proper collision handling)
  std::vector<int32_t> h_sorted_keys(n_target);
  std::vector<int32_t> h_sorted_indices(n_target);
  std::vector<float4> h_target(n_target);
  CUDA_CHECK(cudaMemcpy(h_sorted_keys.data(), d_voxel_keys_, n_target * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_sorted_indices.data(), d_point_indices_, n_target * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_target.data(), d_target, n_target * sizeof(float4), cudaMemcpyDeviceToHost));

  // Build hash table on CPU with open addressing (handles collisions correctly)
  std::vector<HashEntry> h_hash(HASH_TABLE_SIZE);
  for (auto& e : h_hash) { e.start = -1; e.count = 0; e.vx = 0; e.vy = 0; e.vz = 0; }

  size_t i = 0;
  while (i < n_target) {
    int32_t key = h_sorted_keys[i];
    size_t start = i;
    while (i < n_target && h_sorted_keys[i] == key) ++i;

    // Sub-group by actual voxel coords (same hash key may have different voxels)
    struct VoxelBucket { int3 v; int32_t start; int32_t count; };
    std::vector<VoxelBucket> buckets;

    for (size_t j = start; j < i; ++j) {
      int orig_idx = h_sorted_indices[j];
      int3 v = pointToVoxel(h_target[orig_idx].x, h_target[orig_idx].y, h_target[orig_idx].z, inv_voxel);
      bool found = false;
      for (auto& b : buckets) {
        if (b.v.x == v.x && b.v.y == v.y && b.v.z == v.z) { b.count++; found = true; break; }
      }
      if (!found) {
        buckets.push_back({v, static_cast<int32_t>(j), 1});
      }
    }

    // Fix start/count: start is the first occurrence in sorted array
    for (auto& b : buckets) {
      // Find actual start in the sorted range
      b.start = -1;
      b.count = 0;
      for (size_t j = start; j < i; ++j) {
        int orig_idx = h_sorted_indices[j];
        int3 v = pointToVoxel(h_target[orig_idx].x, h_target[orig_idx].y, h_target[orig_idx].z, inv_voxel);
        if (v.x == b.v.x && v.y == b.v.y && v.z == b.v.z) {
          if (b.start < 0) b.start = static_cast<int32_t>(j);
          b.count++;
        }
      }
    }

    // Insert each bucket with open addressing
    for (const auto& b : buckets) {
      uint32_t h = hashVoxel(b.v, HASH_TABLE_SIZE);
      for (int p = 0; p < MAX_PROBE; ++p) {
        uint32_t slot = (h + p) & (HASH_TABLE_SIZE - 1);
        if (h_hash[slot].count <= 0) {
          h_hash[slot].start = b.start;
          h_hash[slot].count = b.count;
          h_hash[slot].vx = b.v.x;
          h_hash[slot].vy = b.v.y;
          h_hash[slot].vz = b.v.z;
          break;
        }
      }
    }
  }

  CUDA_CHECK(cudaMemcpy(d_hash_table_, h_hash.data(), HASH_TABLE_SIZE * sizeof(HashEntry), cudaMemcpyHostToDevice));
}

void VoxelHashMapGPU::clear() {
  if (d_sorted_points_) { cudaFree(d_sorted_points_); d_sorted_points_ = nullptr; }
  if (d_voxel_keys_) { cudaFree(d_voxel_keys_); d_voxel_keys_ = nullptr; }
  if (d_point_indices_) { cudaFree(d_point_indices_); d_point_indices_ = nullptr; }
  if (d_hash_table_) { cudaFree(d_hash_table_); d_hash_table_ = nullptr; }
  n_target_ = 0;
}

VoxelHashMapGPU::~VoxelHashMapGPU() { clear(); }

// ============================================================
// NN Search kernel (with open addressing)
// ============================================================

__constant__ int3 c_offsets[27] = {
  {0,0,0}, {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1},
  {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0},
  {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1},
  {0,1,1}, {0,1,-1}, {0,-1,1}, {0,-1,-1},
  {1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1},
  {-1,1,1}, {-1,1,-1}, {-1,-1,1}, {-1,-1,-1}
};

__global__ void nnSearchKernel(const float4* __restrict__ query,
                                const float4* __restrict__ sorted_target,
                                const VoxelHashMapGPU::HashEntry* __restrict__ table,
                                size_t n_query, float inv_voxel, uint32_t table_size,
                                int max_probe, float max_dist_sq,
                                int* __restrict__ out_indices,
                                float* __restrict__ out_sq_dist) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_query) return;

  float4 q = query[i];
  int3 qv = pointToVoxel(q.x, q.y, q.z, inv_voxel);

  float best_dist = max_dist_sq;
  int best_idx = -1;

  for (int n = 0; n < 27; ++n) {
    int3 v = make_int3(qv.x + c_offsets[n].x, qv.y + c_offsets[n].y, qv.z + c_offsets[n].z);

    // Open addressing lookup
    uint32_t h = hashVoxel(v, table_size);
    const VoxelHashMapGPU::HashEntry* entry = nullptr;
    for (int p = 0; p < max_probe; ++p) {
      uint32_t idx = (h + p) & (table_size - 1);
      if (table[idx].count <= 0) break; // empty slot
      if (table[idx].vx == v.x && table[idx].vy == v.y && table[idx].vz == v.z) {
        entry = &table[idx]; break;
      }
    }
    if (!entry) continue;

    for (int j = 0; j < entry->count; ++j) {
      float4 t = sorted_target[entry->start + j];
      float dx = q.x - t.x, dy = q.y - t.y, dz = q.z - t.z;
      float d2 = dx*dx + dy*dy + dz*dz;
      if (d2 < best_dist) { best_dist = d2; best_idx = entry->start + j; }
    }
  }

  out_indices[i] = best_idx;
  out_sq_dist[i] = best_dist;
}

void VoxelHashMapGPU::search(const float4* d_query, size_t n_query,
                              int* d_indices, float* d_sq_dist,
                              float max_dist, cudaStream_t stream) const {
  if (n_query == 0 || n_target_ == 0) return;
  constexpr int kBlock = 256;
  int grid = (n_query + kBlock - 1) / kBlock;
  float inv_voxel = 1.0f / voxel_size_;
  nnSearchKernel<<<grid, kBlock, 0, stream>>>(
      d_query, d_sorted_points_, d_hash_table_,
      n_query, inv_voxel, HASH_TABLE_SIZE, MAX_PROBE, max_dist * max_dist,
      d_indices, d_sq_dist);
}

// ============================================================
// Combined: correspondence + Mahalanobis
// ============================================================

__device__ void invert3x3(const float* __restrict__ M, float* __restrict__ Minv) {
  float a=M[0],b=M[1],c=M[2];
  float d=M[4],e=M[5],f=M[6];
  float g=M[8],h=M[9],k=M[10];
  float det = a*(e*k-f*h) - b*(d*k-f*g) + c*(d*h-e*g);
  if (fabsf(det) < 1e-12f) { for(int j=0;j<16;++j)Minv[j]=0.f; return; }
  float inv_det = 1.0f / det;
  for(int j=0;j<16;++j) Minv[j]=0.f;
  Minv[0]=(e*k-f*h)*inv_det; Minv[1]=(c*h-b*k)*inv_det; Minv[2]=(b*f-c*e)*inv_det;
  Minv[4]=(f*g-d*k)*inv_det; Minv[5]=(a*k-c*g)*inv_det; Minv[6]=(c*d-a*f)*inv_det;
  Minv[8]=(d*h-e*g)*inv_det; Minv[9]=(b*g-a*h)*inv_det; Minv[10]=(a*e-b*d)*inv_det;
}

__global__ void correspondenceAndMahalanobisKernel(
    const float4* __restrict__ source,
    const float* __restrict__ source_covs,
    const float* __restrict__ target_covs,
    const float* __restrict__ T,
    const float4* __restrict__ sorted_target,
    const int* __restrict__ nn_indices,
    const float* __restrict__ nn_sq_dist,
    size_t n_source,
    const int32_t* __restrict__ point_indices,
    float max_dist_sq,
    int* __restrict__ correspondences,
    float* __restrict__ sq_distances,
    float* __restrict__ mahalanobis) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_source) return;

  int sorted_idx = nn_indices[i];
  float dist_sq = nn_sq_dist[i];

  if (sorted_idx < 0 || dist_sq >= max_dist_sq) {
    correspondences[i] = -1;
    sq_distances[i] = dist_sq;
    return;
  }

  int target_idx = point_indices[sorted_idx];
  correspondences[i] = target_idx;
  sq_distances[i] = dist_sq;

  const float* covA = &source_covs[i * 16];
  const float* covB = &target_covs[target_idx * 16];

  // RCR = cov_B + T * cov_A * T^T (3x3 block)
  float TcA[9];
  for (int r = 0; r < 3; ++r)
    for (int col = 0; col < 3; ++col)
      TcA[r*3+col] = T[r*4+0]*covA[0*4+col] + T[r*4+1]*covA[1*4+col] + T[r*4+2]*covA[2*4+col];

  float RCR[16];
  for (int j = 0; j < 16; ++j) RCR[j] = 0.f;
  for (int r = 0; r < 3; ++r)
    for (int col = 0; col < 3; ++col) {
      float val = covB[r*4+col];
      for (int k = 0; k < 3; ++k) val += TcA[r*3+k] * T[col*4+k];
      RCR[r*4+col] = val;
    }
  RCR[15] = 1.0f;

  float* Minv = &mahalanobis[i * 16];
  invert3x3(RCR, Minv);
  Minv[15] = 0.0f;
}

int findCorrespondencesAndMahalanobis(
    const float4* d_source, size_t n_source,
    const float* d_source_covs,
    const float* d_target_covs,
    const float* d_T_4x4,
    const VoxelHashMapGPU& voxel_map,
    float max_corr_dist,
    int* d_correspondences,
    float* d_sq_distances,
    float* d_mahalanobis,
    float4* d_transformed,
    int* d_nn_idx,
    float* d_nn_dist,
    cudaStream_t stream) {

  if (n_source == 0) return 0;

  constexpr int kBlock = 256;
  int grid = (n_source + kBlock - 1) / kBlock;

  dlio::cuda::transformPointCloud(d_source, d_transformed, n_source, d_T_4x4, stream);

  voxel_map.search(d_transformed, n_source, d_nn_idx, d_nn_dist, max_corr_dist, stream);

  float max_dist_sq = max_corr_dist * max_corr_dist;
  correspondenceAndMahalanobisKernel<<<grid, kBlock, 0, stream>>>(
      d_source, d_source_covs, d_target_covs, d_T_4x4,
      voxel_map.d_sorted_points_, d_nn_idx, d_nn_dist,
      n_source, voxel_map.d_point_indices_, max_dist_sq,
      d_correspondences, d_sq_distances, d_mahalanobis);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<int> h_corr(n_source);
  CUDA_CHECK(cudaMemcpy(h_corr.data(), d_correspondences, n_source * sizeof(int), cudaMemcpyDeviceToHost));
  int count = 0;
  for (size_t i = 0; i < n_source; ++i)
    if (h_corr[i] >= 0) ++count;
  return count;
}

} // namespace cuda
} // namespace dlio
