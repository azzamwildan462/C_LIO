#include "dlio/cuda/robust_icp_cuda.h"
#include "dlio/cuda/nn_search.cuh"
#include "dlio/cuda/transforms.cuh"
#include "dlio/cuda/common.cuh"

#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace dlio {
namespace cuda {

// Per-point output: 21 (upper-tri JTJ) + 6 (JTr) + 1 (count) = 28
static constexpr int RICP_WORK = 28;

// ============================================================
// GPU Kernel: per-point Jacobian + GM weight + JTJ_i/JTr_i
// ============================================================

__global__ void robustIcpKernel(
    const float4* __restrict__ trans_src,   // [N] transformed source (float4: x,y,z,w)
    const float4* __restrict__ sorted_tgt,  // sorted target points
    const int* __restrict__ nn_indices,     // [N] NN index into sorted_tgt (-1=no match)
    const float* __restrict__ nn_sq_dist,   // [N] squared distance
    const int32_t* __restrict__ point_indices, // sorted→original target mapping
    size_t n_source,
    float max_dist_sq,
    float sigma2,
    float* __restrict__ out)                // [N * 28]
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_source) return;

  float* my = out + i * RICP_WORK;
  for (int j = 0; j < RICP_WORK; ++j) my[j] = 0.f;

  int si = nn_indices[i];
  if (si < 0 || nn_sq_dist[i] >= max_dist_sq) return;

  float4 sp = trans_src[i];
  float4 tp = sorted_tgt[si];

  float rx = sp.x - tp.x;
  float ry = sp.y - tp.y;
  float rz = sp.z - tp.z;
  float r2 = rx*rx + ry*ry + rz*rz;

  // Geman-McClure weight: w = σ² / (σ² + r²)²
  float denom = sigma2 + r2;
  float w = sigma2 / (denom * denom);

  // Jacobian J (3x6): [I₃ | -hat(src)]
  // J = [[1,0,0,  0, sz,-sy],
  //      [0,1,0, -sz, 0, sx],
  //      [0,0,1,  sy,-sx, 0]]
  float sx = sp.x, sy = sp.y, sz = sp.z;

  // JTJ = J^T * w * J (6x6, symmetric → 21 upper triangle)
  // JTr = J^T * w * r (6x1)
  // Compute J^T * w * r first (6x1)
  float wr[3] = { w*rx, w*ry, w*rz };
  // JTr[0] = wr[0], JTr[1] = wr[1], JTr[2] = wr[2]
  // JTr[3] = -sz*wr[1] + sy*wr[2]
  // JTr[4] = sz*wr[0] - sx*wr[2]
  // JTr[5] = -sy*wr[0] + sx*wr[1]
  float jtr[6];
  jtr[0] = wr[0];
  jtr[1] = wr[1];
  jtr[2] = wr[2];
  jtr[3] = -sz*wr[1] + sy*wr[2];
  jtr[4] =  sz*wr[0] - sx*wr[2];
  jtr[5] = -sy*wr[0] + sx*wr[1];

  // JTJ (6x6 symmetric) — compute all 21 upper-triangle entries
  // J^T * w * J, where J cols are: e1,e2,e3, [-hat(src)] cols
  // Col 0-2 of J = I₃, Col 3-5 = [-hat(src)]
  // So J^T*w*J(i,j) = w * J_col_i . J_col_j

  // J columns as 3-vectors:
  // c0=[1,0,0], c1=[0,1,0], c2=[0,0,1]
  // c3=[0,-sz,sy], c4=[sz,0,-sx], c5=[-sy,sx,0]
  float c3[3] = {0, -sz, sy};
  float c4[3] = {sz, 0, -sx};
  float c5[3] = {-sy, sx, 0};

  // Upper triangle (row-major index mapping):
  // (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),
  //       (1,1),(1,2),(1,3),(1,4),(1,5),
  //             (2,2),(2,3),(2,4),(2,5),
  //                   (3,3),(3,4),(3,5),
  //                         (4,4),(4,5),
  //                               (5,5)
  int idx = 0;
  float cols[6][3] = {{1,0,0},{0,1,0},{0,0,1},{c3[0],c3[1],c3[2]},{c4[0],c4[1],c4[2]},{c5[0],c5[1],c5[2]}};
  for (int r = 0; r < 6; ++r)
    for (int c = r; c < 6; ++c)
      my[idx++] = w * (cols[r][0]*cols[c][0] + cols[r][1]*cols[c][1] + cols[r][2]*cols[c][2]);

  // JTr (6 values)
  for (int j = 0; j < 6; ++j) my[21+j] = jtr[j];

  // Count
  my[27] = 1.f;
}

// Reduction: sum RICP_WORK floats across N points
__global__ void reduceRicpKernel(const float* __restrict__ in, float* __restrict__ out,
                                  size_t n_points) {
  int comp = blockIdx.x;
  if (comp >= RICP_WORK) return;

  float sum = 0.f;
  for (size_t i = threadIdx.x; i < n_points; i += blockDim.x)
    sum += in[i * RICP_WORK + comp];

  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);

  __shared__ float sdata[32];
  int lane = threadIdx.x % 32, warp_id = threadIdx.x / 32;
  if (lane == 0) sdata[warp_id] = sum;
  __syncthreads();

  if (warp_id == 0) {
    sum = (lane < (blockDim.x + 31) / 32) ? sdata[lane] : 0.f;
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) atomicAdd(&out[comp], sum);
  }
}

// ============================================================
// RobustIcpCudaImpl
// ============================================================

class RobustIcpCudaImpl {
public:
  VoxelHashMapGPU voxel_map;

  DeviceBuffer<float4> d_source;
  DeviceBuffer<float4> d_trans_src;
  DeviceBuffer<float4> d_target_raw; // for voxel map build
  DeviceBuffer<int> d_nn_idx;
  DeviceBuffer<float> d_nn_dist;
  DeviceBuffer<float> d_workspace; // N * RICP_WORK + RICP_WORK
  DeviceBuffer<float> d_T;         // 16

  size_t n_source = 0;
  size_t n_target = 0;

  void uploadSource(const std::vector<float4>& pts) {
    n_source = pts.size();
    d_source.upload(pts.data(), n_source);
    d_trans_src.allocate(n_source);
    d_nn_idx.allocate(n_source);
    d_nn_dist.allocate(n_source);
    size_t ws = n_source * RICP_WORK + RICP_WORK;
    d_workspace.allocate(ws);
    d_T.allocate(16);
  }

  void uploadTarget(const std::vector<float4>& pts, float voxel_size) {
    n_target = pts.size();
    d_target_raw.upload(pts.data(), n_target);
    voxel_map.build(d_target_raw.data(), n_target, voxel_size);
  }

  void transformSource(const float T[16]) {
    d_T.upload(T, 16);
    dlio::cuda::transformPointCloud(d_source.data(), d_trans_src.data(), n_source, d_T.data());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Returns n_correspondences
  int computeNormalEquations(float max_corr_dist, float sigma2, float* h_jtj_upper, float* h_jtr, float& h_count) {
    if (n_source == 0) return 0;

    // NN search
    voxel_map.search(d_trans_src.data(), n_source, d_nn_idx.data(), d_nn_dist.data(), max_corr_dist);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel
    constexpr int kBlock = 256;
    int grid = (n_source + kBlock - 1) / kBlock;
    float max_dist_sq = max_corr_dist * max_corr_dist;
    robustIcpKernel<<<grid, kBlock>>>(
        d_trans_src.data(), voxel_map.d_sorted_points_,
        d_nn_idx.data(), d_nn_dist.data(), voxel_map.d_point_indices_,
        n_source, max_dist_sq, sigma2, d_workspace.data());

    // Reduce
    float* d_reduced = d_workspace.data() + n_source * RICP_WORK;
    CUDA_CHECK(cudaMemset(d_reduced, 0, RICP_WORK * sizeof(float)));
    reduceRicpKernel<<<RICP_WORK, 256>>>(d_workspace.data(), d_reduced, n_source);

    float h_reduced[RICP_WORK];
    CUDA_CHECK(cudaMemcpy(h_reduced, d_reduced, RICP_WORK * sizeof(float), cudaMemcpyDeviceToHost));

    for (int j = 0; j < 21; ++j) h_jtj_upper[j] = h_reduced[j];
    for (int j = 0; j < 6; ++j) h_jtr[j] = h_reduced[21 + j];
    h_count = h_reduced[27];
    return static_cast<int>(h_count);
  }
};

// ============================================================
// SE3 exp map (CPU, same as RobustICP::expSE3)
// ============================================================

static Eigen::Matrix3f hat(const Eigen::Vector3f& v) {
  Eigen::Matrix3f m;
  m << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
  return m;
}

static Eigen::Matrix4f expSE3(const Eigen::Matrix<float,6,1>& twist) {
  Eigen::Vector3f t = twist.head<3>();
  Eigen::Vector3f w = twist.tail<3>();
  float theta = w.norm();
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  if (theta < 1e-6f) {
    T.block<3,3>(0,0) = Eigen::Matrix3f::Identity() + hat(w);
    T.block<3,1>(0,3) = t;
  } else {
    Eigen::Matrix3f W = hat(w / theta);
    float s = std::sin(theta), c = std::cos(theta);
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity() + s*W + (1.f-c)*W*W;
    Eigen::Matrix3f V = Eigen::Matrix3f::Identity() + ((1.f-c)/theta)*W + ((theta-s)/theta)*W*W;
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = V * t;
  }
  return T;
}

// ============================================================
// RobustIcpCuda public API
// ============================================================

RobustIcpCuda::RobustIcpCuda() : impl_(std::make_unique<RobustIcpCudaImpl>()) {}
RobustIcpCuda::~RobustIcpCuda() = default;

void RobustIcpCuda::setInputSource(pcl::PointCloud<PointType>::ConstPtr cloud) {
  source_ = cloud;
  std::vector<float4> pts(cloud->size());
  for (size_t i = 0; i < cloud->size(); ++i)
    pts[i] = make_float4(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, cloud->points[i].intensity);
  impl_->uploadSource(pts);
}

void RobustIcpCuda::setInputTarget(pcl::PointCloud<PointType>::ConstPtr cloud) {
  target_ = cloud;
  std::vector<float4> pts(cloud->size());
  for (size_t i = 0; i < cloud->size(); ++i)
    pts[i] = make_float4(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, cloud->points[i].intensity);
  // Voxel size for NN hash: use max_corr_dist as a reasonable voxel size
  float vs = static_cast<float>(max_corr_dist_ > 0 ? max_corr_dist_ : 1.0);
  impl_->uploadTarget(pts, vs);
}

void RobustIcpCuda::align(pcl::PointCloud<PointType>& output) {
  align(output, Eigen::Matrix4f::Identity());
}

void RobustIcpCuda::align(pcl::PointCloud<PointType>& output, const Eigen::Matrix4f& guess) {
  converged_ = false;
  Eigen::Matrix4f T_accum = guess;

  double sigma = (kernel_scale_ > 0) ? kernel_scale_ : max_corr_dist_;
  float sigma2 = static_cast<float>(sigma * sigma);

  for (int iter = 0; iter < max_iterations_; ++iter) {
    // GPU: transform source by T_accum
    float T_arr[16];
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        T_arr[r*4+c] = T_accum(r, c);
    impl_->transformSource(T_arr);

    // GPU: NN search + Jacobian + weighted normal equations
    float h_jtj_upper[21], h_jtr[6], h_count;
    int n_corr = impl_->computeNormalEquations(
        static_cast<float>(max_corr_dist_), sigma2, h_jtj_upper, h_jtr, h_count);

    if (n_corr < 10) break;

    // Reconstruct full 6x6 JTJ from upper triangle
    Eigen::Matrix<float,6,6> JTJ;
    int idx = 0;
    for (int r = 0; r < 6; ++r)
      for (int c = r; c < 6; ++c) {
        JTJ(r, c) = h_jtj_upper[idx];
        JTJ(c, r) = h_jtj_upper[idx];
        idx++;
      }

    Eigen::Matrix<float,6,1> JTr;
    for (int j = 0; j < 6; ++j) JTr(j) = h_jtr[j];

    // LDLT solve
    Eigen::Matrix<float,6,1> dx = JTJ.ldlt().solve(-JTr);

    // SE3 exp map
    Eigen::Matrix4f T_inc = expSE3(dx);
    T_accum = T_inc * T_accum;

    if (dx.norm() < convergence_eps_) { converged_ = true; break; }
  }

  final_transformation_ = T_accum;
  if (source_) pcl::transformPointCloud(*source_, output, final_transformation_);
}

Eigen::Matrix4f RobustIcpCuda::getFinalTransformation() { return final_transformation_; }
bool RobustIcpCuda::hasConverged() { return converged_; }

double RobustIcpCuda::getFitnessScore(double max_range) {
  if (!source_ || !target_ || source_->empty()) return std::numeric_limits<double>::max();
  // CPU fitness (not perf critical)
  pcl::PointCloud<PointType> transformed;
  pcl::transformPointCloud(*source_, transformed, final_transformation_);

  // Download voxel hash to host for lookup
  std::vector<VoxelHashMapGPU::HashEntry> h_hash(VoxelHashMapGPU::HASH_TABLE_SIZE);
  CUDA_CHECK(cudaMemcpy(h_hash.data(), impl_->voxel_map.d_hash_table_,
                         VoxelHashMapGPU::HASH_TABLE_SIZE * sizeof(VoxelHashMapGPU::HashEntry), cudaMemcpyDeviceToHost));
  std::vector<float4> h_sorted(impl_->voxel_map.numPoints());
  CUDA_CHECK(cudaMemcpy(h_sorted.data(), impl_->voxel_map.d_sorted_points_,
                         h_sorted.size() * sizeof(float4), cudaMemcpyDeviceToHost));

  float inv_voxel = 1.0f / impl_->voxel_map.voxel_size_;
  double score = 0; int nr = 0;
  for (const auto& pt : transformed.points) {
    if (!std::isfinite(pt.x)) continue;
    int3 v = make_int3(static_cast<int>(floorf(pt.x * inv_voxel)),
                        static_cast<int>(floorf(pt.y * inv_voxel)),
                        static_cast<int>(floorf(pt.z * inv_voxel)));
    float best_d2 = static_cast<float>(max_range * max_range);
    // Lookup self voxel with open addressing
    uint32_t hv = 2166136261u;
    hv ^= static_cast<uint32_t>(v.x); hv *= 16777619u;
    hv ^= static_cast<uint32_t>(v.y); hv *= 16777619u;
    hv ^= static_cast<uint32_t>(v.z); hv *= 16777619u;
    uint32_t h = hv & (VoxelHashMapGPU::HASH_TABLE_SIZE - 1);
    for (int p = 0; p < VoxelHashMapGPU::MAX_PROBE; ++p) {
      uint32_t slot = (h + p) & (VoxelHashMapGPU::HASH_TABLE_SIZE - 1);
      const auto& entry = h_hash[slot];
      if (entry.count <= 0) break;
      if (entry.vx == v.x && entry.vy == v.y && entry.vz == v.z) {
        for (int j = 0; j < entry.count; ++j) {
          float4 tp = h_sorted[entry.start + j];
          float dx = pt.x-tp.x, dy = pt.y-tp.y, dz = pt.z-tp.z;
          float d2 = dx*dx+dy*dy+dz*dz;
          if (d2 < best_d2) best_d2 = d2;
        }
        break;
      }
    }
    if (best_d2 < max_range * max_range) { score += std::sqrt(best_d2); nr++; }
  }
  return (nr > 0) ? (score / nr) : std::numeric_limits<double>::max();
}

} // namespace cuda
} // namespace dlio
