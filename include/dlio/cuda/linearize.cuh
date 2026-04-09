#ifndef DLIO_CUDA_LINEARIZE_CUH
#define DLIO_CUDA_LINEARIZE_CUH

#include <cuda_runtime.h>
#include <cstddef>

namespace dlio
{
    namespace cuda
    {

        // GPU linearization for GICP.
        // Per-point: compute error, Jacobian, H_i and b_i contributions.
        // Then reduce to get total H (6x6), b (6x1), and sum_errors.
        //
        // All source/target/mahalanobis data must already be on device
        // (from update_correspondences GPU path).
        //
        // d_source:          [N] source points (float4: x,y,z,w)
        // d_target:          [M] target points (float4)
        // d_correspondences: [N] matched target index (-1 if no match)
        // d_mahalanobis:     [N * 16] Mahalanobis inverse matrices (4x4 row-major float)
        // d_T_4x4:           [16] current transform (row-major float)
        // n_source:           number of source points
        //
        // Outputs (host):
        // h_H:       [36] upper-triangle of 6x6 Hessian (double), or nullptr to skip
        // h_b:       [6] gradient vector (double), or nullptr to skip
        // returns:   sum of weighted errors (double)
        double linearizeGpu(
            const float4 *d_source, size_t n_source,
            const float4 *d_target,
            const int *d_correspondences,
            const float *d_mahalanobis,
            const float *d_T_4x4,
            double *h_H, // [36] output on host, nullptr = skip H/b computation
            double *h_b, // [6] output on host
            // Workspace buffers (pre-allocated, size >= n_source * 43)
            double *d_workspace,
            cudaStream_t stream = 0);

        // GPU compute_error only (no H/b).
        double computeErrorGpu(
            const float4 *d_source, size_t n_source,
            const float4 *d_target,
            const int *d_correspondences,
            const float *d_mahalanobis,
            const float *d_T_4x4,
            double *d_workspace,
            cudaStream_t stream = 0);

    } // namespace cuda
} // namespace dlio

#endif // DLIO_CUDA_LINEARIZE_CUH
