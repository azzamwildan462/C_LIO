#ifndef C_LIO_CUDA_TRANSFORMS_CUH
#define C_LIO_CUDA_TRANSFORMS_CUH

#include <cuda_runtime.h>
#include <cstddef>

namespace c_lio
{
    namespace cuda
    {

        // Transform N points by a 4x4 matrix (row-major float[16])
        // out[i] = T * in[i]  (homogeneous, w component preserved)
        // Can operate in-place (out == in).
        void transformPointCloud(const float4 *d_in, float4 *d_out, size_t n,
                                 const float *d_T_4x4, cudaStream_t stream = 0);

        // Convenience: upload host T, run kernel, synchronize
        void transformPointCloudSync(float4 *d_points, size_t n,
                                     const float T_4x4[16]);

    } // namespace cuda
} // namespace c_lio

#endif // C_LIO_CUDA_TRANSFORMS_CUH
