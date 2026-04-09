#ifndef DLIO_CUDA_COMMON_CUH
#define DLIO_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Error checking macro
#define CUDA_CHECK(call)                                                \
  do                                                                    \
  {                                                                     \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess)                                             \
    {                                                                   \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      throw std::runtime_error(cudaGetErrorString(err));                \
    }                                                                   \
  } while (0)

namespace dlio
{
  namespace cuda
  {

    // Simple RAII device buffer
    template <typename T>
    class DeviceBuffer
    {
    public:
      DeviceBuffer() = default;

      explicit DeviceBuffer(size_t count) { allocate(count); }

      ~DeviceBuffer() { free(); }

      // No copy
      DeviceBuffer(const DeviceBuffer &) = delete;
      DeviceBuffer &operator=(const DeviceBuffer &) = delete;

      // Move
      DeviceBuffer(DeviceBuffer &&o) noexcept : ptr_(o.ptr_), count_(o.count_)
      {
        o.ptr_ = nullptr;
        o.count_ = 0;
      }
      DeviceBuffer &operator=(DeviceBuffer &&o) noexcept
      {
        if (this != &o)
        {
          free();
          ptr_ = o.ptr_;
          count_ = o.count_;
          o.ptr_ = nullptr;
          o.count_ = 0;
        }
        return *this;
      }

      void allocate(size_t count)
      {
        if (count == count_ && ptr_)
          return;
        free();
        count_ = count;
        if (count_ > 0)
          CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
      }

      void free()
      {
        if (ptr_)
        {
          cudaFree(ptr_);
          ptr_ = nullptr;
          count_ = 0;
        }
      }

      void upload(const T *host_data, size_t count)
      {
        allocate(count);
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
      }

      void download(T *host_data, size_t count) const
      {
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
      }

      T *data() { return ptr_; }
      const T *data() const { return ptr_; }
      size_t size() const { return count_; }
      bool empty() const { return count_ == 0; }

    private:
      T *ptr_ = nullptr;
      size_t count_ = 0;
    };

  } // namespace cuda
} // namespace dlio

#endif // DLIO_CUDA_COMMON_CUH
