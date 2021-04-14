
#include <benchmark/benchmark.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <synchronization.hpp>
#include <string>
#include <exception>

inline void throw_cuda_error(cudaError_t error, const char *file, unsigned int line)
{
  throw std::runtime_error(std::string{"CUDA error encountered at: " + std::string{file} + ":" +
                                       std::to_string(line) + ": " + std::to_string(error) + " " +
                                       cudaGetErrorName(error) + " " + cudaGetErrorString(error)});
}

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 */
#define CUDA_TRY(call)                                            \
  do {                                                            \
    cudaError_t const status = (call);                            \
    if (cudaSuccess != status) {                                  \
      cudaGetLastError();                                         \
      throw_cuda_error(status, __FILE__, __LINE__); \
    }                                                             \
  } while (0)


void BM_cuda_async_no_threshold(benchmark::State &state)
{
  cudaStream_t s;
  cudaStreamCreate(&s);

  auto constexpr block_size = 256;
  auto const grid_size      = (state.range(0) + block_size - 1) / block_size;

  for (auto _ : state) {
    int *ptr;
    CUDA_TRY(cudaMallocAsync(&ptr, state.range(0) * sizeof(int), s));
    CUDA_TRY(cudaFreeAsync(ptr, s));
    CUDA_TRY(cudaStreamSynchronize(s));
  }
  cudaStreamDestroy(s);
}
BENCHMARK(BM_cuda_async_no_threshold)
  ->RangeMultiplier(10)
  ->Range(100'000, 100'000'000)
  ->Unit(benchmark::kMicrosecond);

void BM_cuda_async_threshold(benchmark::State &state)
{
  cudaStream_t s;
  CUDA_TRY(cudaStreamCreate(&s));

  auto constexpr block_size = 256;
  auto const grid_size      = (state.range(0) + block_size - 1) / block_size;

  cudaMemPool_t mempool;
  CUDA_TRY(cudaDeviceGetDefaultMemPool(&mempool, 0));
  uint64_t threshold = UINT64_MAX;
  CUDA_TRY(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));

  for (auto _ : state) {
    int *ptr;
    CUDA_TRY(cudaMallocAsync(&ptr, state.range(0) * sizeof(int), s));
    CUDA_TRY(cudaFreeAsync(ptr, s));
    CUDA_TRY(cudaStreamSynchronize(s));
  }
  CUDA_TRY(cudaStreamDestroy(s));
}
BENCHMARK(BM_cuda_async_threshold)
  ->RangeMultiplier(10)
  ->Range(100'000, 100'000'000)
  ->Unit(benchmark::kMicrosecond);
