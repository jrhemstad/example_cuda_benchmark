
#include <benchmark/benchmark.h>

#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <cuda/atomic>
#include <synchronization.hpp>

template <typename T>
void BM_weak_sequential_load(benchmark::State &state)
{
  thrust::device_vector<T> v(state.range(0));
  for (auto _ : state) {
    cuda_event_timer raii{state};
    auto const begin = thrust::make_counting_iterator(0);
    auto const end   = begin + state.range(0);
    thrust::for_each(thrust::device,
                     begin,
                     end,
                     [input_data = v.data().get(), input_size = v.size()] __device__(auto index) {
                       volatile auto l = input_data[index];
                     });
  }
}
BENCHMARK_TEMPLATE(BM_weak_sequential_load, int32_t)
  ->RangeMultiplier(10)
  ->Range(100'000, 1'000'000'000)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);

template <typename T>
void BM_atomic_sequential_load(benchmark::State &state)
{
  thrust::device_vector<cuda::atomic<T, cuda::thread_scope_device>> v(state.range(0));
  for (auto _ : state) {
    cuda_event_timer raii{state};
    auto const begin = thrust::make_counting_iterator(0);
    auto const end   = begin + state.range(0);
    thrust::for_each(thrust::device,
                     begin,
                     end,
                     [input_data = v.data().get(), input_size = v.size()] __device__(auto index) {
                       volatile auto l = input_data[index].load(cuda::std::memory_order_relaxed);
                     });
  }
}
BENCHMARK_TEMPLATE(BM_atomic_sequential_load, int32_t)
  ->RangeMultiplier(10)
  ->Range(100'000, 1'000'000'000)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);
