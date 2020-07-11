
#include <benchmark/benchmark.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <synchronization.hpp>

template <typename T>
void BM_thrust_reduce(benchmark::State &state)
{
  thrust::device_vector<T> v(state.range(0));

  for (auto _ : state) {
    cuda_event_timer raii{state};
    auto result = thrust::reduce(v.begin(), v.end());
  }
}
BENCHMARK_TEMPLATE(BM_thrust_reduce, int32_t)
  ->RangeMultiplier(10)
  ->Range(10'000, 10'000'000)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);
