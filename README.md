# CUDA Google Benchmark

A starting point for writing and building Google Benchmarks of both CUDA and CPU code.

# How to Use

An example CPU benchmark is located in `benchmarks/example/example_benchmark.cpp`.

To build this benchmark:
```bash
mkdir -p build
cd build
cmake .. && make
```

To run this benchmark:
```bash
./gbenchmarks/EXAMPLE_BENCH
```

Example output:
```
~/example_cuda_benchmark/build$ ./gbenchmarks/EXAMPLE_BENCH 
2020-06-25T09:38:23-05:00
Running ./gbenchmarks/EXAMPLE_BENCH
Run on (20 X 4500 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x10)
  L1 Instruction 32 KiB (x10)
  L2 Unified 1024 KiB (x10)
  L3 Unified 14080 KiB (x1)
Load Average: 1.14, 0.79, 0.67
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
-------------------------------------------------------------------------------
Benchmark                                     Time             CPU   Iterations
-------------------------------------------------------------------------------
BM_test                                   0.000 ns        0.000 ns   1000000000
BM_StringCopy                              4.23 ns         4.23 ns    139185384
BM_StringCompare/1024                      21.5 ns         21.5 ns     32624809
BM_StringCompare/2048                      39.3 ns         39.3 ns     17823169
BM_StringCompare/4096                      73.4 ns         73.4 ns      9546744
BM_StringCompare/8192                       147 ns          147 ns      4754180
BM_StringCompare/16384                      308 ns          308 ns      2282049
BM_StringCompare/32768                      865 ns          865 ns       845382
BM_StringCompare/65536                     1542 ns         1541 ns       447876
BM_StringCompare/131072                    3136 ns         3136 ns       215387
BM_StringCompare/262144                    6005 ns         6005 ns       115466
BM_StringCompare_BigO                      0.02 N          0.02 N    
BM_StringCompare_RMS                          5 %             5 %    
BM_Sequential<std::vector<int>>/1          5.36 ns         5.36 ns    109825554 bytes_per_second=177.79M/s
BM_Sequential<std::vector<int>>/8          50.8 ns         50.8 ns     10000000 bytes_per_second=150.11M/s
BM_Sequential<std::vector<int>>/64          420 ns          420 ns      2283232 bytes_per_second=145.212M/s
BM_Sequential<std::vector<int>>/512        2504 ns         2504 ns       252367 bytes_per_second=194.988M/s
BM_Sequential<std::vector<int>>/1024       5119 ns         5119 ns       115397 bytes_per_second=190.757M/s
```
# Adding New Benchmark

To add a new benchmark:

1. Add a new `*.cpp` or `*.cu` file to the `benchmarks/` directory.

2. Update the `benchmarks/CMakeLists.txt` to point to the new benchmark, e.g.,
```
set(MY_NEW_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/*my_new_bench_dir*/*my_new_bench.cpp/.cu*")

ConfigureBench(*MY_NEW_BENCH* "${*MY_NEW_BENCH_SRC*}"
```

3. Follow the above build instructions and your new benchmark executable `*MY_NEW_BENCH*` will be built into the `build/gbenchmarks` directory.



