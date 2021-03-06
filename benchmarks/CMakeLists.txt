﻿cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(EXAMPLE LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{tCC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    add_executable(${CMAKE_BENCH_NAME}
                   ${CMAKE_BENCH_SRC})
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_BENCH_NAME} benchmark benchmark_main pthread)
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
endfunction(ConfigureBench)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_SOURCE_DIR}"
                    "${GBENCH_INCLUDE_DIR}"
                    "${CMAKE_CURRENT_SOURCE_DIR}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${GBENCH_LIBRARY_DIR}")

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - example benchmark --------------------------------------------------------------------------------

set(EXAMPLE_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/example/example_benchmark.cpp")

ConfigureBench(EXAMPLE_BENCH "${EXAMPLE_BENCH_SRC}")

###################################################################################################
# - example CUDA benchmark --------------------------------------------------------------------------------

set(EXAMPLE_CUDA_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/example/example_benchmark.cu")

ConfigureBench(EXAMPLE_CUDA_BENCH "${EXAMPLE_CUDA_BENCH_SRC}")

###################################################################################################