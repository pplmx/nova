cmake_minimum_required(VERSION 4.0)

project(nova_benchmarks CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

set(BENCHMARK_SOURCES
    benchmark_kernels.cu
)

add_executable(nova_benchmarks ${BENCHMARK_SOURCES})

target_include_directories(nova_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(nova_benchmarks PRIVATE
    benchmark
    cuda_impl
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
)

target_compile_definitions(nova_benchmarks PRIVATE
    NOVA_NVTX_ENABLED=1
)

set_target_properties(nova_benchmarks PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS nova_benchmarks DESTINATION bin)
