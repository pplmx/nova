find_package(CUDAToolkit REQUIRED)

add_executable(test_patterns-tests
    test_patterns_test.cpp
)
target_link_libraries(test_patterns-tests PRIVATE GTest::gtest_main)
target_include_directories(test_patterns-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/data
)
include(GoogleTest)
gtest_discover_tests(test_patterns-tests)

add_executable(nova-tests
    image_utils_test.cu
    gaussian_blur_test.cu
    sobel_edge_test.cu
    brightness_test.cu
    reduce_test.cu
    scan_test.cu
    sort_test.cu
    matrix_mult_test.cu
    convolution_test.cu
    memory_pool_test.cpp
    memory_pool_metrics_test.cpp
    memory_pool_stream_test.cpp
    stream_event_test.cpp
    cublas_context_test.cu
    device_info_test.cpp
    memory_metrics_test.cpp
    error_context_test.cpp
    benchmark/benchmark_test.cpp
    benchmark/throughput_test.cpp
    benchmark/regression_test.cpp
    stream_manager_test.cpp
    pinned_memory_test.cpp
    async_copy_test.cpp
    fft/fft_plan_test.cpp
    fft/fft_inverse_test.cpp
    fft/fft_accuracy_test.cpp
    raytrace/ray_box_test.cpp
    raytrace/ray_sphere_test.cpp
    raytrace/bvh_test.cpp
    graph/csr_graph_test.cpp
    graph/bfs_test.cpp
    graph/pagerank_test.cpp
    neural/matmul_test.cpp
    neural/softmax_test.cpp
    neural/activations_test.cpp
    neural/layer_norm_test.cpp
    mesh/device_mesh_test.cu
    distributed/distributed_ops_test.cu
)

set(CUDA_DEVICE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/device)
set(CUDA_ALGO_DIR ${CMAKE_SOURCE_DIR}/include/cuda/algo)
set(CUDA_STREAM_DIR ${CMAKE_SOURCE_DIR}/include/cuda/stream)
set(CUDA_PERFORMANCE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/performance)
set(CUDA_MEMORY_DIR ${CMAKE_SOURCE_DIR}/include/cuda/memory)
set(CUDA_BENCHMARK_DIR ${CMAKE_SOURCE_DIR}/include/cuda/benchmark)
set(CUDA_ASYNC_DIR ${CMAKE_SOURCE_DIR}/include/cuda/async)
set(CUDA_FFT_DIR ${CMAKE_SOURCE_DIR}/include/cuda/fft)
set(CUDA_RAYTRACE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/raytrace)
set(CUDA_GRAPH_DIR ${CMAKE_SOURCE_DIR}/include/cuda/graph)
set(CUDA_NEURAL_DIR ${CMAKE_SOURCE_DIR}/include/cuda/neural)
set(CUDA_MESH_DIR ${CMAKE_SOURCE_DIR}/include/cuda/mesh)
set(CUDA_DISTRIBUTED_DIR ${CMAKE_SOURCE_DIR}/include/cuda/distributed)

target_link_libraries(nova-tests
    PRIVATE
    GTest::gtest_main
    GTest::gmock
    CUDA::cudart
    cuda_api
    cuda_impl
)
target_include_directories(nova-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/data
    ${CUDA_DEVICE_DIR}
    ${CUDA_ALGO_DIR}
    ${CUDA_STREAM_DIR}
    ${CUDA_PERFORMANCE_DIR}
    ${CUDA_MEMORY_DIR}
    ${CUDA_BENCHMARK_DIR}
    ${CUDA_ASYNC_DIR}
    ${CUDA_FFT_DIR}
    ${CUDA_RAYTRACE_DIR}
    ${CUDA_GRAPH_DIR}
    ${CUDA_NEURAL_DIR}
    ${CUDA_MESH_DIR}
    ${CUDA_DISTRIBUTED_DIR}
)

target_compile_options(nova-tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -lineinfo
    >
)

include(GoogleTest)
gtest_discover_tests(nova-tests)
