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
    timeout_test.cpp
    timeout_propagation_test.cpp
    timeout_error_category_test.cpp
    retry_test.cpp
    degrade_test.cpp
    algorithm_registry_test.cpp
    error_handling_integration_test.cpp
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
    neural/sync_batch_norm_test.cu
    neural/tensor_parallel_test.cpp
    neural/fusion/fused_matmul_bias_act_test.cpp
    neural/fusion/fused_matmul_bias_act_edge_test.cpp
    neural/transformer/attention_test.cpp
    neural/loss/loss_functions_test.cpp
    neural/optimizers/optimizers_test.cpp
    tools/tools_test.cpp
    mesh/device_mesh_test.cu
    cuda/nccl/test_nccl_collectives.cpp
    distributed/distributed_ops_test.cu
    distributed/distributed_pool_test.cu
    distributed/distributed_matmul_test.cu
    pipeline/pipeline_test.cpp
    mpi/test_mpi_context.cpp
    topology/test_topology_map.cpp
    multinode/test_multi_node_context.cpp
    performance/profiler_test.cpp
    performance/autotuner_test.cu
    performance/autotuner_edge_test.cpp
    memory_opt/memory_optimizer_test.cpp
    memory_opt/memory_optimizer_edge_test.cpp
    algo_sort_test.cu
    production/graph_executor_test.cu
    production/performance_test.cpp
    production/observability_test.cpp
    production/stress_test.cpp
    algo/flash_attention_test.cu
    algo/flash_attention_edge_test.cu
    memory/kv_cache_allocator_test.cpp
    memory/kv_cache_allocator_edge_test.cpp
    memory/streaming_cache_test.cpp
    memory/prefix_sharing_test.cpp
    memory/attention_sink_test.cpp
    memory/fragmentation_test.cpp
    inference/block_manager_test.cpp
    inference/block_manager_edge_test.cpp
    inference/beam_search_test.cpp
    inference/speculative_decoding_test.cpp
    inference/scheduler_test.cpp
    inference/scheduler_edge_test.cpp
    inference/integration_test.cpp
    distributed/sequence_parallel_test.cpp
    distributed/sequence_parallel_edge_test.cpp
    observability/timeline_test.cpp
    observability/bandwidth_tracker_test.cpp
    observability/kernel_stats_test.cpp
    observability/occupancy_analyzer_test.cpp
    algo/segmented_sort_test.cpp
    algo/spmv_test.cpp
    algo/sample_sort_test.cpp
    algo/sssp_test.cpp
    testing/memory_safety_test.cpp
    testing/robustness_test.cpp
    testing/integration_test.cpp
    sparse/matrix_test.cpp
    sparse/krylov_test.cpp
    sparse/krylov_gpu_test.cpp
    sparse/sparse_matrix_test.cpp
    sparse/integration_test.cpp
    sparse/cusparse_test.cpp
    sparse/preconditioner_test.cpp
    sparse/reordering_test.cpp
    sparse/ilu_preconditioner_test.cpp
    sparse/preconditioned_solver_test.cpp
    sparse/preconditioner_benchmark_test.cpp
    performance/nvblox_metrics_test.cpp
    performance/fusion_analyzer_test.cpp
    performance/bandwidth_analysis_test.cpp
    performance/dashboard_test.cpp
    performance/integration_test.cpp
    quantize/fp8_types_test.cpp
    quantize/fp8_gemm_test.cpp
    quantize/int8_kernels_test.cpp
    quantize/calibrator_test.cpp
    quantize/qat_test.cpp
    quantize/benchmark_test.cpp
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
set(CUDA_NCCL_DIR ${CMAKE_SOURCE_DIR}/include/cuda/nccl)
set(CUDA_PIPELINE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/pipeline)
set(CUDA_DISTRIBUTED_DIR ${CMAKE_SOURCE_DIR}/include/cuda/distributed)
set(CUDA_MPI_DIR ${CMAKE_SOURCE_DIR}/include/cuda/mpi)
set(CUDA_TOPOLOGY_DIR ${CMAKE_SOURCE_DIR}/include/cuda/topology)
set(CUDA_MULTINODE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/multinode)
set(CUDA_PRODUCTION_DIR ${CMAKE_SOURCE_DIR}/include/cuda/production)
set(CUDA_BENCHMARK_DIR ${CMAKE_SOURCE_DIR}/include/cuda/benchmark)
set(CUDA_OBSERVABILITY_DIR ${CMAKE_SOURCE_DIR}/include/cuda/observability)
set(CUDA_ERROR_DIR ${CMAKE_SOURCE_DIR}/include/cuda/error)
set(CUDA_TIMEOUT_DIR ${CMAKE_SOURCE_DIR}/include/cuda/timeout)
set(CUDA_SPARSE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/sparse)
set(CUDA_QUANTIZE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/quantize)
set(DISTRIBUTED_TESTS_DIR ${CMAKE_SOURCE_DIR}/tests/distributed)

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
    ${CUDA_NCCL_DIR}
    ${CUDA_PIPELINE_DIR}
    ${CUDA_DISTRIBUTED_DIR}
    ${CUDA_MPI_DIR}
    ${CUDA_TOPOLOGY_DIR}
    ${CUDA_MULTINODE_DIR}
    ${CUDA_PRODUCTION_DIR}
    ${CUDA_BENCHMARK_DIR}
    ${CUDA_OBSERVABILITY_DIR}
    ${CUDA_ERROR_DIR}
    ${CUDA_SPARSE_DIR}
    ${CUDA_QUANTIZE_DIR}
    ${DISTRIBUTED_TESTS_DIR}
)

target_compile_options(nova-tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -lineinfo
    >
)

include(GoogleTest)
gtest_discover_tests(nova-tests)

# Fuzz Testing
include(CMakeFuzz.txt)

# Property-Based Testing
include(CMakeProperty.txt)
