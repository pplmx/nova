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
    histogram_test.cu
    matrix_ops_test.cu
    morphology_test.cu
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
    neural/tensor_parallel_multigpu_test.cpp
    neural/tensor_parallel_layers_test.cpp
    neural/micro_trainer_test.cpp
    neural/micro_trainer_multigpu_test.cpp
    neural/tensor_parallel_attention_test.cpp
    neural/tensor_parallel_attention_multigpu_test.cpp
    neural/attention_kernels_test.cpp
    neural/layer_norm_trainable_test.cpp
    neural/tensor_parallel_block_test.cpp
    neural/tensor_parallel_block_multigpu_test.cpp
    neural/fusion/fused_matmul_bias_act_test.cpp
    neural/fusion/fused_matmul_bias_act_edge_test.cpp
    neural/transformer/attention_test.cpp
    neural/loss/loss_functions_test.cpp
    neural/optimizers/optimizers_test.cpp
    neural/checkpoint_test.cpp
    neural/checkpoint_multigpu_test.cpp
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
    distributed/sequence_parallel_multigpu_test.cu
    preemption/preemption_handler_test.cpp
    observability/timeline_test.cpp
    observability/bandwidth_tracker_test.cpp
    observability/kernel_stats_test.cpp
    observability/occupancy_analyzer_test.cpp
    algo/segmented_sort_test.cpp
    algo/spmv_test.cpp
    algo/sample_sort_test.cpp
    algo/sssp_test.cpp
    linalg/linalg_test.cpp
    numeric/numeric_test.cpp
    testing/memory_safety_test.cpp
    testing/robustness_test.cpp
    testing/integration_test.cpp
    sparse/matrix_test.cpp
    sparse/krylov_gpu_test.cpp
    sparse/sparse_matrix_test.cpp
    sparse/integration_test.cpp
    sparse/cusparse_test.cpp
    sparse/preconditioner_test.cpp
    sparse/reordering_test.cpp
    sparse/ilu_preconditioner_test.cpp
    sparse/hyb_matrix_test.cpp
    sparse/header_self_containment_test.cpp
    sparse/benchmark_dequantization_test.cpp
    sparse/declared_but_unimplemented_test.cpp
    sparse/multiply_exception_safety_test.cpp
    sparse/documented_examples_test.cpp
    sparse/device_pointer_contract_test.cpp
    sparse/unique_ptr_ownership_test.cpp
    sparse/priority_stream_pool_test.cpp
    sparse/stream_event_ownership_test.cpp
    sparse/sync_batch_norm_ownership_test.cpp
    sparse/move_semantics_audit_test.cpp
    sparse/reduce_nan_semantics_test.cpp
    sparse/layer_norm_optional_params_test.cpp
    sparse/sparse_header_compile_test.cpp
    sparse/spmv_host_test.cpp
    sparse/krylov_zero_rhs_test.cpp
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
    quantize/quantize_tensor_test.cpp
    quantize/quantize_ops_test.cpp
    quantize/fp8_activation_test.cpp
    quantize/fp8_kernels_test.cpp
    gnn/gnn_test.cpp
    sparse/cuda_error_test.cpp
    sparse/memory_error_handler_test.cpp
    sparse/comm_error_recovery_test.cpp
    sparse/activation_buffer_test.cpp
    sparse/p2p_primitives_test.cpp
    sparse/pipeline_comm_test.cpp
    sparse/nccl_validation_test.cpp
    sparse/checkpoint_manager_test.cpp
    sparse/logger_test.cpp
    sparse/fft_types_test.cpp
    sparse/signal_processing_test.cpp
    sparse/nccl_types_test.cpp
    sparse/nvtx_extensions_test.cpp
    sparse/uncovered_headers_test.cpp
    sparse/quantize_benchmark_test.cpp
    sparse/cublas_gemm_test.cpp
    sparse/raytrace_primitives_test.cpp
    sparse/matrix_add_test.cpp
    sparse/numeric_test.cpp
    sparse/optimizers_kernels_test.cpp
    sparse/linalg_decomposition_test.cpp
    sparse/ring_attention_test.cpp
    sparse/tensor_parallel_profile_test.cpp
    sparse/sync_batch_norm_test.cpp
    sparse/kernel_fusion_test.cpp
    sparse/boundary_testing_test.cpp
    sparse/timeline_export_test.cpp
    sparse/fft_plan_validation_test.cpp
    sparse/error_message_quality_test.cpp
    sparse/occupancy_analyzer_test.cpp
    sparse/histogram_bins_test.cpp
    sparse/collective_selector_env_test.cpp
    sparse/activations_edge_test.cpp
    sparse/cross_entropy_validation_test.cpp
    sparse/sampling_validation_test.cpp
    sparse/pagerank_result_test.cpp
    sparse/graph_container_state_test.cpp
    sparse/sssp_reported_stats_test.cpp
    sparse/robustness_profile_test.cpp
    sparse/device_bandwidth_formula_test.cpp
    sparse/fp8_activation_kernels_test.cpp
    sparse/fp8_gemm_test.cpp
    sparse/benchmark_units_test.cpp
    sparse/kernel_stats_sentinel_test.cpp
    sparse/autotuner_search_space_test.cpp
    sparse/occupancy_calculator_test.cpp
    sparse/fp_determinism_verdict_test.cpp
    sparse/sync_batch_norm_backward_test.cpp
    sparse/sdpa_head_dim_test.cpp
    sparse/transpose_edge_test.cpp
    sparse/image_morphology_edge_test.cpp
    sparse/sobel_edge_bounds_test.cpp
    sparse/spmv_output_initialisation_test.cpp
    sparse/scan_race_test.cpp
    sparse/fft_roundtrip_test.cpp
    sparse/softmax_kernel_test.cpp
    sparse/memory_pool_release_test.cpp
    sparse/dashboard_export_test.cpp
    sparse/cache_analyzer_honesty_test.cpp
    sparse/matrix_multiply_edge_test.cpp
    sparse/matrix_elementwise_edge_test.cpp
    sparse/fp8_conversion_edge_test.cpp
    sparse/launch_error_check_test.cpp
    sparse/int8_calibration_edge_test.cpp
    sparse/auto_block_device_test.cpp
    sparse/locked_accessor_test.cpp
    sparse/profiler_deadlock_test.cpp
    sparse/signal_handler_state_test.cpp
    sparse/singleton_reset_registry_test.cpp
    sparse/singleton_isolation_order_test.cpp
    sparse/quantize_precision_odr_test.cpp
    sparse/header_self_containment_test.cpp
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
set(CUDA_GNN_DIR ${CMAKE_SOURCE_DIR}/include/cuda/gnn)
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
    ${CMAKE_SOURCE_DIR}/include/matrix
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
    ${CUDA_GNN_DIR}
    ${DISTRIBUTED_TESTS_DIR}
)

# sparse/krylov_test.cpp is a standalone self-checking program (it defines its
# own main and asserts on the solver results), so it cannot be linked into the
# gtest binary. Build and register it separately.
add_executable(krylov-standalone-tests
    sparse/krylov_test.cpp
)
target_link_libraries(krylov-standalone-tests
    PRIVATE
    CUDA::cudart
    cuda_impl
)
target_include_directories(krylov-standalone-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_SPARSE_DIR}
)
add_test(NAME krylov_standalone COMMAND krylov-standalone-tests)

# reduce_kernels.h is nvcc-only (it names threadIdx/__syncthreads at namespace
# scope), so its test is a .cu file built into its own executable. Registering it
# with gtest_discover_tests would run a CUDA binary during discovery of the host
# test target.
add_executable(reduce_kernels-tests
    device/reduce_kernels_test.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/device/reduce_kernels.cu
)
target_link_libraries(reduce_kernels-tests
    PRIVATE
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(reduce_kernels-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cuda/device
)
add_test(NAME reduce_kernels COMMAND reduce_kernels-tests)

# algo_wrapper.h declares a __global__ kernel at namespace scope, so its test is
# a .cu file in its own executable for the same reason as reduce_kernels.
# Only the test source: cuda_impl already carries algo_wrapper.cu and
# graph_executor.cu, and compiling them again collided at link time.
add_executable(algo_wrapper-tests
    device/algo_wrapper_test.cu
)
target_link_libraries(algo_wrapper-tests
    PRIVATE
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
    cuda_impl
)
target_include_directories(algo_wrapper-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cuda/production
    ${CMAKE_SOURCE_DIR}/include/cuda/device
)
add_test(NAME algo_wrapper COMMAND algo_wrapper-tests)

# segmented_sort.h pulls in cuda_runtime.h and is implemented in device code, so
# its test is a .cu executable.
add_executable(segmented_sort-tests
    device/segmented_sort_test.cu
    ${CMAKE_SOURCE_DIR}/src/algo/segmented_sort.cu
)
target_link_libraries(segmented_sort-tests
    PRIVATE
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(segmented_sort-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cuda/algo
    ${CMAKE_SOURCE_DIR}/include/cuda/device
    ${CMAKE_SOURCE_DIR}/include/cuda/memory
)
add_test(NAME segmented_sort COMMAND segmented_sort-tests)

include(GoogleTest)
# The leak measurement reads device-wide free memory, so it cannot share the
# device with another test process. Exclude it from the parallel discovery set and
# register it separately as a serial test.
# Suites whose assertions compare device-wide memory snapshots: they cannot share the
# device with another test process, so they are excluded from the parallel discovery
# set and registered separately with RUN_SERIAL.
# The multi-GPU tests gate themselves on the NCCL_TESTS_AVAILABLE environment variable, while
# the library decides at compile time whether NCCL is linked (NOVA_NCCL_ENABLED). Nothing
# connected the two, so on a machine with NCCL the tests still reported SKIPPED — and gtest
# counts a skipped test as passed, so ctest showed the group green while none of it ran.
# Measured before this: ctest reported 49 passed; their verbose output held 132 SKIPPED lines;
# no assertion in them executed.
#
# Derived from the same condition the library used, so the two cannot disagree.
set(NOVA_NCCL_TESTS_ENV "")
if(TARGET cuda_nccl)
    get_target_property(_nova_nccl_defs cuda_nccl INTERFACE_COMPILE_DEFINITIONS)
    if(_nova_nccl_defs MATCHES "NOVA_NCCL_ENABLED=1")
        set(NOVA_NCCL_TESTS_ENV "NCCL_TESTS_AVAILABLE=1")
    endif()
endif()

gtest_discover_tests(nova-tests
    TEST_FILTER "-RobustnessProfileSerialTest.*:TensorParallelSerialTest.*"
    PROPERTIES ENVIRONMENT "${NOVA_NCCL_TESTS_ENV}"
)

add_test(NAME robustness_profile_serial
         COMMAND nova-tests --gtest_filter=RobustnessProfileSerialTest.*)
set_tests_properties(robustness_profile_serial PROPERTIES RUN_SERIAL TRUE)

add_test(NAME tensor_parallel_serial
         COMMAND nova-tests --gtest_filter=TensorParallelSerialTest.*)
set_tests_properties(tensor_parallel_serial PROPERTIES RUN_SERIAL TRUE)

# Fuzz Testing
include(CMakeFuzz.txt)

# Property-Based Testing
include(CMakeProperty.txt)
