# =============================================================================
# Nova CUDA Library - CMake Build Configuration
# =============================================================================
#
# Build Options:
#   -DNOVA_ENABLE_NCCL=ON|OFF         Enable NCCL multi-GPU collectives (default: ON)
#   -DNOVA_ENABLE_MPI=ON|OFF           Enable MPI multi-node support (default: OFF)
#   -DNOVA_ENABLE_UNITY_BUILD=ON|OFF    Faster compilation via unity builds (default: ON)
#
# Generators:
#   Ninja (recommended): cmake -G Ninja -B build -D<options>
#   Make:                  cmake -B build -D<options>
#
# Build & Test:
#   cmake --build build --parallel          # Compile with all CPU cores
#   ctest -j16                             # Run tests (GPU memory limited)
#
# Example (fastest build):
#   cmake -G Ninja -B build-ninja \
#       -DNOVA_ENABLE_NCCL=OFF \
#       -DNOVA_ENABLE_MPI=OFF
#   cmake --build build-ninja --parallel 128
#   cd build-ninja && ctest -j16
#
# =============================================================================

cmake_minimum_required(VERSION 4.0)

project(nova
        VERSION 0.1.0
        DESCRIPTION "A CUDA parallel algorithms library with layered architecture"
        HOMEPAGE_URL "https://github.com/pplmx/nova"
        LANGUAGES CXX CUDA
)

# =============================================================================
# Module Path
# =============================================================================
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# =============================================================================
# Language Standards
# =============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED)

# =============================================================================
# Build Options
# =============================================================================

# NCCL support (optional)
# Enable NCCL for optimized multi-GPU collectives with P2P fallback when unavailable
option(NOVA_ENABLE_NCCL "Enable NCCL for optimized multi-GPU collectives" ON)

# MPI support (optional)
# Enable multi-node training with MPI-based NCCL bootstrapping
option(NOVA_ENABLE_MPI "Enable MPI for multi-node support" OFF)

# Unity build for faster compilation (CMake 3.16+)
option(NOVA_ENABLE_UNITY_BUILD "Enable unity build to speed up compilation" ON)

# NVBlox support (optional)
# Enable NVIDIA Nsight Compute profiling integration
option(NOVA_ENABLE_NVBLOX "Enable NVBlox for kernel-level profiling" ON)

# ccache support for faster rebuilds
option(NOVA_USE_CCACHE "Enable ccache compiler launcher for faster rebuilds" OFF)

# Code coverage support
option(NOVA_COVERAGE "Enable code coverage instrumentation" OFF)
set(NOVA_COVERAGE_MIN "80" CACHE STRING "Minimum required line coverage percentage")

if(NOVA_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Enabling code coverage instrumentation")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fprofile-instr-generate,-fcoverage-mapping")
    else()
        message(WARNING "Coverage requires Clang or GCC. Coverage disabled.")
        set(NOVA_COVERAGE OFF)
    endif()
endif()

if(NOVA_USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "ccache found: ${CCACHE_PROGRAM} - enabling compiler caching")
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM} CACHE STRING "" FORCE)
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM} CACHE STRING "" FORCE)
    else()
        message(WARNING "ccache not found, but NOVA_USE_CCACHE=ON. Install ccache or set NOVA_USE_CCACHE=OFF")
    endif()
endif()

# =============================================================================
# Parallel Build Configuration
# =============================================================================
include(ProcessorCount)
if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR)
    set(CMAKE_SYSTEM_PROCESSOR "unknown")
endif()
ProcessorCount(NCPU)
if(NCPU EQUAL 0)
    set(NCPU 4)
endif()
message(STATUS "Detected ${NCPU} CPU cores - enabling parallel builds")

# GPU parallel compilation jobs
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads ${NCPU}")
endif()

# Enable Unity build batch size based on CPU cores
if(NOVA_ENABLE_UNITY_BUILD AND NCPU GREATER_EQUAL 64)
    set_property(GLOBAL PROPERTY UNITY_BUILD_BATCH_SIZE 64)
elseif(NOVA_ENABLE_UNITY_BUILD AND NCPU GREATER_EQUAL 32)
    set_property(GLOBAL PROPERTY UNITY_BUILD_BATCH_SIZE 32)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")

set(CUDA_MEMORY_DIR ${CMAKE_SOURCE_DIR}/include/cuda/memory)
set(CUDA_DEVICE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/device)
set(CUDA_ALGO_DIR ${CMAKE_SOURCE_DIR}/include/cuda/algo)
set(CUDA_LINALG_DIR ${CMAKE_SOURCE_DIR}/include/cuda/linalg)
set(CUDA_NUMERIC_DIR ${CMAKE_SOURCE_DIR}/include/cuda/numeric)
set(CUDA_SIGNAL_DIR ${CMAKE_SOURCE_DIR}/include/cuda/signal)
set(CUDA_API_DIR ${CMAKE_SOURCE_DIR}/include/cuda/api)
set(CUDA_SRC_DEVICE ${CMAKE_SOURCE_DIR}/src/cuda/device)
set(CUDA_SRC_ALGO ${CMAKE_SOURCE_DIR}/src/cuda/algo)

set(IMAGE_DIR ${CMAKE_SOURCE_DIR}/src/image)
set(PARALLEL_DIR ${CMAKE_SOURCE_DIR}/src/parallel)
set(MATRIX_DIR ${CMAKE_SOURCE_DIR}/src/matrix)
set(CONVOLUTION_DIR ${CMAKE_SOURCE_DIR}/src/convolution)
set(MEMORY_DIR ${CMAKE_SOURCE_DIR}/src/memory)
set(CUDA_FFT_DIR ${CMAKE_SOURCE_DIR}/src/cuda/fft)
set(CUDA_RAYTRACE_DIR ${CMAKE_SOURCE_DIR}/src/cuda/raytrace)
set(CUDA_GRAPH_DIR ${CMAKE_SOURCE_DIR}/src/cuda/graph)
set(CUDA_NEURAL_DIR ${CMAKE_SOURCE_DIR}/src/cuda/neural)
set(CUDA_MESH_DIR ${CMAKE_SOURCE_DIR}/include/cuda/mesh)
set(SRC_CUDA_MESH ${CMAKE_SOURCE_DIR}/src/cuda/mesh)
set(CUDA_DISTRIBUTED_DIR ${CMAKE_SOURCE_DIR}/include/cuda/distributed)
set(SRC_CUDA_DISTRIBUTED ${CMAKE_SOURCE_DIR}/src/cuda/distributed)
set(CUDA_NCCL_DIR ${CMAKE_SOURCE_DIR}/include/cuda/nccl)
set(SRC_CUDA_NCCL ${CMAKE_SOURCE_DIR}/src/cuda/nccl)
set(CUDA_PIPELINE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/pipeline)
set(SRC_CUDA_PIPELINE ${CMAKE_SOURCE_DIR}/src/cuda/pipeline)
set(CUDA_MPI_DIR ${CMAKE_SOURCE_DIR}/include/cuda/mpi)
set(SRC_CUDA_MPI ${CMAKE_SOURCE_DIR}/src/cuda/mpi)
set(CUDA_TOPOLOGY_DIR ${CMAKE_SOURCE_DIR}/include/cuda/topology)
set(SRC_CUDA_TOPOLOGY ${CMAKE_SOURCE_DIR}/src/cuda/topology)
set(CUDA_MULTINODE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/multinode)
set(SRC_CUDA_MULTINODE ${CMAKE_SOURCE_DIR}/src/cuda/multinode)
set(CUDA_CHECKPOINT_DIR ${CMAKE_SOURCE_DIR}/include/cuda/checkpoint)
set(SRC_CUDA_CHECKPOINT ${CMAKE_SOURCE_DIR}/src/cuda/checkpoint)
set(CUDA_COMM_DIR ${CMAKE_SOURCE_DIR}/include/cuda/comm)
set(SRC_CUDA_COMM ${CMAKE_SOURCE_DIR}/src/cuda/comm)
set(CUDA_MEMORY_ERROR_DIR ${CMAKE_SOURCE_DIR}/include/cuda/memory_error)
set(SRC_CUDA_MEMORY_ERROR ${CMAKE_SOURCE_DIR}/src/cuda/memory_error)
set(CUDA_PREEMPTION_DIR ${CMAKE_SOURCE_DIR}/include/cuda/preemption)
set(SRC_CUDA_PREEMPTION ${CMAKE_SOURCE_DIR}/src/cuda/preemption)
set(CUDA_SPARSE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/sparse)
set(SRC_CUDA_SPARSE ${CMAKE_SOURCE_DIR}/src/cuda/sparse)

set(CUDA_ERROR_DIR ${CMAKE_SOURCE_DIR}/include/cuda/error)
set(SRC_CUDA_ERROR ${CMAKE_SOURCE_DIR}/src/cuda/error)

add_library(cuda_memory INTERFACE)
target_include_directories(cuda_memory INTERFACE
        $<BUILD_INTERFACE:${CUDA_MEMORY_DIR}>
        $<BUILD_INTERFACE:${MEMORY_DIR}>
)
target_link_libraries(cuda_memory INTERFACE CUDA::cudart)

add_library(cuda_device INTERFACE)
target_include_directories(cuda_device INTERFACE
        $<BUILD_INTERFACE:${CUDA_DEVICE_DIR}>
        $<BUILD_INTERFACE:${CUDA_SRC_DEVICE}>
)
target_link_libraries(cuda_device INTERFACE CUDA::cudart cuda_memory)
set_target_properties(cuda_device PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

add_library(cuda_algo INTERFACE)
target_include_directories(cuda_algo INTERFACE
        $<BUILD_INTERFACE:${CUDA_ALGO_DIR}>
        $<BUILD_INTERFACE:${CUDA_DEVICE_DIR}>
        $<BUILD_INTERFACE:${CUDA_SRC_DEVICE}>
        $<BUILD_INTERFACE:${CUDA_MEMORY_DIR}>
)
target_link_libraries(cuda_algo INTERFACE
        cuda_device
        CUDA::cublas
        CUDA::cusparse
)

add_library(cuda_api INTERFACE)
target_include_directories(cuda_api INTERFACE
        $<BUILD_INTERFACE:${CUDA_API_DIR}>
        $<BUILD_INTERFACE:${CUDA_ALGO_DIR}>
)
target_link_libraries(cuda_api INTERFACE cuda_algo)

set(DEVICE_SOURCES
        ${CUDA_SRC_DEVICE}/reduce_kernels.cu
        ${CUDA_SRC_DEVICE}/cublas_context.cu
)

set(ALGO_SOURCES
        ${CUDA_SRC_ALGO}/reduce.cu
        ${CUDA_SRC_ALGO}/sort.cu
        ${CMAKE_SOURCE_DIR}/src/algo/segmented_sort.cpp
        ${CMAKE_SOURCE_DIR}/src/algo/spmv.cpp
        ${CMAKE_SOURCE_DIR}/src/algo/sample_sort.cpp
        ${CMAKE_SOURCE_DIR}/src/algo/sssp.cpp
)

set(SRC_CUDA_LINALG ${CMAKE_SOURCE_DIR}/src/cuda/linalg)
set(LINALG_SOURCES
        ${SRC_CUDA_LINALG}/linalg.cu
)

set(SRC_CUDA_NUMERIC ${CMAKE_SOURCE_DIR}/src/cuda/numeric)
set(NUMERIC_SOURCES
        ${SRC_CUDA_NUMERIC}/numeric.cu
)

set(SRC_CUDA_SIGNAL ${CMAKE_SOURCE_DIR}/src/cuda/signal)
set(SIGNAL_SOURCES
        ${SRC_CUDA_SIGNAL}/signal.cu
)

set(SPARSE_SOURCES
        ${SRC_CUDA_SPARSE}/cusparse_context.cpp
        ${SRC_CUDA_SPARSE}/sparse_ops.cu
)

set(SRC_CUDA_PRODUCTION ${CMAKE_SOURCE_DIR}/src/cuda/production)
set(PRODUCTION_SOURCES
        ${SRC_CUDA_PRODUCTION}/graph_executor.cu
        ${SRC_CUDA_PRODUCTION}/memory_node.cu
        ${SRC_CUDA_PRODUCTION}/algo_wrapper.cu
        ${SRC_CUDA_PRODUCTION}/l2_persistence.cu
        ${SRC_CUDA_PRODUCTION}/priority_stream.cu
        ${SRC_CUDA_PRODUCTION}/async_error_tracker.cu
        ${SRC_CUDA_PRODUCTION}/health_metrics.cu
        ${SRC_CUDA_PRODUCTION}/error_injection.cu
)

set(IMAGE_SOURCES
        ${IMAGE_DIR}/brightness.cu
        ${IMAGE_DIR}/gaussian_blur.cu
        ${IMAGE_DIR}/morphology.cu
        ${IMAGE_DIR}/sobel_edge.cu
        ${IMAGE_DIR}/types.cu
)

set(PARALLEL_SOURCES
        ${PARALLEL_DIR}/histogram.cu
        ${PARALLEL_DIR}/scan.cu
        ${PARALLEL_DIR}/sort.cu
)

set(MATRIX_SOURCES
        ${MATRIX_DIR}/add.cu
        ${MATRIX_DIR}/mult.cu
        ${MATRIX_DIR}/ops.cu
)

set(CONVOLUTION_SOURCES
        ${CONVOLUTION_DIR}/conv2d.cu
)

set(SRC_CUDA_MEMORY ${CMAKE_SOURCE_DIR}/src/cuda/memory)

set(MEMORY_SOURCES
        ${MEMORY_DIR}/buffer.cpp
        ${MEMORY_DIR}/memory_pool.cpp
        ${SRC_CUDA_MEMORY}/streaming_cache_manager.cpp
)

set(MEMORY_DISTRIBUTED_SOURCES
        ${SRC_CUDA_MEMORY}/distributed_pool.cpp
)

set(FFT_SOURCES
        ${CUDA_FFT_DIR}/fft.cu
)

set(RAYTRACE_SOURCES
        ${CUDA_RAYTRACE_DIR}/primitives.cu
        ${CUDA_RAYTRACE_DIR}/bvh.cu
)

set(GRAPH_SOURCES
        ${CUDA_GRAPH_DIR}/csr_graph.cu
        ${CUDA_GRAPH_DIR}/bfs.cu
        ${CUDA_GRAPH_DIR}/pagerank.cu
)

set(NEURAL_SOURCES
        ${CUDA_NEURAL_DIR}/matmul.cu
        ${CUDA_NEURAL_DIR}/softmax.cu
        ${CUDA_NEURAL_DIR}/activations.cu
        ${CUDA_NEURAL_DIR}/layer_norm.cu
        ${CUDA_NEURAL_DIR}/sync_batch_norm.cu
        ${CUDA_NEURAL_DIR}/tensor_parallel_matmul.cpp
        ${CUDA_NEURAL_DIR}/tensor_parallel_layers.cpp
        ${CUDA_NEURAL_DIR}/tensor_parallel_profile.cpp
        ${CUDA_NEURAL_DIR}/transformer/attention.cu
        ${CUDA_NEURAL_DIR}/loss/loss_functions.cu
        ${CUDA_NEURAL_DIR}/optimizers/optimizers.cpp
)

set(MESH_SOURCES
        ${SRC_CUDA_MESH}/device_mesh.cu
        ${SRC_CUDA_MESH}/peer_copy.cu
)

set(NCCL_SOURCES
        ${SRC_CUDA_NCCL}/nccl_context.cpp
        ${SRC_CUDA_NCCL}/nccl_error.cpp
        ${SRC_CUDA_NCCL}/nccl_validation.cpp
        ${SRC_CUDA_NCCL}/nccl_collective.cpp
        ${SRC_CUDA_NCCL}/nccl_all_reduce.cpp
        ${SRC_CUDA_NCCL}/nccl_broadcast.cpp
        ${SRC_CUDA_NCCL}/nccl_barrier.cpp
        ${SRC_CUDA_NCCL}/nccl_all_gather.cpp
        ${SRC_CUDA_NCCL}/nccl_reduce_scatter.cpp
        ${SRC_CUDA_NCCL}/nccl_group.cpp
        ${SRC_CUDA_NCCL}/nccl_ops.cpp
)

set(PIPELINE_SOURCES
        ${SRC_CUDA_PIPELINE}/pipeline_scheduler.cpp
        ${SRC_CUDA_PIPELINE}/p2p_primitives.cpp
        ${SRC_CUDA_PIPELINE}/activation_buffer.cpp
        ${SRC_CUDA_PIPELINE}/pipeline_comm.cpp
        ${SRC_CUDA_PIPELINE}/stage_balance.cpp
)

set(DISTRIBUTED_SOURCES
        ${SRC_CUDA_DISTRIBUTED}/common.cu
        ${SRC_CUDA_DISTRIBUTED}/reduce.cu
        ${SRC_CUDA_DISTRIBUTED}/broadcast.cu
        ${SRC_CUDA_DISTRIBUTED}/all_gather.cu
        ${SRC_CUDA_DISTRIBUTED}/barrier.cu
        ${SRC_CUDA_DISTRIBUTED}/matmul.cu
)

set(MPI_SOURCES
        ${SRC_CUDA_MPI}/mpi_context.cpp
)

set(TOPOLOGY_SOURCES
        ${SRC_CUDA_TOPOLOGY}/topology_map.cpp
)

set(MULTINODE_SOURCES
        ${SRC_CUDA_MULTINODE}/multi_node_context.cpp
)

set(CHECKPOINT_SOURCES
        ${SRC_CUDA_CHECKPOINT}/checkpoint_manager.cpp
)

set(COMM_SOURCES
        ${SRC_CUDA_COMM}/comm_error_recovery.cpp
)

set(MEMORY_ERROR_SOURCES
        ${SRC_CUDA_MEMORY_ERROR}/memory_error_handler.cpp
)

set(PREEMPTION_SOURCES
        ${SRC_CUDA_PREEMPTION}/preemption_handler.cpp
)

set(ERROR_SOURCES
        ${SRC_CUDA_ERROR}/cuda_error.cpp
        ${SRC_CUDA_ERROR}/cublas_error.cpp
        ${SRC_CUDA_ERROR}/timeout.cpp
        ${SRC_CUDA_ERROR}/timeout_context.cpp
        ${SRC_CUDA_ERROR}/retry.cpp
        ${SRC_CUDA_ERROR}/degrade.cpp
)

set(PERFORMANCE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/performance)
set(SRC_CUDA_PERFORMANCE ${CMAKE_SOURCE_DIR}/src/cuda/performance)

set(FUSION_DIR ${CMAKE_SOURCE_DIR}/include/cuda/neural/fusion)
set(SRC_CUDA_FUSION ${CMAKE_SOURCE_DIR}/src/cuda/neural/fusion)

set(FUSION_SOURCES
        ${SRC_CUDA_FUSION}/kernel_fusion.cu
        ${SRC_CUDA_FUSION}/fused_matmul_bias_act.cu
)

set(PERFORMANCE_SOURCES
        ${SRC_CUDA_PERFORMANCE}/profiler.cpp
        ${SRC_CUDA_PERFORMANCE}/autotuner.cpp
        ${SRC_CUDA_PERFORMANCE}/nvblox_metrics.cpp
        ${SRC_CUDA_PERFORMANCE}/kernel_profiler.cpp
        ${SRC_CUDA_PERFORMANCE}/metric_aggregators.cpp
        ${SRC_CUDA_PERFORMANCE}/fusion/kernel_fusion_analyzer.cpp
        ${SRC_CUDA_PERFORMANCE}/fusion/fusion_profitability.cpp
        ${SRC_CUDA_PERFORMANCE}/fusion/fusion_patterns.cpp
        ${SRC_CUDA_PERFORMANCE}/bandwidth/roofline_model.cpp
        ${SRC_CUDA_PERFORMANCE}/bandwidth/cache_analyzer.cpp
        ${SRC_CUDA_PERFORMANCE}/dashboard/dashboard_exporter.cpp
        ${SRC_CUDA_PERFORMANCE}/dashboard/flame_graph.cpp
)

set(MEMORY_OPT_DIR ${CMAKE_SOURCE_DIR}/include/cuda/memory_opt)
set(SRC_CUDA_MEMORY_OPT ${CMAKE_SOURCE_DIR}/src/cuda/memory_opt)

set(MEMORY_OPT_SOURCES
        ${SRC_CUDA_MEMORY_OPT}/memory_optimizer.cpp
)

set(INFERENCE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/inference)
set(SRC_CUDA_INFERENCE ${CMAKE_SOURCE_DIR}/src/cuda/inference)

set(INFERENCE_SOURCES
        ${SRC_CUDA_INFERENCE}/block_manager.cpp
        ${SRC_CUDA_INFERENCE}/beam_search.cpp
)

set(TOOLS_DIR ${CMAKE_SOURCE_DIR}/include/cuda/tools)
set(SRC_CUDA_TOOLS ${CMAKE_SOURCE_DIR}/src/cuda/tools)

set(TOOLS_SOURCES
        ${SRC_CUDA_TOOLS}/tools.cpp
)

set(TESTING_DIR ${CMAKE_SOURCE_DIR}/include/cuda/testing)
set(SRC_CUDA_TESTING ${CMAKE_SOURCE_DIR}/src/testing)

set(TESTING_SOURCES
        ${SRC_CUDA_TESTING}/memory_safety.cpp
        ${SRC_CUDA_TESTING}/test_isolation.cpp
        ${SRC_CUDA_TESTING}/layer_error_injection.cpp
        ${SRC_CUDA_TESTING}/boundary_testing.cpp
        ${SRC_CUDA_TESTING}/fp_determinism.cpp
        ${SRC_CUDA_TESTING}/integration.cpp
)

set(OBSERVABILITY_DIR ${CMAKE_SOURCE_DIR}/include/cuda/observability)
set(SRC_CUDA_OBSERVABILITY ${CMAKE_SOURCE_DIR}/src/observability)

set(OBSERVABILITY_SOURCES
        ${SRC_CUDA_OBSERVABILITY}/timeline.cpp
        ${SRC_CUDA_OBSERVABILITY}/bandwidth_tracker.cpp
        ${SRC_CUDA_OBSERVABILITY}/kernel_stats.cpp
        ${SRC_CUDA_OBSERVABILITY}/occupancy_analyzer.cpp
)

set(QUANTIZE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/quantize)
set(SRC_CUDA_QUANTIZE ${CMAKE_SOURCE_DIR}/src/cuda/quantize)

set(QUANTIZE_SOURCES
        ${SRC_CUDA_QUANTIZE}/fp8_kernels.cu
        ${SRC_CUDA_QUANTIZE}/fp8_gemm.cu
        ${SRC_CUDA_QUANTIZE}/int8_kernels.cu
        ${SRC_CUDA_QUANTIZE}/calibrator.cpp
        ${SRC_CUDA_QUANTIZE}/qat.cpp
)

set(ALL_CUDA_SOURCES
        ${DEVICE_SOURCES}
        ${ALGO_SOURCES}
        ${LINALG_SOURCES}
        ${NUMERIC_SOURCES}
        ${SIGNAL_SOURCES}
        ${PRODUCTION_SOURCES}
        ${IMAGE_SOURCES}
        ${PARALLEL_SOURCES}
        ${MATRIX_SOURCES}
        ${CONVOLUTION_SOURCES}
        ${MEMORY_SOURCES}
        ${MEMORY_DISTRIBUTED_SOURCES}
        ${FFT_SOURCES}
        ${RAYTRACE_SOURCES}
        ${GRAPH_SOURCES}
        ${NEURAL_SOURCES}
        ${MESH_SOURCES}
        ${NCCL_SOURCES}
        ${PIPELINE_SOURCES}
        ${DISTRIBUTED_SOURCES}
        ${PERFORMANCE_SOURCES}
        ${FUSION_SOURCES}
        ${MEMORY_OPT_SOURCES}
        ${INFERENCE_SOURCES}
        ${TOOLS_SOURCES}
        ${OBSERVABILITY_SOURCES}
        ${TESTING_SOURCES}
        ${ERROR_SOURCES}
        ${SPARSE_SOURCES}
        ${QUANTIZE_SOURCES}
)

add_library(cuda_nccl STATIC ${NCCL_SOURCES})
target_include_directories(cuda_nccl PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_NCCL_DIR}
)
target_link_libraries(cuda_nccl PUBLIC
        CUDA::cudart
        cuda_device
)
if(NCCL_FOUND AND TARGET NCCL::nccl)
        target_link_libraries(cuda_nccl PUBLIC NCCL::nccl)
        target_compile_definitions(cuda_nccl PUBLIC NOVA_NCCL_ENABLED=1)
else()
        target_compile_definitions(cuda_nccl PUBLIC NOVA_NCCL_ENABLED=0)
endif()

add_library(cuda_mpi STATIC ${MPI_SOURCES})
target_include_directories(cuda_mpi PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_MPI_DIR}
)
target_link_libraries(cuda_mpi PUBLIC CUDA::cudart)
if(MPI_FOUND AND TARGET MPI::MPI_CXX)
        target_link_libraries(cuda_mpi PUBLIC MPI::MPI_CXX)
        target_compile_definitions(cuda_mpi PUBLIC NOVA_MPI_ENABLED=1)
else()
        target_compile_definitions(cuda_mpi PUBLIC NOVA_MPI_ENABLED=0)
endif()

add_library(cuda_topology STATIC ${TOPOLOGY_SOURCES})
target_include_directories(cuda_topology PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_TOPOLOGY_DIR}
)
target_link_libraries(cuda_topology PUBLIC CUDA::cudart)
if(NCCL_FOUND AND TARGET NCCL::nccl)
        target_link_libraries(cuda_topology PUBLIC NCCL::nccl)
endif()

add_library(cuda_multinode STATIC ${MULTINODE_SOURCES})
target_include_directories(cuda_multinode PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_MULTINODE_DIR}
)
target_link_libraries(cuda_multinode PUBLIC CUDA::cudart cuda_nccl)

add_library(cuda_checkpoint STATIC ${CHECKPOINT_SOURCES})
target_include_directories(cuda_checkpoint PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_CHECKPOINT_DIR}
)
target_link_libraries(cuda_checkpoint PUBLIC
        CUDA::cudart
        OpenSSL::Crypto
)

add_library(cuda_comm STATIC ${COMM_SOURCES})
target_include_directories(cuda_comm PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_COMM_DIR}
)
target_link_libraries(cuda_comm PUBLIC
        CUDA::cudart
        Threads::Threads
)

add_library(cuda_memory_error STATIC ${MEMORY_ERROR_SOURCES})
target_include_directories(cuda_memory_error PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_MEMORY_ERROR_DIR}
)
target_link_libraries(cuda_memory_error PUBLIC
        CUDA::cudart
        Threads::Threads
)

add_library(cuda_preemption STATIC ${PREEMPTION_SOURCES})
target_include_directories(cuda_preemption PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_PREEMPTION_DIR}
)
target_link_libraries(cuda_preemption PUBLIC
        CUDA::cudart
        Threads::Threads
)

add_library(cuda_impl STATIC ${ALL_CUDA_SOURCES})
target_include_directories(cuda_impl PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_DEVICE_DIR}
        ${CUDA_ALGO_DIR}
        ${CUDA_LINALG_DIR}
        ${CUDA_NUMERIC_DIR}
        ${CUDA_SIGNAL_DIR}
        ${CUDA_MEMORY_DIR}
        ${CUDA_ERROR_DIR}
        ${CUDA_SPARSE_DIR}
)
target_link_libraries(cuda_impl PUBLIC
        cuda_device
        cuda_algo
        cuda_nccl
        cuda_mpi
        cuda_topology
        cuda_multinode
        cuda_checkpoint
        cuda_comm
        cuda_memory_error
        cuda_preemption
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::curand
        CUDA::cufft
)
set_target_properties(cuda_impl PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE cuda_impl cuda_api)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/data
        ${CUDA_DEVICE_DIR}
        ${CUDA_ALGO_DIR}
        ${CUDA_API_DIR}
        ${CUDA_MEMORY_DIR}
)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable parallel test execution
include(CTest)
include(ProcessorCount)
if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR)
    set(CMAKE_SYSTEM_PROCESSOR "unknown")
endif()
ProcessorCount(NCPU)
if(NCPU EQUAL 0)
    set(NCPU 4)
endif()
# GPU tests need more memory, cap parallel tests at 16
if(NCPU GREATER 16)
    set(TEST_PARALLEL_LEVEL 16)
else()
    set(TEST_PARALLEL_LEVEL ${NCPU})
endif()
set(CTEST_PARALLEL_LEVEL ${TEST_PARALLEL_LEVEL} CACHE STRING "Number of parallel test jobs")
message(STATUS "CPU cores: ${NCPU} | Build parallelism: ${NCPU} | Test parallelism: ${TEST_PARALLEL_LEVEL} (GPU memory constrained)")
message(STATUS "For faster builds, use: cmake -G Ninja -B build-ninja && cmake --build build-ninja --parallel ${NCPU}")

# =============================================================================
# CMake Package Export
# =============================================================================
# Enable find_package(nova) support

include(CMakePackageConfigHelpers)

# Create NovaTargets.cmake manually for the install tree
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/NovaTargets.cmake.in "
# NovaTargets.cmake - CMake targets for Nova library
# This file is generated by Nova's build system

if(TARGET Nova::cuda_impl)
    return()
endif()

add_library(Nova::cuda_impl STATIC IMPORTED)
set_target_properties(Nova::cuda_impl PROPERTIES
    IMPORTED_LOCATION \"@CMAKE_INSTALL_PREFIX@/lib/libcuda_impl.a\"
    INTERFACE_INCLUDE_DIRECTORIES \"@CMAKE_INSTALL_PREFIX@/include\"
)

add_library(Nova::cuda_device INTERFACE IMPORTED)
set_target_properties(Nova::cuda_device PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES \"@CMAKE_INSTALL_PREFIX@/include\"
)
add_dependencies(Nova::cuda_impl Nova::cuda_device)

add_library(Nova::cuda_algo INTERFACE IMPORTED)
set_target_properties(Nova::cuda_algo PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES \"@CMAKE_INSTALL_PREFIX@/include\"
)
add_dependencies(Nova::cuda_impl Nova::cuda_algo)
")

configure_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NovaTargets.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NovaTargets.cmake
    @ONLY
)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/NovaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/novaConfig.cmake
    INSTALL_DESTINATION lib/cmake/nova
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NovaConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/novaConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/NovaConfigVersion.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/NovaTargets.cmake
    DESTINATION lib/cmake/nova
)

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmark)
