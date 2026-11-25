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

# clangd, .vscode/c_cpp_properties.json and docs/ide-setup.md all look for
# compile_commands.json at the repository root, but CMake writes it into the
# binary directory. With more than one build tree (build, build-ninja,
# build-tidy, build-san, build-release, build-ci) there is no single directory
# to point an editor at. Symlink the active build tree's database to the root
# on every configure, so whichever tree was configured last is the one the
# editor sees. The root file is gitignored.
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(NOVA_COMPDB_LINK "${CMAKE_SOURCE_DIR}/compile_commands.json")
    if(EXISTS "${NOVA_COMPDB_LINK}" OR IS_SYMLINK "${NOVA_COMPDB_LINK}")
        file(REMOVE "${NOVA_COMPDB_LINK}")
    endif()
    file(CREATE_LINK
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${NOVA_COMPDB_LINK}"
        SYMBOLIC
        RESULT NOVA_COMPDB_RESULT
    )
    if(NOT NOVA_COMPDB_RESULT STREQUAL "0")
        message(STATUS "Could not link compile_commands.json to the source root: ${NOVA_COMPDB_RESULT}")
    endif()
endif()

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

# Examples, tests and benchmarks (optional)
#
# ON by default so nothing existing changes. Off lets a consumer build just the
# library and the `nova` demo, which is what the container image needs: those
# subdirectories pull dependencies at configure time (benchmark/CMakeLists.txt
# FetchContent's Google Benchmark, tests/CMakeLists.txt FetchContent's
# GoogleTest), so they cost a network fetch and build time even when only the
# `nova` target is requested.
option(NOVA_BUILD_TESTS_AND_BENCHMARKS
       "Build examples, tests and benchmarks" ON)

# Unity build for faster compilation (CMake 3.16+)
option(NOVA_ENABLE_UNITY_BUILD "Enable unity build to speed up compilation" ON)

# NVBlox support (optional)
# Enable NVIDIA Nsight Compute profiling integration
option(NOVA_ENABLE_NVBLOX "Enable NVBlox for kernel-level profiling" ON)

# ccache support for faster rebuilds
#
# ON by default: ccache is a pure cache in front of the compiler and produces
# the same object files, so the only effect of having it on without ccache
# installed is the message below. Turning it off used to be the default, which
# meant the option nobody flipped was the one that helps most.
option(NOVA_USE_CCACHE "Enable ccache (or sccache) compiler launcher for faster rebuilds" ON)

# mold linker support
#
# Link time is the part a compiler cache cannot help with: it reuses object
# files, but every link still runs the linker. mold parallelises that step.
#
# It is worth measuring before expecting much: on this project a relink goes
# from 224 ms to 105 ms, which is half the link step but well under 1% of a
# full build, because the build is compile-bound. It is on by default anyway
# because it is free when installed and scales with the number of targets.
#
# CMake 3.29 added LINKER_TYPE, and its MOLD value explicitly covers the
# NVIDIA compiler on Linux, so no toolchain-specific flag juggling is needed.
# CMAKE_LINKER_TYPE only affects targets defined after it is set, which is why
# this sits before every add_library/add_executable.
option(NOVA_USE_MOLD "Link with mold when it is available" ON)
option(NOVA_USE_LLD "Link with lld when it is available (used only without mold)" ON)

# Compiler warnings.
#
# The build enabled none, so the compiler reported nothing: no -Wall, no -Wextra, no
# -Werror anywhere in CMakeLists.txt or cmake/. Two defects found by inspection in
# earlier rounds -- a narrowing conversion to a launch parameter and an unused variable
# standing in for a wrong unit conversion -- would both have been reported by the
# compiler for free.
#
# The option defaults on. Turning it into -Werror is left to the caller
# (-DNOVA_WARNINGS_AS_ERRORS=ON) rather than being the default, because the two
# exemptions below are real and permanent for this tree: unused parameters are
# unavoidable in the NCCL stubs compiled when NOVA_ENABLE_NCCL=OFF, and the CUDA
# runtime still marks the shared-memory bank config calls deprecated even though a
# guard legitimately saves and restores them.
# Checkpoint compression codec. memory_optimizer.cpp selects a backend with
# -DNOVA_USE_ZSTD / -DNOVA_USE_LZ4, and until now nothing defined either macro, so the "compressed"
# path always fell through to the verbatim memcpy branch and has_compression_backend() was
# permanently false. The code was written as an optional capability; these options make it one.
#
# Both stay OFF by default: turning them on changes what compress() writes to disk, and a checkpoint
# produced with a codec must be read by a build with the same codec.
option(NOVA_USE_ZSTD "Compress checkpoints with zstd (requires libzstd)" OFF)
option(NOVA_USE_LZ4 "Compress checkpoints with lz4 (requires liblz4)" OFF)

option(NOVA_ENABLE_WARNINGS "Enable compiler warnings for host code" ON)
option(NOVA_WARNINGS_AS_ERRORS "Treat the enabled warnings as errors" OFF)

if(NOVA_ENABLE_WARNINGS AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    set(NOVA_CXX_WARNING_FLAGS
        -Wall
        -Wextra
        # Deliberate: NCCL is off by default and its files are then stubs whose
        # parameters exist only to match the enabled signature.
        -Wno-unused-parameter
        # cudaDeviceGet/SetSharedMemConfig: inert since Maxwell, still deprecated.
        -Wno-deprecated-declarations
    )
    if(NOVA_WARNINGS_AS_ERRORS)
        list(APPEND NOVA_CXX_WARNING_FLAGS -Werror)
    endif()
    # One flag per generator expression entry: a $<COMPILE_LANGUAGE:CXX> wrapping the
    # list expands to a single semicolon-joined argument and the compiler sees
    # "-Werror>" as one token.
    foreach(warning_flag IN LISTS NOVA_CXX_WARNING_FLAGS)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${warning_flag}>)
        # CUDA translation units compile host code through nvcc, which only forwards
        # flags to the host compiler via -Xcompiler. Without this the .cu files were the
        # one part of the tree the warnings never reached -- measured: enabling them
        # surfaced six more, all unused-result.
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${warning_flag}>)
    endforeach()
    message(STATUS "Compiler warnings: ${NOVA_CXX_WARNING_FLAGS}")
endif()


# Code coverage support
option(NOVA_COVERAGE "Enable code coverage instrumentation" OFF)
set(NOVA_COVERAGE_MIN "80" CACHE STRING "Minimum required line coverage percentage")

# The instrumentation flag depends on the compiler, because lcov/gcov (the
# only consumer scripts/coverage reads) understands gcov data, not LLVM
# profile data:
#
#   - GCC     -> --coverage, emits .gcno/.gcda that lcov reads directly.
#   - Clang   -> -fprofile-instr-generate -fcoverage-mapping would emit
#                .profraw, which needs llvm-profdata/llvm-cov and which the
#                lcov-based scripts cannot consume. Clang does support
#                --coverage too (it forwards to the same gcov format), so use
#                that to keep one pipeline.
#
# The previous code applied the Clang-only -fprofile-instr-generate to GCC as
# well, and GCC rejects it outright, so NOVA_COVERAGE=ON never configured with
# the GCC toolchain the CI uses.
if(NOVA_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Enabling gcov-compatible coverage instrumentation (${CMAKE_CXX_COMPILER_ID})")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=--coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    else()
        message(WARNING "Coverage requires Clang or GCC. Coverage disabled.")
        set(NOVA_COVERAGE OFF)
    endif()
endif()



if(NOVA_USE_CCACHE)
    # sccache is preferred over ccache even when both are installed, for a
    # CUDA-specific reason: sccache understands nvcc and caches the picc/ptxas
    # stages individually, while ccache cannot cache cicc and ptxas at all and
    # misses whenever cudafe++ names its temporary files differently. Measured
    # here, sccache reports 50% hits on CUDA compilations and 64% on CUBIN, and
    # turns a 76.8 s cold build of the nova target into 30.6 s. sccache is also
    # the one of the two that can share a cache between CI runners.
    find_program(SCCACHE_PROGRAM sccache)
    find_program(CCACHE_PROGRAM ccache)
    if(SCCACHE_PROGRAM)
        set(NOVA_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
        message(STATUS "Compiler cache: sccache (${SCCACHE_PROGRAM})")
    elseif(CCACHE_PROGRAM)
        set(NOVA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        message(STATUS "Compiler cache: ccache (${CCACHE_PROGRAM})")
    else()
        set(NOVA_COMPILER_LAUNCHER "")
        message(STATUS "Compiler cache: none (install ccache or sccache; the build is unchanged)")
    endif()

    if(NOVA_COMPILER_LAUNCHER)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${NOVA_COMPILER_LAUNCHER}" CACHE STRING "" FORCE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${NOVA_COMPILER_LAUNCHER}" CACHE STRING "" FORCE)
    endif()
endif()

# Pick a faster linker when one is installed. Each candidate is accepted only
# if the compiler in use can actually drive it: CMAKE_LINKER_TYPE assumes the
# requested linker is fully compatible with the default one and does no option
# translation, so a value the toolchain rejects would fail at link time rather
# than at configure time.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Two spellings per candidate, and they are not interchangeable:
    # CMAKE_LINKER_TYPE wants the uppercase CMake enum (MOLD, LLD), while the
    # compiler flag wants the lowercase driver name (-fuse-ld=mold). Checking
    # the enum spelling with -fuse-ld= reports "failed" even when the linker
    # is installed, so the mapping is explicit here.
    set(NOVA_LINKER_ENUM "")
    set(NOVA_LINKER_FLAG "")
    if(NOVA_USE_MOLD)
        set(NOVA_LINKER_ENUM "MOLD")
        set(NOVA_LINKER_FLAG "mold")
    endif()
    if(NOT NOVA_LINKER_ENUM AND NOVA_USE_LLD)
        set(NOVA_LINKER_ENUM "LLD")
        set(NOVA_LINKER_FLAG "lld")
    endif()

    if(NOVA_LINKER_ENUM)
        include(CheckLinkerFlag)
        check_linker_flag(CXX "-fuse-ld=${NOVA_LINKER_FLAG}" NOVA_LINKER_${NOVA_LINKER_ENUM}_OK)
        if(NOVA_LINKER_${NOVA_LINKER_ENUM}_OK)
            set(CMAKE_LINKER_TYPE "${NOVA_LINKER_ENUM}")
            message(STATUS "Linker: ${NOVA_LINKER_FLAG}")
        else()
            message(STATUS "Linker: ${NOVA_LINKER_FLAG} not usable by this compiler, keeping the default")
        endif()
    else()
        message(STATUS "Linker: default for this toolchain")
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

# Mark CUDA translation units as device-reachable. nvcc stops at C++20 while the
# host build is C++23, so a header that relies on C++23 must refuse to be
# included here rather than fail later inside some unrelated .cu file. See
# include/cuda/cpp_level.hpp.
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNOVA_DEVICE_REACHABLE=1")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-DNOVA_DEVICE_REACHABLE=1")
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
        ${CMAKE_SOURCE_DIR}/src/algo/spmv.cu
        ${CMAKE_SOURCE_DIR}/src/algo/segmented_sort.cu
        ${CMAKE_SOURCE_DIR}/src/algo/sample_sort.cu
        ${CMAKE_SOURCE_DIR}/src/algo/sssp.cu
        ${CMAKE_SOURCE_DIR}/src/algo/flash_attention.cpp
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
        ${CUDA_NEURAL_DIR}/tensor_parallel_attention.cpp
        ${CUDA_NEURAL_DIR}/attention_kernels.cu
        ${CUDA_NEURAL_DIR}/tensor_parallel_block.cpp
        ${CUDA_NEURAL_DIR}/tensor_parallel_profile.cpp
        ${CUDA_NEURAL_DIR}/transformer/attention.cu
        ${CUDA_NEURAL_DIR}/loss/loss_functions.cu
        ${CUDA_NEURAL_DIR}/optimizers/optimizers.cpp
        ${CUDA_NEURAL_DIR}/optimizers/optimizers_kernels.cu
        ${CUDA_NEURAL_DIR}/training.cpp
        ${CUDA_NEURAL_DIR}/training_transformer.cpp
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
        ${SRC_CUDA_DISTRIBUTED}/ring_attention.cu
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
        ${SRC_CUDA_INFERENCE}/speculative_decoding.cpp
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
        ${SRC_CUDA_QUANTIZE}/fp8_activation.cu
        ${SRC_CUDA_QUANTIZE}/fp8_gemm.cu
        ${SRC_CUDA_QUANTIZE}/int8_kernels.cu
        ${SRC_CUDA_QUANTIZE}/calibrator.cpp
        ${SRC_CUDA_QUANTIZE}/qat.cpp
        ${SRC_CUDA_QUANTIZE}/benchmark.cpp
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
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CUDA_NCCL_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(cuda_nccl PUBLIC
        CUDA::cudart
        cuda_device
)
# The documented -DNOVA_ENABLE_NCCL=OFF switch (see option() above) must
# actually disable NCCL: before this gate the compile definition was keyed only
# off library discovery, so the option was a no-op. When enabled,
# keep the existing auto-detection (the NCCL_FOUND/NCCL::nccl branch is only
# reached if cmake/FindNCCL.cmake is ever invoked — it is NOT today, so the
# find_library fallback does the detection, which is why FindNCCL is not
# activated as-is). The else() branch makes the OFF switch real.
if(NOVA_ENABLE_NCCL)
        if(NCCL_FOUND AND TARGET NCCL::nccl)
                target_link_libraries(cuda_nccl PUBLIC NCCL::nccl)
                target_compile_definitions(cuda_nccl PUBLIC NOVA_NCCL_ENABLED=1)
        else()
                find_library(NCCL_LIB nccl)
                if(NCCL_LIB)
                        target_link_libraries(cuda_nccl PUBLIC ${NCCL_LIB})
                        target_compile_definitions(cuda_nccl PUBLIC NOVA_NCCL_ENABLED=1)
                else()
                        target_compile_definitions(cuda_nccl PUBLIC NOVA_NCCL_ENABLED=0)
                endif()
        endif()
else()
        message(STATUS "Nova: NCCL explicitly disabled (NOVA_ENABLE_NCCL=OFF)")
        target_compile_definitions(cuda_nccl PUBLIC NOVA_NCCL_ENABLED=0)
endif()

add_library(cuda_mpi STATIC ${MPI_SOURCES})
target_include_directories(cuda_mpi PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CUDA_MPI_DIR}>
        $<INSTALL_INTERFACE:include>
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
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CUDA_TOPOLOGY_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(cuda_topology PUBLIC CUDA::cudart)
if(NCCL_FOUND AND TARGET NCCL::nccl)
        target_link_libraries(cuda_topology PUBLIC NCCL::nccl)
endif()

add_library(cuda_multinode STATIC ${MULTINODE_SOURCES})
target_include_directories(cuda_multinode PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CUDA_MULTINODE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(cuda_multinode PUBLIC CUDA::cudart cuda_nccl)

add_library(cuda_checkpoint STATIC ${CHECKPOINT_SOURCES})
target_include_directories(cuda_checkpoint PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CUDA_CHECKPOINT_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(cuda_checkpoint PUBLIC
        CUDA::cudart
        OpenSSL::Crypto
)

add_library(cuda_comm STATIC ${COMM_SOURCES})
target_include_directories(cuda_comm PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CUDA_COMM_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(cuda_comm PUBLIC
        CUDA::cudart
        Threads::Threads
        # comm_error_recovery.cpp switches its whole NCCL error classification
        # on NOVA_NCCL_ENABLED, which cuda_nccl exports. Without this link the
        # macro was undefined in that translation unit, so #if treated it as 0
        # and classify_from_nccl returned Unknown for every NCCL status.
        cuda_nccl
)

add_library(cuda_memory_error STATIC ${MEMORY_ERROR_SOURCES})
target_include_directories(cuda_memory_error PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CUDA_MEMORY_ERROR_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(cuda_memory_error PUBLIC
        CUDA::cudart
        Threads::Threads
)

add_library(cuda_preemption STATIC ${PREEMPTION_SOURCES})
target_include_directories(cuda_preemption PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CUDA_PREEMPTION_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(cuda_preemption PUBLIC
        CUDA::cudart
        Threads::Threads
)

add_library(cuda_impl STATIC ${ALL_CUDA_SOURCES})

# Resolve the codec selected above. A missing library is an error when the option was asked for, not
# a silent fallback: quietly dropping back to verbatim storage would leave a caller who enabled
# compression with uncompressed checkpoints and no indication.
if(NOVA_USE_ZSTD AND NOVA_USE_LZ4)
        message(FATAL_ERROR "NOVA_USE_ZSTD and NOVA_USE_LZ4 are mutually exclusive; set at most one")
endif()
if(NOVA_USE_ZSTD)
        find_path(ZSTD_INCLUDE_DIR zstd.h)
        find_library(ZSTD_LIBRARY NAMES zstd)
        if(NOT ZSTD_INCLUDE_DIR OR NOT ZSTD_LIBRARY)
                message(FATAL_ERROR "NOVA_USE_ZSTD=ON but zstd was not found (need zstd.h and libzstd)")
        endif()
        target_include_directories(cuda_impl PRIVATE ${ZSTD_INCLUDE_DIR})
        target_link_libraries(cuda_impl PRIVATE ${ZSTD_LIBRARY})
        target_compile_definitions(cuda_impl PRIVATE NOVA_USE_ZSTD=1)
elseif(NOVA_USE_LZ4)
        find_path(LZ4_INCLUDE_DIR lz4.h)
        find_library(LZ4_LIBRARY NAMES lz4)
        if(NOT LZ4_INCLUDE_DIR OR NOT LZ4_LIBRARY)
                message(FATAL_ERROR "NOVA_USE_LZ4=ON but lz4 was not found (need lz4.h and liblz4)")
        endif()
        target_include_directories(cuda_impl PRIVATE ${LZ4_INCLUDE_DIR})
        target_link_libraries(cuda_impl PRIVATE ${LZ4_LIBRARY})
        target_compile_definitions(cuda_impl PRIVATE NOVA_USE_LZ4=1)
endif()
# BUILD_INTERFACE for the source-tree paths, INSTALL_INTERFACE for the installed
# tree. Without the split, exporting the target fails with "INTERFACE_INCLUDE_
# DIRECTORIES property contains path ... which is prefixed in the source
# directory", which is why the export was hand-written in the first place.
#
# The sub-directories under include/cuda are listed for the build tree only: the
# install copies include/ wholesale, so a consumer includes "cuda/..." there.
target_include_directories(cuda_impl PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CUDA_DEVICE_DIR}>
        $<BUILD_INTERFACE:${CUDA_ALGO_DIR}>
        $<BUILD_INTERFACE:${CUDA_LINALG_DIR}>
        $<BUILD_INTERFACE:${CUDA_NUMERIC_DIR}>
        $<BUILD_INTERFACE:${CUDA_SIGNAL_DIR}>
        $<BUILD_INTERFACE:${CUDA_MEMORY_DIR}>
        $<BUILD_INTERFACE:${CUDA_ERROR_DIR}>
        $<BUILD_INTERFACE:${CUDA_SPARSE_DIR}>
        $<INSTALL_INTERFACE:include>
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

# GoogleTest is only needed by examples/tests/benchmarks. Guarding it with the
# same option keeps a library-only configure from cloning it and building it.
if(NOVA_BUILD_TESTS_AND_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.17.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

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

# The exported target file used to be written by hand into the build tree, and it
# described artifacts the install never placed:
#
#   add_library(Nova::cuda_impl STATIC IMPORTED)
#   set_target_properties(... IMPORTED_LOCATION "@CMAKE_INSTALL_PREFIX@/lib/libcuda_impl.a")
#
# but nothing installed libcuda_impl.a or include/, so `cmake --install` produced
# only GoogleTest and a package config that pointed into an empty prefix. A consumer
# doing find_package(nova) then failed at link time with a missing library.
#
# The library, its public headers and the targets are installed and exported here,
# so the config file refers to files that the same install writes.
install(TARGETS cuda_impl cuda_api cuda_device cuda_algo cuda_nccl cuda_memory
        cuda_checkpoint cuda_comm cuda_memory_error cuda_mpi cuda_multinode
        cuda_preemption cuda_topology
        EXPORT NovaTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT NovaTargets
        FILE NovaTargets.cmake
        NAMESPACE Nova::
        DESTINATION lib/cmake/nova)

# What the installed package config should report. Derived from the same facts the
# build uses, recorded as a variable because the compile definitions above are
# target properties and invisible to configure_file().
set(NOVA_NCCL_ENABLED_AT_CONFIGURE OFF)
if(NOVA_ENABLE_NCCL)
    if((NCCL_FOUND AND TARGET NCCL::nccl) OR NCCL_LIB)
        set(NOVA_NCCL_ENABLED_AT_CONFIGURE ON)
    endif()
endif()
set(NOVA_MPI_ENABLED_AT_CONFIGURE OFF)
if(NOVA_ENABLE_MPI AND MPI_CXX_FOUND)
    set(NOVA_MPI_ENABLED_AT_CONFIGURE ON)
endif()

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

# NovaTargets.cmake is written by install(EXPORT ...) above; only the config and
# version files are listed here. Listing the generated target file too made
# `cmake --install` fail at the last step with "file INSTALL cannot find
# .../NovaTargets.cmake", after everything else had already been copied.
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/novaConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/NovaConfigVersion.cmake
    DESTINATION lib/cmake/nova
)

if(NOVA_BUILD_TESTS_AND_BENCHMARKS)
    enable_testing()
    add_subdirectory(examples)
    add_subdirectory(tests)
    add_subdirectory(benchmark)
endif()
