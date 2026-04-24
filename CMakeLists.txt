cmake_minimum_required(VERSION 4.0)

project(nova
        VERSION 0.1.0
        DESCRIPTION "A CUDA parallel algorithms library with layered architecture"
        HOMEPAGE_URL "https://github.com/pplmx/nova"
        LANGUAGES CXX CUDA
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

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

# NCCL support (optional)
# Enable NCCL for optimized multi-GPU collectives with P2P fallback when unavailable
option(NOVA_ENABLE_NCCL "Enable NCCL for optimized multi-GPU collectives" ON)

# MPI support (optional)
# Enable multi-node training with MPI-based NCCL bootstrapping
option(NOVA_ENABLE_MPI "Enable MPI for multi-node support" OFF)

if(NOVA_ENABLE_MPI)
    find_package(MPI 3.1 REQUIRED)
    if(MPI_FOUND)
        message(STATUS "MPI ${MPI_VERSION} found -- enabling multi-node support")
        set(NOVA_MPI_ENABLED TRUE)
    else()
        message(FATAL_ERROR "MPI 3.1+ required when NOVA_ENABLE_MPI=ON")
    endif()
else()
    message(STATUS "MPI disabled by user option (multi-node not needed)")
    set(NOVA_MPI_ENABLED FALSE)
endif()

if(NOVA_ENABLE_NCCL)
    find_package(NCCL 2.25 QUIET)
    if(NCCL_FOUND)
        message(STATUS "NCCL ${NCCL_VERSION} found -- enabling NCCL support")
        set(NOVA_NCCL_ENABLED TRUE)
    else()
        message(STATUS "NCCL not found -- using P2P fallback for multi-GPU")
        set(NOVA_NCCL_ENABLED FALSE)
    endif()
else()
    message(STATUS "NCCL disabled by user option")
    set(NOVA_NCCL_ENABLED FALSE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")

set(CUDA_MEMORY_DIR ${CMAKE_SOURCE_DIR}/include/cuda/memory)
set(CUDA_DEVICE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/device)
set(CUDA_ALGO_DIR ${CMAKE_SOURCE_DIR}/include/cuda/algo)
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
        ${CUDA_NEURAL_DIR}/tensor_parallel_matmul.cpp
        ${CUDA_NEURAL_DIR}/tensor_parallel_layers.cpp
        ${CUDA_NEURAL_DIR}/tensor_parallel_profile.cpp
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

set(ALL_CUDA_SOURCES
        ${DEVICE_SOURCES}
        ${ALGO_SOURCES}
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
if(NCCL_FOUND)
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
if(MPI_FOUND)
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
if(NCCL_FOUND)
        target_link_libraries(cuda_topology PUBLIC NCCL::nccl)
endif()

add_library(cuda_multinode STATIC ${MULTINODE_SOURCES})
target_include_directories(cuda_multinode PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_MULTINODE_DIR}
)
target_link_libraries(cuda_multinode PUBLIC CUDA::cudart cuda_nccl)

add_library(cuda_impl STATIC ${ALL_CUDA_SOURCES})
target_include_directories(cuda_impl PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_DEVICE_DIR}
        ${CUDA_ALGO_DIR}
        ${CUDA_MEMORY_DIR}
)
target_link_libraries(cuda_impl PUBLIC
        cuda_device
        cuda_algo
        cuda_nccl
        cuda_mpi
        cuda_topology
        cuda_multinode
        CUDA::cudart
        CUDA::cublas
        CUDA::cufft
)
set_target_properties(cuda_impl PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
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
set(CTEST_PARALLEL_LEVEL 4 CACHE STRING "Number of parallel test jobs")

enable_testing()
add_subdirectory(tests)
