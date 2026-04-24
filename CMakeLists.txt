cmake_minimum_required(VERSION 3.25)

project(nova
        VERSION 0.1.0
        DESCRIPTION "A CUDA parallel algorithms library with layered architecture"
        HOMEPAGE_URL "https://github.com/pplmx/nova"
        LANGUAGES CXX CUDA
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

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
)

set(MESH_SOURCES
        ${SRC_CUDA_MESH}/device_mesh.cu
        ${SRC_CUDA_MESH}/peer_copy.cu
)

set(DISTRIBUTED_SOURCES
        ${SRC_CUDA_DISTRIBUTED}/common.cu
        ${SRC_CUDA_DISTRIBUTED}/reduce.cu
        ${SRC_CUDA_DISTRIBUTED}/broadcast.cu
        ${SRC_CUDA_DISTRIBUTED}/all_gather.cu
        ${SRC_CUDA_DISTRIBUTED}/barrier.cu
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
        ${DISTRIBUTED_SOURCES}
)

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
  GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)
