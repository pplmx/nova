cmake_minimum_required(VERSION 3.25)

project(cuda-samples
        VERSION 0.1.0
        DESCRIPTION "A CUDA parallel algorithms library with layered architecture"
        HOMEPAGE_URL "https://github.com/pplmx/cuda-samples"
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

set(CUDA_KERNEL_DIR ${CMAKE_SOURCE_DIR}/include/cuda/kernel)
set(CUDA_ALGO_DIR ${CMAKE_SOURCE_DIR}/include/cuda/algo)
set(CUDA_API_DIR ${CMAKE_SOURCE_DIR}/include/cuda/api)
set(CUDA_SRC_KERNEL ${CMAKE_SOURCE_DIR}/src/cuda/kernel)
set(CUDA_SRC_ALGO ${CMAKE_SOURCE_DIR}/src/cuda/algo)

set(IMAGE_DIR ${CMAKE_SOURCE_DIR}/src/image)
set(PARALLEL_DIR ${CMAKE_SOURCE_DIR}/src/parallel)
set(MATRIX_DIR ${CMAKE_SOURCE_DIR}/src/matrix)
set(CONVOLUTION_DIR ${CMAKE_SOURCE_DIR}/src/convolution)
set(MEMORY_DIR ${CMAKE_SOURCE_DIR}/src/memory)

add_library(cuda_kernel INTERFACE)
target_include_directories(cuda_kernel INTERFACE
        $<BUILD_INTERFACE:${CUDA_KERNEL_DIR}>
        $<BUILD_INTERFACE:${CUDA_SRC_KERNEL}>
)
target_link_libraries(cuda_kernel INTERFACE CUDA::cudart)
set_target_properties(cuda_kernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

add_library(cuda_algo INTERFACE)
target_include_directories(cuda_algo INTERFACE
        $<BUILD_INTERFACE:${CUDA_ALGO_DIR}>
        $<BUILD_INTERFACE:${CUDA_KERNEL_DIR}>
        $<BUILD_INTERFACE:${CUDA_SRC_KERNEL}>
)
target_link_libraries(cuda_algo INTERFACE
        cuda_kernel
        CUDA::cublas
)

add_library(cuda_api INTERFACE)
target_include_directories(cuda_api INTERFACE
        $<BUILD_INTERFACE:${CUDA_API_DIR}>
        $<BUILD_INTERFACE:${CUDA_ALGO_DIR}>
        $<BUILD_INTERFACE:${CUDA_KERNEL_DIR}>
)
target_link_libraries(cuda_api INTERFACE cuda_algo)

file(GLOB_RECURSE KERNEL_SOURCES CONFIGURE_DEPENDS
        ${CUDA_SRC_KERNEL}/*.cu
)

file(GLOB_RECURSE ALGO_SOURCES CONFIGURE_DEPENDS
        ${CUDA_SRC_ALGO}/*.cpp
)

file(GLOB_RECURSE IMAGE_SOURCES CONFIGURE_DEPENDS
        ${IMAGE_DIR}/*.cu
)

file(GLOB_RECURSE PARALLEL_SOURCES CONFIGURE_DEPENDS
        ${PARALLEL_DIR}/*.cu
)

file(GLOB_RECURSE MATRIX_SOURCES CONFIGURE_DEPENDS
        ${MATRIX_DIR}/*.cu
)

file(GLOB_RECURSE CONVOLUTION_SOURCES CONFIGURE_DEPENDS
        ${CONVOLUTION_DIR}/*.cu
)

file(GLOB_RECURSE MEMORY_SOURCES CONFIGURE_DEPENDS
        ${MEMORY_DIR}/*.cpp
)

set(ALL_CUDA_SOURCES
        ${KERNEL_SOURCES}
        ${ALGO_SOURCES}
        ${IMAGE_SOURCES}
        ${PARALLEL_SOURCES}
        ${MATRIX_SOURCES}
        ${CONVOLUTION_SOURCES}
        ${MEMORY_SOURCES}
)

add_library(cuda_impl STATIC ${ALL_CUDA_SOURCES})
target_include_directories(cuda_impl PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_KERNEL_DIR}
        ${CUDA_ALGO_DIR}
)
target_link_libraries(cuda_impl PUBLIC
        cuda_kernel
        cuda_algo
        CUDA::cudart
        CUDA::cublas
)
set_target_properties(cuda_impl PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE cuda_impl cuda_api)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/data
        ${CUDA_KERNEL_DIR}
        ${CUDA_ALGO_DIR}
        ${CUDA_API_DIR}
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
