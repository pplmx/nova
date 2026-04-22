# CUDA Architecture Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform flat CUDA samples into a production-ready layered architecture with kernel/algo/api separation

**Architecture:** Three-layer architecture following cuBLAS/cuDNN patterns:
- Layer 1 (kernel): Pure device kernels, no memory management
- Layer 2 (algo): Algorithm wrappers with memory management
- Layer 3 (api): STL-style high-level API

**Tech Stack:** CMake 3.25+, CUDA 17+, GoogleTest

---

## File Structure Overview

```
include/cuda/
├── kernel/
│   ├── reduce.h          # kernel declarations
│   ├── scan.h
│   ├── sort.h
│   └── cuda_utils.h      # shared kernel utilities
├── algo/
│   ├── reduce.h          # algorithm wrapper
│   ├── scan.h
│   ├── sort.h
│   └── device_buffer.h   # RAII device memory
└── api/
    └── device_vector.h   # high-level container

src/
├── kernel/
│   ├── reduce.cu
│   ├── scan.cu
│   └── sort.cu
├── algo/
│   ├── reduce.cpp
│   ├── scan.cpp
│   └── sort.cpp
├── benchmark/
│   └── timer.h
└── main.cpp              # demo

tests/
├── unit/
│   └── kernel_reduce_test.cu
└── integration/
    └── algo_reduce_test.cu
```

---

## Task 1: Create Directory Structure

**Files:**
- Create: `include/cuda/kernel/`
- Create: `include/cuda/algo/`
- Create: `include/cuda/api/`
- Create: `src/kernel/`
- Create: `src/algo/`
- Create: `src/benchmark/`
- Create: `tests/unit/`
- Create: `tests/integration/`

- [ ] **Step 1: Create all directories**

```bash
mkdir -p include/cuda/kernel include/cuda/algo include/cuda/api
mkdir -p src/kernel src/algo src/benchmark
mkdir -p tests/unit tests/integration
```

Run: `find include/cuda src/kernel src/algo tests -type d | sort`
Expected: All 8 directories created

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "chore: create layered directory structure"
```

---

## Task 2: Refactor CMake for Modern Architecture

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write new CMakeLists.txt with INTERFACE libraries**

```cmake
cmake_minimum_required(VERSION 3.25)

project(cuda-samples
        VERSION 0.0.1
        DESCRIPTION "A CUDA parallel algorithms library"
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
set(CUDA_SRC_KERNEL ${CMAKE_SOURCE_DIR}/src/kernel)
set(CUDA_SRC_ALGO ${CMAKE_SOURCE_DIR}/src/algo)

# Layer 1: cuda_kernel (INTERFACE library)
add_library(cuda_kernel INTERFACE)
target_include_directories(cuda_kernel INTERFACE
        $<BUILD_INTERFACE:${CUDA_KERNEL_DIR}>
        $<BUILD_INTERFACE:${CUDA_SRC_KERNEL}>
)
target_link_libraries(cuda_kernel INTERFACE CUDA::cudart)
set_target_properties(cuda_kernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Layer 2: cuda_algo (depends on cuda_kernel)
add_library(cuda_algo INTERFACE)
target_include_directories(cuda_algo INTERFACE
        $<BUILD_INTERFACE:${CUDA_ALGO_DIR}>
        $<BUILD_INTERFACE:${CUDA_KERNEL_DIR}>
)
target_link_libraries(cuda_algo INTERFACE
        cuda_kernel
        CUDA::cublas
)

# Layer 3: cuda_api (depends on cuda_algo)
add_library(cuda_api INTERFACE)
target_include_directories(cuda_api INTERFACE
        $<BUILD_INTERFACE:${CUDA_API_DIR}>
        $<BUILD_INTERFACE:${CUDA_ALGO_DIR}>
)
target_link_libraries(cuda_api INTERFACE cuda_algo)

# Demo executable
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE cuda_api)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/data)

# GoogleTest
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
```

- [ ] **Step 2: Run CMake configuration**

Run: `cmake -S . -B build`
Expected: Configuration successful, no errors

- [ ] **Step 3: Verify CMake cache**

Run: `grep -E "^cuda_kernel|cuda_algo|cuda_api" build/CMakeCache.txt`
Expected: Target definitions present

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt && git commit -m "build: modernize CMake with layered INTERFACE libraries"
```

---

## Task 3: Create Layer 1 - Kernel Utilities

**Files:**
- Create: `include/cuda/kernel/cuda_utils.h`

- [ ] **Step 1: Write cuda_utils.h with kernel utilities**

```cpp
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace cuda::kernel {

enum class ReduceOp { SUM, MAX, MIN };

constexpr int WARP_SIZE = 32;

template<typename T>
__device__ T warp_reduce(T val, ReduceOp op) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        if (op == ReduceOp::SUM) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        } else if (op == ReduceOp::MAX) {
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        } else {
            val = min(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    return val;
}

} // namespace cuda::kernel
```

- [ ] **Step 2: Verify file compiles**

Run: `nvcc -c -std=c++17 include/cuda/kernel/cuda_utils.h -o /dev/null 2>&1 | head -20`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git add include/cuda/kernel/cuda_utils.h && git commit -m "feat: add Layer 1 kernel utilities"
```

---

## Task 4: Create Layer 1 - Reduce Kernel

**Files:**
- Create: `include/cuda/kernel/reduce.h`
- Create: `src/kernel/reduce.cu`

- [ ] **Step 1: Write reduce kernel header**

```cpp
#pragma once

#include "cuda_utils.h"
#include <cstddef>

namespace cuda::kernel {

template<typename T>
__global__ void reduce_basic_kernel(const T* input, T* output, size_t size, ReduceOp op);

template<typename T>
__global__ void reduce_optimized_kernel(const T* input, T* output, size_t size, ReduceOp op);

} // namespace cuda::kernel
```

- [ ] **Step 2: Write reduce kernel implementation**

```cpp
#include "cuda_utils.h"

namespace cuda::kernel {

template<typename T>
__global__ void reduce_basic_kernel(const T* input, T* output, size_t size, ReduceOp op) {
    __shared__ T sdata[256];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T val = 0;
    if (i < size) val = input[i];
    if (i + blockDim.x < size) {
        if (op == ReduceOp::SUM) {
            val += input[i + blockDim.x];
        } else if (op == ReduceOp::MAX) {
            val = max(val, input[i + blockDim.x]);
        } else {
            val = min(val, input[i + blockDim.x]);
        }
    }
    sdata[tid] = val;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (op == ReduceOp::SUM) {
                sdata[tid] += sdata[tid + s];
            } else if (op == ReduceOp::MAX) {
                sdata[tid] = max(sdata[tid], sdata[tid + s]);
            } else {
                sdata[tid] = min(sdata[tid], sdata[tid + s]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

template<typename T>
__global__ void reduce_optimized_kernel(const T* input, T* output, size_t size, ReduceOp op) {
    __shared__ T sdata[32];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T val = 0;
    if (i < size) val = input[i];
    if (i + blockDim.x < size) {
        if (op == ReduceOp::SUM) {
            val += input[i + blockDim.x];
        } else if (op == ReduceOp::MAX) {
            val = max(val, input[i + blockDim.x]);
        } else {
            val = min(val, input[i + blockDim.x]);
        }
    }

    val = warp_reduce(val, op);

    if (tid % WARP_SIZE == 0) sdata[tid / WARP_SIZE] = val;
    __syncthreads();

    if (tid < WARP_SIZE) val = (tid < blockDim.x / WARP_SIZE) ? sdata[tid] : T{};
    val = warp_reduce(val, op);

    if (tid == 0) output[blockIdx.x] = val;
}

#define REDUCE_KERNEL_INSTANTIATE(T)                                                       \
    template __global__ void reduce_basic_kernel<T>(const T*, T*, size_t, ReduceOp);       \
    template __global__ void reduce_optimized_kernel<T>(const T*, T*, size_t, ReduceOp);

REDUCE_KERNEL_INSTANTIATE(int)
REDUCE_KERNEL_INSTANTIATE(float)
REDUCE_KERNEL_INSTANTIATE(double)
REDUCE_KERNEL_INSTANTIATE(unsigned int)

} // namespace cuda::kernel
```

- [ ] **Step 3: Build kernel library**

Run: `cmake --build build --target cuda_kernel 2>&1 | tail -10`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add include/cuda/kernel/reduce.h src/kernel/reduce.cu && git commit -m "feat: add Layer 1 reduce kernel"
```

---

## Task 5: Create Layer 2 - Device Buffer (RAII Memory)

**Files:**
- Create: `include/cuda/algo/device_buffer.h`

- [ ] **Step 1: Write device_buffer.h**

```cpp
#pragma once

#include "cuda_utils.h"
#include <memory>
#include <cstddef>

namespace cuda::algo {

template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : size_(count) {
        CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }

    void copy_from(const T* host_data, size_t count) {
        CUDA_CHECK(cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to(T* host_data, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T* data_ = nullptr;
    size_t size_ = 0;
};

} // namespace cuda::algo
```

- [ ] **Step 2: Verify compilation**

Run: `nvcc -c -std=c++17 include/cuda/algo/device_buffer.h -o /dev/null 2>&1 | head -20`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git add include/cuda/algo/device_buffer.h && git commit -m "feat: add DeviceBuffer RAII wrapper for Layer 2"
```

---

## Task 6: Create Layer 2 - Reduce Algorithm Wrapper

**Files:**
- Create: `include/cuda/algo/reduce.h`
- Create: `src/algo/reduce.cpp`

- [ ] **Step 1: Write reduce algorithm header**

```cpp
#pragma once

#include "device_buffer.h"
#include "reduce.h"
#include <cstddef>

namespace cuda::algo {

template<typename T>
T reduce_sum(const T* input, size_t size);

template<typename T>
T reduce_sum_optimized(const T* input, size_t size);

template<typename T>
T reduce_max(const T* input, size_t size);

template<typename T>
T reduce_min(const T* input, size_t size);

} // namespace cuda::algo
```

Note: This references `reduce.h` which we'll create as forwarding header later. For now, create a minimal version.

- [ ] **Step 2: Write reduce.cpp with wrapper implementation**

```cpp
#include "reduce.h"
#include "device_buffer.h"
#include <vector>
#include <algorithm>

namespace cuda::kernel {
extern template __global__ void reduce_basic_kernel<int>(const int*, int*, size_t, ReduceOp);
extern template __global__ void reduce_optimized_kernel<int>(const int*, int*, size_t, ReduceOp);
extern template __global__ void reduce_basic_kernel<float>(const float*, float*, size_t, ReduceOp);
extern template __global__ void reduce_optimized_kernel<float>(const float*, float*, size_t, ReduceOp);
extern template __global__ void reduce_basic_kernel<double>(const double*, double*, size_t, ReduceOp);
extern template __global__ void reduce_optimized_kernel<double>(const double*, double*, size_t, ReduceOp);
extern template __global__ void reduce_basic_kernel<unsigned int>(const unsigned int*, unsigned int*, size_t, ReduceOp);
extern template __global__ void reduce_optimized_kernel<unsigned int>(const unsigned int*, unsigned int*, size_t, ReduceOp);
}

namespace cuda::algo {

namespace {
template<typename T>
T execute_reduce(const T* input, size_t size, bool optimized, ReduceOp op) {
    if (size == 0) return T{};

    const int blockSize = 256;
    int gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);

    DeviceBuffer<T> output(gridSize);

    if (optimized) {
        cuda::kernel::reduce_optimized_kernel<T><<<gridSize, blockSize>>>(input, output.data(), size, op);
    } else {
        cuda::kernel::reduce_basic_kernel<T><<<gridSize, blockSize>>>(input, output.data(), size, op);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> h_output(gridSize);
    output.copy_to(h_output.data(), gridSize);

    if (op == ReduceOp::SUM) {
        T result = 0;
        for (T val : h_output) result += val;
        return result;
    } else if (op == ReduceOp::MAX) {
        T result = h_output[0];
        for (size_t i = 1; i < h_output.size(); ++i) {
            result = max(result, h_output[i]);
        }
        return result;
    } else {
        T result = h_output[0];
        for (size_t i = 1; i < h_output.size(); ++i) {
            result = min(result, h_output[i]);
        }
        return result;
    }
}
} // anonymous namespace

template<typename T>
T reduce_sum(const T* input, size_t size) {
    return execute_reduce(input, size, false, ReduceOp::SUM);
}

template<typename T>
T reduce_sum_optimized(const T* input, size_t size) {
    return execute_reduce(input, size, true, ReduceOp::SUM);
}

template<typename T>
T reduce_max(const T* input, size_t size) {
    return execute_reduce(input, size, false, ReduceOp::MAX);
}

template<typename T>
T reduce_min(const T* input, size_t size) {
    return execute_reduce(input, size, false, ReduceOp::MIN);
}

#define REDUCE_ALGO_INSTANTIATE(T)  \
    template T reduce_sum<T>(const T*, size_t); \
    template T reduce_sum_optimized<T>(const T*, size_t); \
    template T reduce_max<T>(const T*, size_t); \
    template T reduce_min<T>(const T*, size_t);

REDUCE_ALGO_INSTANTIATE(int)
REDUCE_ALGO_INSTANTIATE(float)
REDUCE_ALGO_INSTANTIATE(double)
REDUCE_ALGO_INSTANTIATE(unsigned int)

} // namespace cuda::algo
```

- [ ] **Step 3: Update CMakeLists.txt to include algo sources**

Run: `cat >> CMakeLists.txt << 'EOF'

# Compile algo sources
add_library(cuda_algo_impl OBJECT ${CUDA_SRC_ALGO}/reduce.cpp)
target_link_libraries(cuda_algo_impl PRIVATE cuda_kernel CUDA::cudart)
set_target_properties(cuda_algo_impl PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

# Compile kernel sources
add_library(cuda_kernel_impl OBJECT ${CUDA_SRC_KERNEL}/reduce.cu)
target_link_libraries(cuda_kernel_impl PRIVATE CUDA::cudart)
set_target_properties(cuda_kernel_impl PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)
EOF`

- [ ] **Step 4: Rebuild and verify**

Run: `cmake -S . -B build && cmake --build build 2>&1 | tail -20`
Expected: Successful build

- [ ] **Step 5: Commit**

```bash
git add include/cuda/algo/reduce.h src/algo/reduce.cpp CMakeLists.txt && git commit -m "feat: add Layer 2 reduce algorithm wrapper"
```

---

## Task 7: Create Layer 3 - Device Vector

**Files:**
- Create: `include/cuda/api/device_vector.h`

- [ ] **Step 1: Write device_vector.h**

```cpp
#pragma once

#include "device_buffer.h"
#include <vector>
#include <algorithm>

namespace cuda::api {

template<typename T>
class DeviceVector {
public:
    explicit DeviceVector(size_t size = 0) : buffer_(size) {}

    size_t size() const { return buffer_.size(); }
    T* data() { return buffer_.data(); }
    const T* data() const { return buffer_.data(); }

    void resize(size_t new_size) {
        buffer_ = DeviceBuffer<T>(new_size);
    }

    void copy_from(const std::vector<T>& host_data) {
        buffer_.copy_from(host_data.data(), host_data.size());
    }

    void copy_to(std::vector<T>& host_data) const {
        host_data.resize(size());
        buffer_.copy_to(host_data.data(), size());
    }

    DeviceBuffer<T>& buffer() { return buffer_; }
    const DeviceBuffer<T>& buffer() const { return buffer_; }

private:
    DeviceBuffer<T> buffer_;
};

} // namespace cuda::api
```

- [ ] **Step 2: Commit**

```bash
git add include/cuda/api/device_vector.h && git commit -m "feat: add Layer 3 DeviceVector API"
```

---

## Task 8: Create Forwarding Headers (Backward Compatibility)

**Files:**
- Create: `include/reduce.h` (forwards to `include/cuda/kernel/reduce.h`)
- Create: `include/scan.h` (placeholder)
- Create: `include/sort.h` (placeholder)
- Create: `include/cuda_utils.h` (forwards to `include/cuda/kernel/cuda_utils.h`)

- [ ] **Step 1: Write forwarding headers**

```cpp
#pragma once

// Forwarding header - redirects to new layered structure
#include "cuda/kernel/cuda_utils.h"
```

```cpp
#pragma once

#include "cuda/kernel/reduce.h"
#include "cuda/algo/reduce.h"
```

```cpp
#pragma once

// Placeholder for backward compatibility
namespace cuda::kernel {
enum class ReduceOp { SUM, MAX, MIN };
}
namespace cuda::algo {
template<typename T> T reduce_sum(const T*, size_t);
template<typename T> T reduce_sum_optimized(const T*, size_t);
template<typename T> T reduce_max(const T*, size_t);
template<typename T> T reduce_min(const T*, size_t);
}
using cuda::algo::reduceSum;
using cuda::algo::reduceSumOptimized;
using cuda::algo::reduceMax;
using cuda::algo::reduceMin;
```

- [ ] **Step 2: Update tests/CMakeLists.txt**

```cmake
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

add_executable(cuda-samples-tests
    image_utils_test.cu
    gaussian_blur_test.cu
    sobel_edge_test.cu
    brightness_test.cu
    reduce_test.cu
    scan_test.cu
    sort_test.cu
    matrix_mult_test.cu
)

target_link_libraries(cuda-samples-tests
    PRIVATE
    GTest::gtest_main
    GTest::gmock
    CUDA::cudart
    cuda_algo_impl
    cuda_kernel_impl
)

target_include_directories(cuda-samples-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/data
)

target_compile_options(cuda-samples-tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -lineinfo
    >
)

include(GoogleTest)
gtest_discover_tests(cuda-samples-tests)
```

- [ ] **Step 3: Rebuild and run tests**

Run: `cmake --build build && make test`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add include/reduce.h include/cuda_utils.h tests/CMakeLists.txt && git commit -m "chore: add forwarding headers for backward compatibility"
```

---

## Task 9: Migrate Main Demo

**Files:**
- Modify: `src/main.cpp`

- [ ] **Step 1: Update main.cpp to use new layered API**

Replace the demo to showcase the new architecture:

```cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>

#include "cuda/algo/reduce.h"
#include "cuda/api/device_vector.h"

class Timer {
public:
    Timer(const char* name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<float, std::milli>(end - start_).count();
        std::cout << std::left << std::setw(35) << name_
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3)
                  << ms << " ms" << std::endl;
    }

private:
    const char* name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

template<typename T>
void printResult(const char* name, T result) {
    std::cout << std::left << std::setw(35) << name << ": " << result << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   CUDA Parallel Algorithms Benchmark   " << std::endl;
    std::cout << "   (Layered Architecture Demo)          " << std::endl;
    std::cout << "========================================" << std::endl;

    constexpr size_t N = 1 << 20;

    std::cout << "\n--- Data Setup ---" << std::endl;
    std::cout << "Array size: " << N << " elements" << std::endl;

    std::vector<int> input(N);
    for (size_t i = 0; i < N; ++i) {
        input[i] = static_cast<int>(i + 1);
    }

    cuda::api::DeviceVector<int> d_input(N);
    d_input.copy_from(input);

    std::cout << "\n--- Reduce (Sum) ---" << std::endl;
    std::cout << std::left << std::setw(35) << "Algorithm" << std::right << std::setw(15) << "Time" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    int result = 0;
    {
        Timer t("Reduce Basic (Layer 2)");
        result = cuda::algo::reduce_sum(d_input.data(), d_input.size());
    }
    printResult("  Sum result", result);

    {
        Timer t("Reduce Optimized (Layer 2)");
        result = cuda::algo::reduce_sum_optimized(d_input.data(), d_input.size());
    }
    printResult("  Sum result", result);

    int maxResult = 0;
    {
        Timer t("Reduce Max (Layer 2)");
        maxResult = cuda::algo::reduce_max(d_input.data(), d_input.size());
    }
    printResult("  Max result", maxResult);

    std::cout << "\n========================================" << std::endl;
    std::cout << "         Benchmark Complete!            " << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
```

- [ ] **Step 2: Build and run demo**

Run: `cmake --build build && ./build/bin/cuda-samples`
Expected: Demo runs successfully

- [ ] **Step 3: Commit**

```bash
git add src/main.cpp && git commit -m "feat: update main demo to use layered API"
```

---

## Task 10: Final Verification

**Files:**
- Verify: All files in correct locations
- Verify: Tests pass

- [ ] **Step 1: Run all tests**

Run: `cd build && ctest --output-on-failure`
Expected: All tests pass

- [ ] **Step 2: Verify directory structure**

Run: `find include/cuda src/kernel src/algo tests -type f -name "*.h" -o -name "*.cu" -o -name "*.cpp" | sort`
Expected: All files in correct locations per spec

- [ ] **Step 3: Commit final changes**

```bash
git add -A && git commit -m "feat: complete layered architecture refactoring"
```

---

## Spec Coverage Checklist

| Requirement | Task |
|-------------|------|
| Directory structure | Task 1 |
| Modern CMake with INTERFACE libs | Task 2 |
| Layer 1 kernel (reduce) | Tasks 3-4 |
| Layer 2 algorithm wrapper (reduce) | Tasks 5-6 |
| Layer 3 device_vector | Task 7 |
| Backward compatibility | Task 8 |
| Demo updated | Task 9 |
| Tests pass | Task 10 |

---

**Plan complete.** Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch subagent per task, two-stage review
2. **Inline Execution** - Execute tasks in this session with checkpoints

**Which approach?**
