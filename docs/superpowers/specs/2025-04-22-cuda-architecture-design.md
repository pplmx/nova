# CUDA Samples Architecture Redesign Specification

**Date:** 2025-04-22
**Status:** Implemented
**Type:** Architecture Refactoring

## 1. Overview

Refactor the existing CUDA samples project from a flat structure into a production-ready, layered architecture that supports:
- Educational demonstrations
- Extensibility for new algorithms
- Production-grade library quality

## 2. Architecture

### 2.1 Directory Structure

```
include/
├── cuda/                # Layered architecture (core library)
│   ├── kernel/          # Layer 1: Pure device kernels
│   │   ├── cuda_utils.h # CUDA_CHECK, ReduceOp, warp_reduce
│   │   └── reduce.h     # Kernel declarations
│   ├── algo/            # Layer 2: Algorithm wrappers
│   │   ├── device_buffer.h  # RAII device memory
│   │   └── reduce.h     # reduce_sum, reduce_max, reduce_min
│   └── api/             # Layer 3: High-level STL-style API
│       └── device_vector.h  # STL-style container
├── image/               # Image processing module
│   ├── types.h          # ImageBuffer, ImageDimensions, PixelFormat
│   ├── brightness.h     # Brightness/contrast adjustment
│   ├── gaussian_blur.h  # Gaussian blur filter
│   └── sobel_edge.h     # Sobel edge detection
├── parallel/            # Parallel primitives module
│   ├── scan.h           # Prefix sum (exclusive/inclusive)
│   └── sort.h           # Odd-even sort, bitonic sort
└── matrix/              # Matrix operations module
    ├── add.h            # Matrix addition
    └── mult.h           # Matrix multiplication

src/
├── cuda/kernel/         # Layer 1 kernel implementations
├── image/               # Image processing implementations
├── parallel/            # Parallel primitive implementations
├── matrix/              # Matrix operation implementations
└── main.cpp             # Benchmark demo
```

### 2.2 Layer Responsibilities

| Layer | Location | Purpose | Dependencies |
|-------|----------|---------|--------------|
| Layer 1 | cuda/kernel/ | Pure CUDA kernels, no memory management, maximum performance | CUDA runtime only |
| Layer 2 | cuda/algo/ | Memory allocation/deallocation, error handling, algorithm orchestration | Layer 1 |
| Layer 3 | cuda/api/ | STL-style containers, iterators, range adapters | Layer 1, Layer 2 |

### 2.3 Design Principles

1. **Single Responsibility**: Each layer has one clear purpose
2. **Interface Segregation**: Headers expose only necessary APIs
3. **Dependency Inversion**: High-level code depends on abstractions
4. **Zero-Cost Abstraction**: Layer 3 overhead must be minimal

## 3. Modules

### 3.1 CUDA Layer (Core)

Three-layer architecture following cuBLAS/cuDNN patterns:
- **Layer 1 (kernel)**: Pure CUDA kernels, no memory management
- **Layer 2 (algo)**: Algorithm wrappers with memory management
- **Layer 3 (api)**: High-level STL-style abstractions

### 3.2 Image Processing

| Header | Description |
|--------|-------------|
| types.h | ImageBuffer template, ImageDimensions, PixelFormat |
| brightness.h | adjustBrightnessContrast() |
| gaussian_blur.h | gaussianBlur() |
| sobel_edge.h | sobelEdgeDetection() |

### 3.3 Parallel Primitives

| Header | Description |
|--------|-------------|
| scan.h | exclusiveScan(), inclusiveScan(), exclusiveScanOptimized() |
| sort.h | oddEvenSort(), bitonicSort() |

### 3.4 Matrix Operations

| Header | Description |
|--------|-------------|
| add.h | Matrix element-wise addition |
| mult.h | multiplyMatricesNaive(), multiplyMatricesTiled(), multiplyMatricesOnGPU() |

## 4. CMake Structure

```cmake
# Layer 1: cuda_kernel (INTERFACE library)
add_library(cuda_kernel INTERFACE)
target_include_directories(cuda_kernel INTERFACE
        $<BUILD_INTERFACE:${CUDA_KERNEL_DIR}>
        $<BUILD_INTERFACE:${CUDA_SRC_KERNEL}>
)
target_link_libraries(cuda_kernel INTERFACE CUDA::cudart)

# Layer 2: cuda_algo (depends on cuda_kernel)
add_library(cuda_algo INTERFACE)
target_include_directories(cuda_algo INTERFACE ...)
target_link_libraries(cuda_algo INTERFACE cuda_kernel CUDA::cublas)

# Layer 3: cuda_api (depends on cuda_algo)
add_library(cuda_api INTERFACE)
target_include_directories(cuda_api INTERFACE ...)
target_link_libraries(cuda_api INTERFACE cuda_algo)

# Implementation library
add_library(cuda_impl STATIC ${ALL_CUDA_SOURCES})
target_link_libraries(cuda_impl PUBLIC cuda_kernel cuda_algo CUDA::cudart CUDA::cublas)
```

## 5. API Design

### 5.1 Layer 1: Kernel Interface

```cpp
// include/cuda/kernel/cuda_utils.h
namespace cuda::kernel {

enum class ReduceOp { SUM, MAX, MIN };

constexpr int WARP_SIZE = 32;

template<typename T>
__device__ T warp_reduce(T val, ReduceOp op);

} // namespace cuda::kernel
```

### 5.2 Layer 2: Algorithm Interface

```cpp
// include/cuda/algo/reduce.h
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

### 5.3 Layer 3: High-level API

```cpp
// include/cuda/api/device_vector.h
namespace cuda::api {

template<typename T>
class DeviceVector {
public:
    explicit DeviceVector(size_t size = 0);

    size_t size() const;
    T* data();
    const T* data() const;

    void copy_from(const std::vector<T>& host_data);
    void copy_to(std::vector<T>& host_data) const;
};

} // namespace cuda::api
```

## 6. Testing

### 6.1 Test Organization

| Test Suite | Description |
|------------|-------------|
| cuda-samples-tests | All algorithm tests (67 tests) |
| test_patterns-tests | Test pattern generators (14 tests) |

### 6.2 Test Coverage

- Image processing: ImageBuffer, GaussianBlur, Sobel, Brightness
- Parallel primitives: Scan, Sort
- Matrix operations: MatrixMult
- Layered architecture: Reduce (sum, max, min, optimized)

## 7. Build System

### 7.1 Requirements

- CUDA Toolkit 12+
- CMake 3.25+
- C++20 compatible compiler
- CUDA-capable GPU

### 7.2 Build Targets

| Target | Description |
|--------|-------------|
| cuda_impl | Static library with all CUDA implementations |
| cuda-samples | Benchmark demo executable |
| cuda-samples-tests | Test executable |

## 8. File Naming Conventions

| Type | Extension | Example |
|------|-----------|---------|
| Header | .h | reduce.h |
| CUDA source | .cu | reduce.cu |
| C++ source | .cpp | - |

## 9. Acceptance Criteria

- [x] All existing tests pass after migration (67 tests)
- [x] Main demo runs successfully
- [x] Clean CMake configuration
- [x] Clear separation between layers
- [x] Consistent naming conventions
- [x] Directory structure organized by module
- [x] No backward compatibility forwarding headers
