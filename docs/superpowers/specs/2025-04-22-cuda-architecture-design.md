# CUDA Parallel Algorithms Library - Architecture Specification

**Date:** 2025-04-22
**Status:** Implemented (Phase 1 Complete)
**Type:** Architecture Refactoring

## 1. Overview

A production-ready CUDA parallel algorithms library with a multi-layered architecture supporting:
- Educational demonstrations
- Extensibility for new algorithms
- Production-grade library quality
- Memory efficiency and allocation optimization
- Future-proof design patterns

## 2. Architecture

### 2.1 Layered Architecture (Five Layers)

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: High-Level API (STL-style)                        │
│  - cuda::reduce(), cuda::sort()                            │
│  - Iterator support, Range-based                           │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Algorithm Wrappers                                │
│  - cuda::algo::reduce_sum(), cuda::algo::sort()           │
│  - Memory management, Stream scheduling                     │
│  - Execution policy support                                │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Device Kernels                                   │
│  - Pure __global__ kernels                                │
│  - No memory allocation                                    │
│  - Device-side utilities only                              │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Memory Foundation                                │
│  - Buffer<T>, unique_ptr<T>, MemoryPool                    │
│  - Allocator concepts                                     │
│  - Memory efficiency                                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
include/cuda/
├── memory/               # Layer 0: Memory Foundation
│   ├── buffer.h         # cuda::memory::Buffer<T>
│   ├── unique_ptr.h     # cuda::memory::unique_ptr<T>
│   ├── memory_pool.h    # MemoryPool for allocation efficiency
│   └── allocator.h      # Allocator concepts and implementations
├── device/              # Layer 1: Device Kernels
│   ├── reduce_kernels.h # Kernel declarations
│   ├── scan_kernels.h
│   ├── sort_kernels.h
│   └── device_utils.h   # CUDA_CHECK, warp_reduce, etc.
├── algo/                 # Layer 2: Algorithm Wrappers
│   ├── reduce.h         # reduce_sum, reduce_max, etc.
│   ├── scan.h
│   ├── sort.h
│   └── policy.h         # ExecutionPolicy (future)
├── api/                  # Layer 3: High-Level API
│   ├── algorithm.h      # cuda::reduce(), cuda::sort()
│   ├── iterator.h
│   └── device_vector.h  # cuda::api::DeviceVector<T>

include/
├── image/               # Image processing module
│   ├── types.h
│   ├── brightness.h
│   ├── gaussian_blur.h
│   ├── sobel_edge.h
│   └── morphology.h     # erode, dilate, opening, closing
├── parallel/            # Parallel primitives module
│   ├── scan.h
│   ├── sort.h
│   └── histogram.h      # Histogram computation
├── matrix/              # Matrix operations module
│   ├── add.h
│   ├── mult.h
│   └── ops.h            # transpose, element-wise ops
└── convolution/         # Convolution module
    └── conv2d.h         # 2D convolution

src/
├── memory/               # Layer 0 implementations
├── cuda/
│   ├── device/          # Layer 1 kernel implementations
│   └── algo/            # Layer 2 wrappers
├── image/               # Image processing implementations
├── parallel/            # Parallel primitive implementations
├── matrix/              # Matrix operation implementations
└── convolution/         # Convolution implementations
```

### 2.3 Layer Responsibilities

| Layer | Namespace | Purpose | Dependencies |
|-------|-----------|---------|--------------|
| **Layer 0** | `cuda::memory` | Memory allocation, RAII wrappers, pooling | CUDA runtime |
| **Layer 1** | `cuda::device` | Pure device kernels, maximum performance | Layer 0 |
| **Layer 2** | `cuda::algo` | Memory management, algorithm orchestration | Layers 0, 1 |
| **Layer 3** | `cuda::api` | STL-style containers, iterators | Layers 0, 1, 2 |

### 2.4 Design Principles

1. **Single Responsibility**: Each layer has one clear purpose
2. **Interface Segregation**: Headers expose only necessary APIs
3. **Dependency Inversion**: High-level code depends on abstractions
4. **Zero-Cost Abstraction**: Higher layers add minimal overhead
5. **Memory Efficiency**: Layer 0 provides allocation optimization
6. **Future-Proof**: Allocator and Policy patterns support evolution

## 3. Layer 0: Memory Foundation

### 3.1 Buffer<T>

```cpp
// include/cuda/memory/buffer.h
namespace cuda::memory {

template<typename T>
class Buffer {
public:
    explicit Buffer(size_t count);
    ~Buffer();

    T* data();
    const T* data() const;
    size_t size() const;

    void copy_from(const T* host_data, size_t count);
    void copy_to(T* host_data, size_t count) const;

    T* release();  // Transfer ownership
};

} // namespace cuda::memory
```

### 3.2 unique_ptr<T>

```cpp
// include/cuda/memory/unique_ptr.h
namespace cuda::memory {

template<typename T>
class unique_ptr {
public:
    unique_ptr() = default;
    explicit unique_ptr(size_t count);
    
    T* get() const;
    T* release();
    explicit operator bool() const;
    
    // Move semantics
    unique_ptr(unique_ptr&&) noexcept;
    unique_ptr& operator=(unique_ptr&&) noexcept;
};

} // namespace cuda::memory
```

### 3.3 MemoryPool

```cpp
// include/cuda/memory/memory_pool.h
namespace cuda::memory {

class MemoryPool {
public:
    struct Config {
        size_t block_size = 1 << 20;  // 1MB
        size_t max_blocks = 16;
        bool preallocate = false;
    };
    
    explicit MemoryPool(const Config& config = {});
    
    Buffer<void> allocate(size_t bytes);
    void deallocate(Buffer<void> buffer);
    
    size_t total_allocated() const;
    size_t total_available() const;
    void clear();
};

} // namespace cuda::memory
```

### 3.4 Allocator Concepts

```cpp
// include/cuda/memory/allocator.h
namespace cuda::memory {

template<typename T>
concept DeviceAllocator = requires(T alloc, size_t n, size_t size) {
    { alloc.allocate(n * size) } -> std::same_as<void*>;
    { alloc.deallocate(nullptr, n * size) } -> std::same_as<void>;
};

struct DefaultAllocator;
struct PooledAllocator;

} // namespace cuda::memory
```

## 4. CMake Structure

```cmake
# Layer 0: cuda_memory (INTERFACE library)
add_library(cuda_memory INTERFACE)
target_include_directories(cuda_memory INTERFACE
        $<BUILD_INTERFACE:${CUDA_MEMORY_DIR}>
        $<BUILD_INTERFACE:${MEMORY_DIR}>
)
target_link_libraries(cuda_memory INTERFACE CUDA::cudart)

# Layer 1: cuda_device (depends on cuda_memory)
add_library(cuda_device INTERFACE)
target_include_directories(cuda_device INTERFACE ...)
target_link_libraries(cuda_device INTERFACE cuda_memory CUDA::cudart)

# Layer 2: cuda_algo (depends on cuda_device)
add_library(cuda_algo INTERFACE)
target_include_directories(cuda_algo INTERFACE ...)
target_link_libraries(cuda_algo INTERFACE cuda_device CUDA::cublas)

# Layer 3: cuda_api (depends on cuda_algo)
add_library(cuda_api INTERFACE)
target_include_directories(cuda_api INTERFACE ...)
target_link_libraries(cuda_api INTERFACE cuda_algo)

# Implementation library
add_library(cuda_impl STATIC ${ALL_SOURCES})
target_link_libraries(cuda_impl PUBLIC cuda_api CUDA::cudart CUDA::cublas)
```

## 5. Modules

### 5.1 Image Processing

| Header | Description |
|--------|-------------|
| types.h | ImageBuffer, ImageDimensions, PixelFormat |
| brightness.h | adjustBrightnessContrast() |
| gaussian_blur.h | gaussianBlur() |
| sobel_edge.h | sobelEdgeDetection() |
| morphology.h | erodeImage, dilateImage, openingImage, closingImage |

### 5.2 Parallel Primitives

| Header | Description |
|--------|-------------|
| scan.h | exclusiveScan, inclusiveScan, exclusiveScanOptimized |
| sort.h | oddEvenSort, bitonicSort |
| histogram.h | computeHistogram, computeHistogramPerChannel, equalizeHistogram |

### 5.3 Matrix Operations

| Header | Description |
|--------|-------------|
| add.h | Matrix element-wise addition |
| mult.h | multiplyMatricesNaive, multiplyMatricesTiled, multiplyMatricesOnGPU |
| ops.h | transposeMatrix, transposeMatrixTiled, matrixElementwiseAdd, matrixScale |

### 5.4 Convolution

| Header | Description |
|--------|-------------|
| conv2d.h | convolve2D, createGaussianKernel, createSobelKernelX/Y |

## 6. Testing

### 6.1 Test Coverage

| Module | Tests | Description |
|--------|-------|-------------|
| Image | 29 | Buffer, GaussianBlur, Sobel, Brightness, Morphology |
| Parallel | 23 | Scan, Sort, Histogram |
| Matrix | 23 | Mult, Ops |
| Convolution | 13 | Conv2D, Kernels |
| CUDA Layer | 11 | Reduce (sum, max, min) |
| **Total** | **119** | **All passing** |

## 7. Build System

### 7.1 Requirements

- CUDA Toolkit 12+
- CMake 3.25+
- C++20 compatible compiler
- CUDA-capable GPU

### 7.2 Build Targets

| Target | Description |
|--------|-------------|
| cuda_impl | Static library with all implementations |
| cu | Benchmark demo executable |
| cu-tests | Test executable (81 tests) |
| test_patterns-tests | Pattern generator tests (14 tests) |

### 7.3 Makefile Targets

| Target | Description |
|--------|-------------|
| make build | Configure and build |
| make run | Run benchmark demo |
| make test | Run all tests |
| make clean | Clean build artifacts |

## 8. Future Extensions

### 8.1 Execution Policy (Planned)

```cpp
// Future: ExecutionPolicy support
struct execution::device_policy {
    cudaStream_t stream;
    MemoryPool* pool;
};

cuda::reduce(execution::device_policy{stream}, data, n);
```

### 8.2 Async/Future (Planned)

```cpp
// Future: Async operations
cuda::future<T> reduce_async(data, n);

auto result = cuda::reduce_async(data, n)
    .then([](T r) { return r * 2; });
```

### 8.3 Multi-GPU (Planned)

```cpp
// Future: GPU topology awareness
cuda::reduce(gpu_topology{}, data, n);
```

## 9. Acceptance Criteria

- [x] All tests pass (119 tests)
- [x] Main demo runs successfully
- [x] Clean CMake configuration
- [x] Layer 0 Memory Foundation implemented
- [x] Layer 1/2/3 separation maintained
- [x] Directory structure organized by module
- [x] Allocator concepts defined
- [x] MemoryPool for allocation efficiency
