# CUDA Samples Architecture Redesign Specification

**Date:** 2025-04-22
**Status:** Approved
**Type:** Architecture Refactoring

## 1. Overview

Refactor the existing CUDA samples project from a flat structure into a production-ready, layered architecture that supports:
- Educational demonstrations
- Extensibility for new algorithms
- Production-grade library quality

## 2. Architecture

### 2.1 Layered Structure

```
include/cuda/
├── kernel/           # Layer 1: Pure device kernels
│   ├── reduce.h
│   ├── scan.h
│   └── sort.h
├── algo/             # Layer 2: Algorithm wrappers
│   ├── reduce.h
│   ├── scan.h
│   ├── sort.h
│   └── device_buffer.h
└── api/              # Layer 3: High-level STL-style API
    ├── device_vector.h
    └── thrust_compat.h

src/
├── kernel/           # Kernel implementations (.cu)
├── algo/             # Algorithm implementations (.cpp/.cu)
├── benchmark/        # Benchmark utilities
└── main.cpp          # Demo entry point

tests/
├── unit/             # Kernel-level unit tests
└── integration/      # Algorithm integration tests
```

### 2.2 Layer Responsibilities

| Layer | Location | Purpose | Dependencies |
|-------|----------|---------|--------------|
| Layer 1 | kernel/ | Pure CUDA kernels, no memory management, maximum performance | None (except CUDA runtime) |
| Layer 2 | algo/ | Memory allocation/deallocation, error handling, algorithm orchestration | Layer 1 |
| Layer 3 | api/ | STL-style containers, iterators, range adapters | Layer 1, Layer 2 |

### 2.3 Design Principles

1. **Single Responsibility**: Each layer has one clear purpose
2. **Interface Segregation**: Headers expose only necessary APIs
3. **Dependency Inversion**: High-level code depends on abstractions
4. **Zero-Cost Abstraction**: Layer 3 overhead must be minimal

## 3. Migration Plan

### 3.1 Directory Mapping

| Existing | New Location |
|----------|-------------|
| include/*.h | include/cuda/kernel/*.h (kernels), include/cuda/algo/*.h (wrappers) |
| src/*.cu | src/kernel/*.cu |
| src/main.cpp | src/main.cpp (benchmark demo) |
| tests/*_test.cu | tests/unit/ or tests/integration/ |

### 3.2 Algorithm Classification

**Parallel Primitives (Core Algorithms):**
- reduce (sum, max, min)
- scan (exclusive, inclusive)
- sort (bitonic, odd-even)

**Image Processing (Future Extension):**
- brightness, gaussian_blur, sobel_edge
- matrix_add, matrix_mult

## 4. CMake Structure

```cmake
# Modern CMake with INTERFACE libraries
add_library(cuda_kernel INTERFACE)      # Layer 1
add_library(cuda_algo INTERFACE)        # Layer 2
add_library(cuda_api INTERFACE)         # Layer 3

# Dependencies
target_link_libraries(cuda_algo PUBLIC cuda_kernel)
target_link_libraries(cuda_api PUBLIC cuda_algo)

# Executable links all layers
add_executable(cuda-samples src/main.cpp)
target_link_libraries(cuda-samples PRIVATE cuda_api)
```

## 5. API Design

### 5.1 Layer 1: Kernel Interface (Example: Reduce)

```cpp
// include/cuda/kernel/reduce.h
namespace cuda::kernel {

template<typename T>
__device__ T warp_reduce(T val, ReduceOp op);

template<typename T>
__global__ void reduce_kernel(const T* input, T* output, size_t size, ReduceOp op);

} // namespace cuda::kernel
```

### 5.2 Layer 2: Algorithm Interface

```cpp
// include/cuda/algo/reduce.h
namespace cuda::algo {

template<typename T>
struct ReduceBuffer {
    T* device_data;
    size_t size;
    // RAII memory management
};

template<typename T>
T reduce_sum(const T* input, size_t size);

template<typename T>
T reduce_max(const T* input, size_t size);

} // namespace cuda::algo
```

### 5.3 Layer 3: High-level API

```cpp
// include/cuda/api/device_vector.h
namespace cuda::api {

template<typename T>
class device_vector {
public:
    explicit device_vector(size_t size);
    ~device_vector();

    size_t size() const;
    T* data();
    const T* data() const;

    void copy_from(const T* host_data, size_t count);
    void copy_to(T* host_data, size_t count) const;

    // STL-like interface
    template<typename Func>
    void for_each(Func&& func);
};

} // namespace cuda::api
```

## 6. Testing Strategy

### 6.1 Unit Tests (tests/unit/)
- Test each kernel in isolation
- Mock-free, use actual CUDA kernels
- Focus on boundary conditions

### 6.2 Integration Tests (tests/integration/)
- Test algorithm wrappers end-to-end
- Compare results with reference CPU implementation
- Property-based testing for numerical algorithms

## 7. Backward Compatibility

During transition:
- Keep old includes working via forwarding headers
- Deprecation warnings for old API
- Full migration in v1.0.0

## 8. File Naming Conventions

| Type | Extension | Example |
|------|-----------|---------|
| Header (declaration) | .h | reduce.h |
| Header (implementation) | .cuh | reduce_kernel.cuh |
| CUDA source | .cu | reduce_kernel.cu |
| C++ source | .cpp | reduce.cpp |

## 9. Scope

### In Scope
- Complete directory restructuring
- CMake modernization
- Code migration to new structure
- Basic documentation

### Out of Scope (Future)
- Image processing module restructuring
- Performance benchmarking framework
- CUDA stream management
- Multi-GPU support

## 10. Acceptance Criteria

1. All existing tests pass after migration
2. Main demo runs successfully
3. Clean CMake configuration with no warnings
4. Clear separation between layers
5. Consistent naming conventions
