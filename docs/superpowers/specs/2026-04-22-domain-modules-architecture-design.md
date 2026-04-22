# CUDA Parallel Algorithms Library - Architecture Design

**Date:** 2026-04-22
**Status:** Approved for Implementation
**Type:** Greenfield Refactoring (no backward compat)

## 1. Philosophy

**Modern C++20 GPU Computing** - Clean, type-safe, zero-overhead abstractions.

- Device-only APIs (no H2D/D2H magic)
- RAII everywhere (Buffer<T>, UniquePtr<T>)
- No global state (remove __constant__ globals)
- Monadic error handling via exceptions
- Builder pattern for kernel launches

## 2. Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Public API Layer                                 │
│                         cuda::algo::*                                   │
│   ImageFilters | MatrixOps | ParallelPrimitives | Convolution           │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Abstraction Layer                                   │
│     KernelLauncher | Buffer<T> | UniquePtr<T> | Stream | Event          │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        CUDA Layer 0                                     │
│                  error.h (CUDA_CHECK, exceptions)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3. Core Abstractions

### 3.1 KernelLauncher

```cpp
namespace cuda::detail {

class KernelLauncher {
public:
    KernelLauncher& grid(dim3 g) &;
    KernelLauncher& block(dim3 b) &;
    KernelLauncher& shared(size_t s) &;
    KernelLauncher& stream(cudaStream_t s) &;

    template<typename... Args>
    void launch(const void* kernel, Args&&... args);

    void synchronize() const;

private:
    dim3 grid_{1,1,1};
    dim3 block_{1,1,1};
    size_t shared_{0};
    cudaStream_t stream_{nullptr};
};

// Helpers
[[nodiscard]] constexpr dim3 calc_grid_1d(size_t n, dim3 block = {256,1,1});
[[nodiscard]] constexpr dim3 calc_grid_2d(size_t w, size_t h, dim3 block = {16,16,1});
[[nodiscard]] constexpr dim3 calc_grid_3d(size_t x, size_t y, size_t z, dim3 block = {8,8,8});
```

### 3.2 Buffer<T>

```cpp
namespace cuda::memory {

template<typename T>
class Buffer {
public:
    Buffer() = default;
    explicit Buffer(size_t count);
    Buffer(size_t count, const T& fill_value);
    ~Buffer();

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    explicit operator bool() const { return ptr_ != nullptr; }

    void copy_to(Buffer<T>& dest) const;
    void copy_from(const T* host_ptr, size_t count);
    void fill(const T& value);

private:
    T* ptr_{nullptr};
    size_t size_{0};
};

}
```

### 3.3 Device-only API Pattern

```cpp
// All functions take device pointers, return void, throw on error
namespace cuda::algo {

struct ImageFilters {
    static void gaussianBlur(
        memory::Buffer<uint8_t> input,
        memory::Buffer<uint8_t> output,
        size_t width, size_t height,
        float sigma, int kernel_size = 5);

    static void brightness(
        memory::Buffer<uint8_t> input,
        memory::Buffer<uint8_t> output,
        size_t width, size_t height,
        float contrast_factor,
        float brightness_offset);
};

struct MatrixOps {
    static void add(
        memory::Buffer<float> a,
        memory::Buffer<float> b,
        memory::Buffer<float> c,
        int rows, int cols);

    static void multiply(
        memory::Buffer<float> a,   // [m, k]
        memory::Buffer<float> b,   // [k, n]
        memory::Buffer<float> c,   // [m, n]
        int m, int k, int n);
};

struct ParallelPrimitives {
    template<typename T>
    static void scan(memory::Buffer<T> input, memory::Buffer<T> output, size_t size);

    template<typename T>
    static void sort(memory::Buffer<T> data, size_t size);
};

}
```

## 4. Implementation Plan

### Phase 1: Core Infrastructure
- [x] `error.h` - CUDA_CHECK, CudaException
- [x] `buffer.h` / `buffer.cpp` - Buffer<T> RAII wrapper
- [x] `memory_pool.h` / `memory_pool.cpp` - Real memory pooling
- [x] `unique_ptr.h` - Device smart pointer
- [x] `kernel_launcher.h` - Launch abstraction

### Phase 2: Domain Refactoring
- [x] `gaussian_blur.cu` - Remove __constant__ globals, use Buffer, use KernelLauncher
- [x] `add.cu` - Use KernelLauncher, consistent API
- [x] `scan.cu` - Use KernelLauncher, use Buffer
- [x] `sort.cu` - Use KernelLauncher, use Buffer

### Phase 3: Matrix Operations
- [x] `mult.cu` - Create CublasContext RAII, use Buffer
- [x] `conv2d.cu` - Refactor to use Buffer, KernelLauncher

### Phase 4: Stream/Event
- [x] `stream.h` / `event.h` - Stream and Event RAII wrappers

### Phase 5: Performance Primitives
- [x] `device_utils.h` - Add block_reduce, warp primitives

### Phase 6: Testing
- [x] Add convolution_test.cu (refactored)
- [x] Add cublas_context_test.cu
- [x] All 166 tests passing

## 5. Files Structure

```
include/cuda/
├── error.h                    # CUDA_CHECK, CudaException
├── device/
│   ├── device_utils.h         # ReduceOp, warp_reduce
│   └── reduce_kernels.h       # reduce sum/min/max
├── memory/
│   ├── buffer.h               # Buffer<T>
│   ├── memory_pool.h          # MemoryPool
│   └── unique_ptr.h           # make_unique, UniquePtr
├── stream/
│   ├── stream.h               # Stream wrapper
│   └── event.h                # Event wrapper
└── algo/
    ├── kernel_launcher.h      # KernelLauncher + helpers
    ├── image_filters.h        # ImageFilters
    ├── matrix_ops.h           # MatrixOps
    ├── parallel_primitives.h  # Scan, Sort, Histogram
    └── convolution.h          # Conv2D

src/
├── memory/
│   ├── buffer.cpp
│   └── memory_pool.cpp
├── cuda/
│   ├── device/
│   │   └── reduce_kernels.cu
│   └── algo/
│       └── kernel_launcher.cu
├── image/
│   └── gaussian_blur.cu       # Refactored
├── matrix/
│   └── add.cu                 # Refactored
├── parallel/
│   ├── scan.cu                # Refactored
│   └── sort.cu                # Refactored
└── convolution/
    └── conv2d.cu              # Refactored
```

## 6. Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Namespaces | lowercase | `cuda::algo`, `cuda::memory` |
| Classes | PascalCase | `Buffer<T>`, `KernelLauncher` |
| Methods | snake_case | `copy_to()`, `launch()` |
| Free functions | snake_case | `calc_grid_1d()` |
| Types | PascalCase | `CudaException`, `ReduceOp` |
| Constants | kCamelCase | `kDefaultBlockSize` |
| Files | snake_case | `buffer.h`, `kernel_launcher.cpp` |

## 7. Error Handling

All CUDA errors throw `CudaException`:

```cpp
namespace cuda {

class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t err, const char* file, int line);
    cudaError_t error() const { return error_; }
    const char* what() const noexcept override;
private:
    cudaError_t error_;
};

// RAII macro
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            throw cuda::CudaException(err, __FILE__, __LINE__); \
        } \
    } while(0)
}
```

## 8. Key Changes from Current Code

### Remove
- `__constant__` globals in gaussian_blur.cu
- `exit()` on error (replace with throw)
- Raw `new[]/delete[]` (use Buffer/vector)
- Duplicate CUDA_CHECK_IMAGE macro
- Hardcoded shared memory sizes

### Replace
- `cudaMalloc` → `Buffer<T>`
- `cudaFree` → Buffer destructor
- Manual H2D/D2H → User manages buffers
- Raw pointers → `UniquePtr<T>`

## 9. Testing Strategy

```cpp
// Unit tests
BufferTest - allocation, copy, fill
KernelLauncherTest - launch, sync, config
MemoryPoolTest - pool, release, stress

// Integration tests
ImageFilterTest - gaussianBlur produces correct output
MatrixOpsTest - add, multiply correctness
ParallelTest - scan, sort correctness

// Performance tests
BenchmarkTest - verify no regression
```

## 10. Success Criteria

- [x] error.h complete with exceptions
- [x] Buffer<T> RAII wrapper working
- [x] MemoryPool real pooling implemented
- [x] KernelLauncher abstraction complete
- [x] All __constant__ globals removed
- [x] All raw cudaMalloc/cudaFree replaced
- [x] All tests passing (166 tests)
- [ ] Clean build with no warnings (deprecation warnings remain)
