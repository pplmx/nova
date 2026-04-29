# Coding Conventions

**Analysis Date:** 2026-04-30

## Language Standards

**C++:** C++23 (CMAKE_CXX_STANDARD 23)
**CUDA:** CUDA 20 (CMAKE_CUDA_STANDARD 20)

## Naming Conventions

Enforced by `.clang-tidy` and `.clang-format`:

| Element | Convention | Example |
|---------|------------|---------|
| Classes | CamelCase | `class BufferTest`, `class MemoryPool` |
| Structs | CamelCase | `struct TestResult`, `struct OperationContext` |
| Enums | CamelCase | `enum class TransformType` |
| Methods/Functions | lower_case | `reduce_sum()`, `copy_from()`, `allocate()` |
| Variables | lower_case | `data_`, `size_`, `num_blocks_` |
| Parameters | lower_case | `host_data`, `count`, `size` |
| Member variables | camelBack (trailing `_`) | `data_`, `size_` |
| Static members | camelBack (trailing `_`) | `instance_` |
| Namespaces | lower_case | `cuda::memory`, `cuda::algo`, `nova::property` |
| Private implementation | suffix with `_` | `data_`, `size_`, `ptr_` |
| Constants | `k` prefix + CamelCase | `MAX_SCAN_SIZE`, `BLOCK_SIZE` |

## Code Style

### Formatting

**Tool:** clang-format (`.clang-format`)

**Key settings:**
- Style: Google-based
- Column limit: 180 characters
- Indent width: 4 spaces (no tabs)
- Brace style: Attach (`if (...) {`)
- Pointer alignment: Left (`T* ptr`, `T& ref`)
- Namespace indentation: All levels
- Access modifier offset: -4 (for `public:`, `private:` alignment within classes)

**Pre-commit hooks** (`.pre-commit-config.yaml`):
- Trailing whitespace fixer
- TOML/YAML validator
- Mixed line ending fixer (LF)
- Commitizen for commit message format
- rumdl for Markdown linting

### Linting

**Tool:** clang-tidy (`.clang-tidy`)

**Enabled checks:**
- `bugprone-*` - Bug-prone code patterns
- `readability-*` - Readability improvements
- `performance-*` - Performance anti-patterns
- `modernize-*` - Modern C++ practices
- `cppcoreguidelines-*` - C++ Core Guidelines
- `clang-analyzer-*` - Static analysis
- `portability-*` - Cross-platform issues
- `security-*` - Security concerns
- `google-*` - Google style guide

**Function cognitive complexity threshold:** 35

## File Organization

### Include Guards

**Pattern for headers:**
```cpp
#pragma once
```

(Also accepts traditional `#ifndef NOVA_*_HPP` guards when appropriate)

### Header Structure

```cpp
#pragma once

/**
 * @file filename.h
 * @brief Brief description of file purpose
 */

#include <system_headers>
#include "cuda/internal/headers"
#include "local/headers"

namespace cuda::module {

// Class/function declarations

}  // namespace cuda::module
```

### Source File Structure

```cpp
#include "include/path.h"

namespace cuda::module {

// Inline implementations or separate .cpp file

}  // namespace cuda::module
```

### Directory Layout

```
include/
├── cuda/              # Public CUDA headers
│   ├── memory/        # Memory management
│   ├── algo/          # Algorithms
│   ├── neural/        # Neural network ops
│   └── ...
src/
├── cuda/              # Implementation sources (.cu, .cpp)
├── memory/            # Non-CUDA implementations
├── image/             # Image processing
└── ...
tests/
├── unit/              # Unit tests
├── property/          # Property-based tests
├── fuzz/              # Fuzz tests
└── ...
```

## Class Design

### RAII Pattern

Use RAII wrappers for resource management:

```cpp
template <typename T>
class Buffer {
public:
    Buffer() : data_(nullptr), size_(0) {}
    explicit Buffer(size_t count);
    ~Buffer() { if (data_) cudaFree(data_); }

    // Delete copy operations
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Allow move operations
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    T* data() { return data_; }
    size_t size() const { return size_; }
private:
    T* data_ = nullptr;
    size_t size_ = 0;
};
```

### Const Correctness

Always use `const` for non-modifying methods:

```cpp
const T* data() const { return data_; }
size_t size() const { return size_; }
```

### Explicit Constructors

Use `explicit` for single-argument constructors to prevent implicit conversions:

```cpp
explicit Buffer(size_t count);
explicit RandomGenerator(uint64_t seed);
```

## Error Handling

### CUDA Error Checking

**Macro pattern** (defined in `include/cuda/device/error.h`):

```cpp
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            throw cuda::device::CudaException(err, __FILE__, __LINE__); \
        }                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                        \
    do {                                                          \
        cublasStatus_t status = call;                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                    \
            throw cuda::device::CublasException(status, __FILE__, __LINE__); \
        }                                                         \
    } while (0)
```

**Usage:**
```cpp
void Buffer<T>::copy_from(const T* host_data, size_t count) {
    CUDA_CHECK(cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
}
```

### Exception Classes

Custom exceptions inherit from `std::runtime_error`:

```cpp
class CudaException : public std::runtime_error {
public:
    explicit CudaException(cudaError_t err, const char* file, int line)
        : std::runtime_error(format_error(err, file, line)),
          error_(err) {}

    [[nodiscard]] auto error() const noexcept -> cudaError_t { return error_; }
private:
    cudaError_t error_;

    static auto format_error(cudaError_t err, const char* file, int line) -> std::string {
        return std::string(file) + ":" + std::to_string(line) +
               " - CUDA error: " + std::string(cudaGetErrorString(err));
    }
};
```

### Validation Patterns

```cpp
#define CUDA_VALIDATE_SIZE(size, max_size, operation)                    \
    do {                                                                \
        if ((size) > (max_size)) {                                      \
            throw cuda::device::CudaExceptionWithContext(               \
                cudaErrorLaunchFailure, __FILE__, __LINE__,              \
                cuda::device::OperationContext{#operation,              \
                    static_cast<size_t>(size), 0});                      \
        }                                                               \
    } while (0)
```

## Memory Management

### Device Memory

Use RAII wrappers for device memory:

```cpp
cuda::memory::Buffer<int> buffer(1024);
std::vector<int> host_data(1024, 42);
buffer.copy_from(host_data.data(), 1024);
// Automatically freed when buffer goes out of scope
```

### No Raw malloc

**clang-tidy rule:** `cppcoreguidelines-no-malloc` is enabled

Use RAII wrappers or `cudaMalloc` with corresponding `cudaFree`.

## Function Design

### Trailing Return Type

Use trailing return type for complex function signatures:

```cpp
template <typename T>
auto reduce_sum(const T* input, size_t size) -> T;
```

### Parameter Order

1. Output parameters (pointers/references that receive results)
2. Input parameters
3. Size/count parameters

## Documentation Standards

### Doxygen Style

```cpp
/**
 * @class Buffer
 * @brief RAII wrapper for CUDA device memory with automatic memory management.
 *
 * Buffer<T> provides RAII semantics for GPU memory allocation and deallocation.
 * Memory is automatically freed when the Buffer goes out of scope.
 *
 * @tparam T The element type stored in the buffer
 *
 * @example
 * @code
 * cuda::memory::Buffer<int> buf(1024);
 * buf.copy_from(host_data.data(), 1024);
 * @endcode
 */
template <typename T>
class Buffer {
public:
    /**
     * @brief Allocates GPU memory for the specified number of elements
     * @param count Number of elements to allocate
     * @throws CudaException if allocation fails
     */
    explicit Buffer(size_t count);
};
```

### Comments

**Trailing comments:** 2 spaces before comment

```cpp
int result = 0;  // Initialize to zero
data_ = nullptr;  // Release ownership
```

## Preprocessor

### Macros

**Use inline functions/lambdas over macros when possible.**
Use macros only for:
- Error checking patterns (`CUDA_CHECK`)
- Code generation (`NOVA_PROPERTY_TEST`)
- Compile-time configuration

**Macro naming:** Uppercase with underscores

```cpp
#define CUDA_CHECK(call) ...
#define NOVA_PROPERTY_TEST(name, iterations, generator, ...) ...
```

## Type Safety

### C-Style Casts

**clang-tidy rule:** `cppcoreguidelines-pro-type-cstyle-cast` is enabled

Use `static_cast`, `reinterpret_cast`, or `dynamic_cast` instead of C-style casts:

```cpp
// Bad
float val = (float)d;

// Good
float val = static_cast<float>(d);
```

## Testing Conventions

Tests follow GoogleTest pattern - see `TESTING.md` for details.

---

*Convention analysis: 2026-04-30*
