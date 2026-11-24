# Quick Start Guide

A 5-minute guide to writing your first CUDA program with Nova.

## Prerequisites

- CUDA Toolkit 12.0+
- CMake 4.0+
- C++ compiler (GCC 11+ or Clang 15+)
- NVIDIA GPU with CUDA support

## Installation

```bash
# Clone the repository
git clone https://github.com/pplmx/nova.git
cd nova

# Configure with CMake
cmake -G Ninja -B build -DNOVA_ENABLE_NCCL=OFF

# Build
cmake --build build --parallel

# Run tests
ctest --test-dir build
```

## Your First Program

Create `hello_nova.cpp`:

```cpp
#include <cuda/error/cuda_error.hpp>
#include <cuda/memory/buffer.h>
#include <cuda/algo/reduce.h>
#include <stdio.h>
#include <vector>

int main() {
    const size_t size = 1024;

    // Build the data on the host; Buffer owns device memory only.
    std::vector<float> host_input(size, 1.0f);

    // Create a buffer on the GPU and upload. copy_from() takes the host pointer and a count.
    cuda::memory::Buffer<float> input(size);
    input.copy_from(host_input.data(), size);

    // reduce_sum() takes the device pointer and the element count and returns the scalar directly:
    // it synchronises internally, so no separate download is needed for the result.
    const float sum = cuda::algo::reduce_sum(input.data(), size);
    printf("Sum: %f\n", sum);

    return 0;
}
```

Build with:

```bash
g++ -std=c++23 -I include -I /usr/local/cuda/include \
    hello_nova.cpp -L build/lib -lcuda_impl -lcudart -o hello_nova
```

## Key Concepts

### Buffers

`cuda::memory::Buffer<T>` manages GPU memory:

```cpp
// Create buffer
cuda::memory::Buffer<float> buffer(size);

// Access host memory
float* host = buffer.host_data();

// Access device memory
float* device = buffer.device_data();

// Sync data between host and device
buffer.sync_to_device();   // host -> device
buffer.sync_to_host();     // device -> host
```

### Error Handling

Use `NOVA_CHECK` for automatic error checking:

```cpp
NOVA_CHECK(cudaMalloc(&ptr, size));
NOVA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
```

### Algorithms

Available algorithms:

- `cuda::algo::reduce` - Sum/min/max reduction
- `cuda::algo::scan` - Prefix sum (inclusive/exclusive)
- `cuda::algo::sort` - Bitonic/odd-even merge sort
- `cuda::algo::histogram` - Histogram computation

## Next Steps

- [Multi-GPU Tutorial](02-multi-gpu.md) - Scale to multiple GPUs
- [Checkpoint Tutorial](03-checkpoint.md) - Save and restore state
- [Architecture](../ARCHITECTURE.md) - Layer and module reference

## Troubleshooting

### "No CUDA-capable device is detected"

Ensure:

1. NVIDIA GPU is installed: `nvidia-smi`
2. CUDA drivers are installed: `nvcc --version`
3. You're running on the GPU machine (not via SSH without GPU passthrough)

### "undefined reference to `nova::*`"

Add the Nova library to your link command:

```bash
-L /path/to/nova/build/lib -lcuda_impl
```

### Build errors with CMake

Clean and rebuild:

```bash
rm -rf build
cmake -G Ninja -B build
cmake --build build
```
