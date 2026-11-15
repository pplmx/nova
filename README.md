# Nova CUDA Library

[![CI](https://github.com/pplmx/nova/workflows/CI/badge.svg)](https://github.com/pplmx/nova/actions)
[![Coverage](https://coveralls.io/repos/github/pplmx/nova/badge.svg?branch=main)](https://coveralls.io/github/pplmx/nova?branch=main)

A production-ready CUDA library for sparse matrix operations, neural network inference optimization, and quantized computation. Built with a five-layer architecture supporting education, research, and production deployments.

## Key Features

- **Sparse Matrix Operations**: CSR, ELL, SELL formats with iterative solvers (CG, GMRES, BiCGSTAB)
- **Neural Network Inference**: KV cache management, quantization-aware operations, FP8 support
- **Quantization**: INT8 and FP8 quantization with calibration and QAT support
- **Multi-GPU Support**: NCCL collectives and distributed memory pools
- **Observability**: Structured logging, NVTX profiling, performance analysis tools

## Quick Start

### Installation

```sh
git clone https://github.com/pplmx/nova.git
cd nova
cmake -B build -DNOVA_ENABLE_NCCL=OFF
cmake --build build --parallel
```

### Basic Sparse Matrix-Vector Multiply

```cpp
#include "cuda/sparse/matrix.hpp"
#include "cuda/sparse/sparse_ops.hpp"

// Create sparse matrix from dense
auto matrix = nova::sparse::SparseMatrix<float>::FromDense(dense.data(), rows, cols);

// Perform SpMV: y = A * x
nova::sparse::spmv(*matrix, d_x, d_y);
```

### Quantized Inference

```cpp
#include "cuda/quantize/quantize_tensor.hpp"
#include "cuda/quantize/fp8_gemm.hpp"

// Calibrate and quantize weights
auto cal = MinMaxCalibrator{};
cal.collect_data(weights.data(), size);
auto result = cal.compute();

// Use FP8 GEMM for accelerated inference
FP8GEMM gemm;
gemm.forward(fp8_weights, d_x, d_y, config);
```

## Modules

| Module             | Description                       | Documentation                                |
| ------------------ | --------------------------------- | -------------------------------------------- |
| **cuda::sparse**   | Sparse matrix formats and solvers | [docs/SPARSE.md](docs/SPARSE.md)             |
| **cuda::quantize** | INT8/FP8 quantization             | [docs/QUANTIZATION.md](docs/QUANTIZATION.md) |
| **cuda::gnn**      | Graph neural network operations   | [docs/GNN.md](docs/GNN.md)                   |
| **cuda::memory**   | Memory management and pools       | Layer 0 foundation                           |
| **cuda::algo**     | Parallel algorithms               | [docs/ALGORITHMS.md](docs/ALGORITHMS.md)     |

## Architecture

### Five-Layer Design

```text
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: High-Level API (STL-style)                        │
│  cuda::reduce(), cuda::sort()                              │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Algorithm Wrappers                                │
│  cuda::algo::reduce_sum(), memory management               │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Device Kernels                                   │
│  Pure __global__ kernels, no memory allocation             │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Memory Foundation                                │
│  Buffer<T>, unique_ptr<T>, MemoryPool, Allocator concepts   │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```text
include/cuda/
├── memory/               # Layer 0: Memory Foundation
│   ├── buffer.h         # cuda::memory::Buffer<T>
│   ├── unique_ptr.h     # cuda::memory::unique_ptr<T>
│   ├── memory_pool.h    # MemoryPool for allocation
│   └── allocator.h      # Allocator concepts
├── device/              # Layer 1: Device Kernels
│   ├── reduce_kernels.h
│   ├── scan_kernels.h
│   └── device_utils.h   # CUDA_CHECK, warp_reduce
├── algo/                 # Layer 2: Algorithm Wrappers
│   ├── reduce.h
│   ├── scan.h
│   └── sort.h
└── api/                  # Layer 3: High-Level API
    ├── device_vector.h   # STL-style device container
    ├── stream.h          # Stream and Event wrappers
    └── config.h          # Algorithm configuration objects

include/
├── image/               # Image processing
│   ├── types.h
│   ├── brightness.h
│   ├── gaussian_blur.h
│   ├── sobel_edge.h
│   └── morphology.h
├── parallel/            # Parallel primitives
│   ├── scan.h
│   ├── sort.h
│   └── histogram.h
├── matrix/              # Matrix operations
│   ├── add.h
│   ├── mult.h
│   └── ops.h
└── convolution/         # Convolution
    └── conv2d.h

src/
├── memory/               # Layer 0 implementations
├── cuda/
│   ├── device/           # Layer 1 implementations
│   └── algo/             # Layer 2 implementations
├── image/
├── parallel/
├── matrix/
└── convolution/
```

### Layer Responsibilities

| Layer       | Namespace      | Purpose                          | Dependencies   |
| ----------- | -------------- | -------------------------------- | -------------- |
| **Layer 0** | `cuda::memory` | Memory allocation, RAII, pooling | CUDA runtime   |
| **Layer 1** | `cuda::device` | Pure device kernels              | Layer 0        |
| **Layer 2** | `cuda::algo`   | Memory management, algorithms    | Layers 0, 1    |
| **Layer 3** | `cuda::api`    | STL-style containers             | Layers 0, 1, 2 |

## Quick Start

### Build (Make)

```sh
git clone https://github.com/pplmx/nova.git
cd nova
cmake -B build -DNOVA_ENABLE_NCCL=OFF
cmake --build build --parallel
```

### Build (Ninja - Faster)

```sh
cmake -G Ninja -B build-ninja -DNOVA_ENABLE_NCCL=OFF
cmake --build build-ninja --parallel
```

### Run Demo

```sh
./build/bin/nova  # or ./build-ninja/bin/nova
```

### Run Tests

```sh
cd build-ninja
ctest -j16        # Parallel tests (GPU memory limited to 16)
```

## Usage Examples

### Layer 0: Memory Foundation

```cpp
#include "cuda/memory/buffer.h"

// RAII memory management
cuda::memory::Buffer<int> buf(1024);
buf.copy_from(host_data.data(), 1024);

// Memory pool for efficiency
cuda::memory::MemoryPool pool({.block_size = 1 << 20});
auto buf2 = pool.allocate(1024);
```

### Layer 2: Algorithm API

```cpp
#include "cuda/algo/reduce.h"

// Use layered API
int sum = cuda::algo::reduce_sum(d_input, N);
int max = cuda::algo::reduce_max(d_input, N);
```

### Layer 3: High-Level API

```cpp
#include "cuda/api/device_vector.h"
#include "cuda/api/stream.h"
#include "cuda/api/config.h"

// DeviceVector - STL-style container
cuda::api::DeviceVector<int> d_vec(N);
d_vec.copy_from(input);
int sum = cuda::algo::reduce_sum(d_vec.data(), d_vec.size());

// Stream - RAII async operations
cuda::api::Stream stream;
stream.synchronize();

// Config - algorithm configuration
auto config = cuda::api::ReduceConfig::optimized_config();
```

## Modules

| Module                  | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| **cuda::sparse**        | Sparse matrix formats (CSR, ELL, SELL, HYB), SpMV, iterative solvers |
| **cuda::quantize**      | INT8/FP8 quantization, calibration, QAT support                      |
| **cuda::gnn**           | Graph sampling, message passing, graph attention                     |
| **cuda::memory**        | Buffer, unique_ptr, MemoryPool, DistributedMemoryPool                |
| **cuda::error**         | Error handling, timeout management, retry logic                      |
| **cuda::observability** | Logging, NVTX, performance analysis                                  |
| **cuda::algo**          | Parallel primitives (reduce, scan, sort)                             |
| **cuda::device**        | CUDA kernels, warp primitives                                        |

## v2.x Additions

The v2.x release adds production features for inference and scientific computing:

- **Sparse Iterative Solvers** (v2.10): CG, GMRES, BiCGSTAB with preconditioners
- **Quantization Infrastructure** (v2.12): FP8 support for H100/H200, calibration
- **Transformer Optimization** (v2.13): KV cache management, FP8 GEMM
- **Documentation Quality** (v2.14): API docs, code comments, guides

## Testing

**505 tests across multiple test suites, 99%+ passing:**

```sh
# Full test suite
ctest -j16

# Single test
./bin/nova-tests --gtest_filter="BufferTest.*"

# v1.4 specific tests
./bin/nova-tests --gtest_filter="*MpiContext*:*TopologyMap*:*MultiNodeContext*"
```

### Test Configuration

| Option                    | Default | Description                                    |
| ------------------------- | ------- | ---------------------------------------------- |
| `NOVA_ENABLE_NCCL`        | ON      | Enable NCCL collectives (requires NCCL)        |
| `NOVA_ENABLE_MPI`         | OFF     | Enable MPI multi-node support                  |
| `NOVA_ENABLE_UNITY_BUILD` | ON      | Faster compilation via unity builds            |
| `CTEST_PARALLEL_LEVEL`    | NCPU    | Test parallelism (capped at 16 for GPU memory) |

## Development

### Build Options

| Generator | Command                   | Speed       |
| --------- | ------------------------- | ----------- |
| Ninja     | `cmake -G Ninja -B build` | **Fastest** |
| Make      | `cmake -B build`          | Standard    |

### Build Targets

| Target                | Description                                     |
| --------------------- | ----------------------------------------------- |
| `cmake --build <dir>` | Build project (use `--parallel` for multi-core) |
| `ctest -j<N>`         | Run tests in parallel                           |
| `make clean`          | Clean build artifacts                           |

## Language standard split

Host code builds as C++23 (`CMAKE_CXX_STANDARD 23`) and uses the C++23 standard
library: `std::expected`, concepts, `std::span`, monadic `std::optional`. Device
code does not: nvcc 12.9 accepts at most `-std=c++20`, and everything a `.cu`
file reaches must respect that ceiling.

`include/cuda/cpp_level.hpp` is the single place that records the split:

- `NOVA_CPP23_HOST` — 1 when the translation unit really has the C++23 library.
  It tests `__cpp_lib_expected` rather than `__cplusplus`, because g++ 13 in
  `-std=c++23` mode reports `202100` and nvcc reports `202002` regardless.
- `NOVA_DEVICE_REACHABLE` — defined for CUDA translation units by
  `CMakeLists.txt`.
- `NOVA_REQUIRE_CPP23_HOST("<header>")` — placed by a header that uses C++23.
  Including it from device code stops the build with a message naming that
  header instead of a confusing failure inside a template.

A header that uses C++23 therefore cannot silently drift into the device include
graph; the build tells you immediately.

## Requirements

- CUDA Toolkit 20+
- CMake 4.0+
- C++23 compatible compiler for host code; nvcc supports up to C++20, so device
  code stays C++20 (see [Language standard split](#language-standard-split))
- CUDA-capable GPU
- (Optional) NCCL 2.25+ for multi-GPU collectives
- (Optional) MPI 3.1+ for multi-node support
- (Optional) Ninja for faster builds

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

See [CONTRIBUTING.md](CONTRIBUTING.md).
