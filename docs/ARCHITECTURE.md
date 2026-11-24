# Nova Architecture

Nova uses a five-layer architecture designed for education, research, and production use.

## Layer Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: Domain-Specific (Sparse, Quantize, GNN)               │
│  nova::sparse, nova::quantize, nova::gnn                        │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: High-Level API (STL-style)                            │
│  cuda::algo::reduce(), cuda::stream::Stream, memory::Buffer    │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: Algorithm Wrappers                                    │
│  cuda::device::reduce_optimized_kernel, memory management       │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Device Kernels                                        │
│  __global__ kernels, warp primitives                            │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│  Layer 0: Memory Foundation                                     │
│  cuda::memory::Buffer, MemoryPool, distributed pools            │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

| Layer       | Namespace                      | Purpose                                                 | Dependencies   |
| ----------- | ------------------------------ | ------------------------------------------------------- | -------------- |
| **Layer 0** | `cuda::memory`                 | Memory allocation, RAII, pooling, distributed pools     | CUDA runtime   |
| **Layer 1** | `cuda::device`                 | Pure CUDA kernels, warp primitives                      | Layer 0        |
| **Layer 2** | `cuda::algo`                   | Algorithm orchestration, device memory management       | Layers 0, 1    |
| **Layer 3** | `cuda::stream`, `cuda::memory` | Stream/event wrappers; device buffers (no DeviceVector) | Layers 0, 1, 2 |
| **Layer 4** | `nova::sparse/quantize/gnn`    | Domain-specific operations                              | Layers 0-3     |

## Domain Modules

### nova::sparse

Sparse matrix operations including:

- Storage formats: CSR, ELL, SELL, HYB
- SpMV operations
- Iterative solvers: CG, GMRES, BiCGSTAB
- Preconditioners: Jacobi, ILU(0)

### nova::quantize

Quantization support:

- INT8 quantization with calibration
- FP8 support (H100/H200)
- Quantization-aware training (QAT)
- Quantized GEMM operations

### nova::gnn

Graph neural network operations:

- Graph sampling (neighbor, k-hop)
- Message passing primitives
- Graph attention networks

## Directory Structure

```text
include/cuda/
├── memory/              # Layer 0: Memory Foundation
│   ├── buffer.h         # cuda::memory::Buffer<T>
│   ├── distributed_pool.h
│   └── memory_pool.h
├── device/              # Layer 1: Device Kernels
│   ├── reduce_kernels.h
│   └── device_utils.h   # CUDA_CHECK, warp_reduce
├── algo/                # Layer 2: Algorithm Wrappers
│   ├── reduce.h
│   └── sort.h
├── api/                 # Layer 3: High-Level API
│   ├── device_vector.h
│   └── stream.h
├── sparse/              # Layer 4: Sparse Operations
├── quantize/            # Layer 4: Quantization
├── gnn/                 # Layer 4: Graph Neural Networks
├── error/               # Error Handling
└── observability/       # Logging, Profiling

src/cuda/
├── memory/
├── device/
├── algo/
├── sparse/
├── quantize/
└── gnn/
```

## Thread Safety

- **cuda::memory::Buffer**: Not thread-safe; use external synchronization
- **cuda::memory::DistributedMemoryPool**: Thread-safe via mutex
- **cuda::memory::MemoryPool**: Thread-safe per-device pools
- **nova::error::timeout_manager**: Thread-safe singleton

## Memory Model

```text
Host Memory                    Device Memory
     │                              │
     │  cuda::memory::Buffer        │
     │  (unified view)              │
     ├──────────────────────────────┤
     │                              │
     │  copy_from()                 │  cudaMemcpy
     │  copy_to()                   │
     │                              │
     ▼                              ▼
┌─────────┐                  ┌─────────┐
│  CPU    │                  │   GPU   │
└─────────┘                  └─────────┘
```

## Error Handling

All Nova functions use structured error handling:

```cpp
// Throws on error with context
NOVA_CHECK(cudaMalloc(&ptr, size));

// Or returns error codes. The namespace is nova::error, not cuda::error: every header under
// include/cuda/error/ declares nova::error, which is why this snippet did not compile.
std::error_code ec = nova::error::make_error_code(err);

// Recovery hints live on timeout_error_category. Both cuda_category() and timeout_category()
// return std::error_category&, which has no recovery_hint(), so the concrete type is needed.
const auto& category = static_cast<const nova::error::timeout_error_category&>(nova::error::timeout_category());
std::string_view hint = category.recovery_hint(static_cast<int>(err));
```
