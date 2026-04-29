# Architecture

**Analysis Date:** 2026-04-30

## System Overview

Nova is a CUDA parallel algorithms library with a layered architecture supporting inference workloads, distributed training, and neural network operations.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Application Layer                                     │
│         Examples: neural_net.cpp, distributed_training.cpp                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                        Inference Layer                                       │
│    scheduler.h, block_manager.h, flash_attention.h                          │
│    (Sequence scheduling, KV-cache, PagedAttention)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                        Neural Layer                                          │
│    matmul.h, activations.h, softmax.h, transformer/attention.h              │
│    loss_functions.h, optimizers.h, tensor_parallel_*.h                      │
├────────────────────────────────┬────────────────────────────────────────────┤
│     Distributed Layer          │           Production Layer                 │
│  nccl_*.h, mpi_context.h,      │   graph_executor.h, profiler.h            │
│  distributed/*.h, mesh/*.h     │   autotuner.h, health_metrics.h           │
├────────────────────────────────┴────────────────────────────────────────────┤
│                        Core CUDA Layer                                       │
│  memory/*.h, device/*.h, algo/*.h, linalg/*.h, fft/*.h, graph/*.h           │
├─────────────────────────────────────────────────────────────────────────────┤
│                      CUDA Runtime Integration                               │
│           cuBLAS, cuFFT, cuSOLVER, cuRAND, NCCL, MPI                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | Primary Files |
|-----------|----------------|---------------|
| **Memory** | GPU memory allocation, RAII wrappers, pooling | `include/cuda/memory/buffer.h`, `memory_pool.h` |
| **Device** | Low-level kernels, error handling | `include/cuda/device/error.h`, `device_utils.h` |
| **Algo** | Algorithm implementations (reduce, sort, flash attention) | `include/cuda/algo/reduce.h`, `flash_attention.h` |
| **Stream** | CUDA stream management | `include/cuda/stream/stream.h` |
| **Inference** | Sequence scheduling, KV-cache, block management | `include/cuda/inference/scheduler.h`, `block_manager.h` |
| **Neural** | ML primitives (matmul, activations, attention, losses) | `include/cuda/neural/*.h` |
| **NCCL** | Multi-GPU collective operations | `include/cuda/nccl/nccl_*.h` |
| **Distributed** | Multi-node communication primitives | `include/cuda/distributed/*.h` |
| **Production** | Graph execution, profiling, health monitoring | `include/cuda/production/*.h` |

## Pattern Overview

**Overall:** Layered Architecture with modular domain separation

**Key Characteristics:**
- Header-only utilities for zero linking overhead
- RAII memory management via `Buffer<T>` template
- Stream-based asynchronous CUDA operations
- Singleton managers for centralized state (e.g., `NcclContext::instance()`)
- Dependency injection pattern for testability

## Layers

### Memory Foundation Layer

**Purpose:** GPU memory allocation and management with RAII semantics

**Location:** `include/cuda/memory/`

**Key Abstractions:**
- `cuda::memory::Buffer<T>` - RAII wrapper for device memory with copy_from/copy_to
- `cuda::memory::unique_ptr<T>` - Smart pointer for device memory
- `cuda::memory::MemoryPool` - Memory pool with metrics and defragmentation
- `cuda::memory::KVCacheAllocator` - Specialized allocator for KV-cache blocks

**Dependencies:** CUDA runtime only

### Device Layer

**Purpose:** Low-level CUDA kernels and device utilities

**Location:** `include/cuda/device/`

**Key Abstractions:**
- `CUDA_CHECK()` macro for automatic error checking
- `CudaException`, `CublasException` exception hierarchy
- Warp-level and block-level reduction primitives
- cuBLAS context management

**Dependencies:** Memory layer

### Algorithm Layer

**Purpose:** Parallel algorithm implementations with memory management

**Location:** `include/cuda/algo/`

**Key Abstractions:**
- `reduce_sum()`, `reduce_max()`, `reduce_min()`
- `sort()` with configurable comparison
- `FlashAttention` for efficient transformer attention
- `KernelLauncher` for kernel execution

**Dependencies:** Memory and Device layers

### Stream Layer

**Purpose:** CUDA stream abstraction with RAII semantics

**Location:** `include/cuda/stream/`

**Key Abstractions:**
- `cuda::stream::Stream` - RAII wrapper for cudaStream_t
- Synchronization primitives (synchronize, query)

### Inference Layer

**Purpose:** LLM inference optimization (vLLM-style architecture)

**Location:** `include/cuda/inference/`

**Key Abstractions:**
- `Scheduler` - Manages sequence batching and scheduling
- `SequenceManager` - Tracks sequence states (Waiting, Running, Finished, Evicted)
- `BlockManager` - Paged KV-cache block allocation
- `PagedAttention` - Flash attention with block-based KV-cache

**Configuration:**
```cpp
struct SchedulerConfig {
    int max_batch_size = 32;
    int max_sequence_length = 8192;
    int prefill_batch_size = 8;
    bool enable_continuous_batching = true;
    bool enable_prefix_caching = true;
    int num_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 128;
    int block_size = 16;
};
```

### Neural Layer

**Purpose:** Neural network primitives for training and inference

**Location:** `include/cuda/neural/`

**Submodules:**
- `matmul.h` - Matrix multiplication via cuBLAS
- `activations.h` - ReLU, GELU, SiLU, sigmoid, tanh
- `softmax.h` - Full and masked softmax
- `layer_norm.h`, `sync_batch_norm.h` - Normalization layers
- `transformer/attention.h` - Multi-head attention with positional encoding
- `loss/loss_functions.h` - Cross-entropy, focal loss, contrastive loss
- `optimizers/optimizers.h` - AdamW, LAMB with gradient clipping
- `fusion/` - Fused kernels (matmul + bias + activation)
- `tensor_parallel_*.h` - Tensor parallelism for multi-GPU

### Distributed Layer

**Purpose:** Multi-GPU and multi-node communication

**Location:** `include/cuda/nccl/`, `include/cuda/distributed/`, `include/cuda/mpi/`

**Key Abstractions:**
- `NcclContext` - NCCL communicator pool with singleton fallback
- Collective operations: all_reduce, broadcast, barrier, all_gather, reduce_scatter
- `MeshStreams` - Per-device CUDA streams for collectives
- `MpiContext` - MPI context for multi-node bootstrapping
- `MultiNodeContext` - High-level multi-node coordination

**Pattern:** Dependency injection with singleton fallback (per D-01)

### Production Layer

**Purpose:** Production readiness and observability

**Location:** `include/cuda/production/`, `include/cuda/performance/`

**Key Abstractions:**
- `GraphExecutor` - CUDA graph capture and execution
- `Profiler` - Performance profiling with Chrome trace export
- `Autotuner` - Hardware-aware parameter optimization
- `HealthMetrics` - System health monitoring
- `PriorityStream` - Priority-based stream scheduling

## Data Flow

### Primary Inference Flow

```
User Request
    │
    ▼
Scheduler::add_request() ──────► SequenceManager (creates sequence state)
    │                                      │
    │                                      ▼
    │                              BlockManager::create_sequence()
    │                                      │
    │                                      ▼
    │                              KVCacheAllocator (allocates blocks)
    │                                      │
    ▼                                      ▼
Scheduler::get_batch() ◄───── Recompose batch (continuous batching)
    │
    ▼
BlockManager::forward_batch()
    │
    ├──► Update block tables to GPU
    │
    ▼
PagedAttention::forward() ──────► FlashAttention (kernel launch)
    │                                      │
    │                                      ▼
    │                              Device memory (KV-cache blocks)
    │
    ▼
Output buffer
```

### Memory Allocation Flow

```
Buffer<T> constructor
    │
    ▼
cudaMalloc(&data_, size * sizeof(T))
    │
    ▼
Buffer destructor (RAII)
    │
    ▼
cudaFree(data_)
```

## Key Abstractions

### Buffer<T> (Memory Layer)

- **Purpose:** RAII wrapper for GPU memory
- **Location:** `include/cuda/memory/buffer.h`
- **Pattern:** Move semantics, automatic deallocation

```cpp
cuda::memory::Buffer<float> buf(1024);
buf.copy_from(host_data, 1024);
// Automatic cleanup in destructor
```

### Stream (Stream Layer)

- **Purpose:** RAII wrapper for CUDA streams
- **Location:** `include/cuda/stream/stream.h`
- **Pattern:** RAII with synchronization support

### Scheduler (Inference Layer)

- **Purpose:** Manages inference batching and sequence scheduling
- **Location:** `include/cuda/inference/scheduler.h`
- **Pattern:** Continuous batching with dynamic batch composition

### NcclContext (Distributed Layer)

- **Purpose:** Centralized NCCL communicator management
- **Location:** `include/cuda/nccl/nccl_context.h`
- **Pattern:** Singleton with dependency injection

## Entry Points

**Main Application:**
- `src/main.cpp` - Demo application showcasing library features

**Examples:**
- `examples/neural_net.cpp` - Neural network primitives usage
- `examples/distributed_training.cpp` - Multi-GPU training example
- `examples/graph_algorithms.cpp` - Graph algorithm examples
- `examples/image_processing.cpp` - Image processing examples

**Library:**
- `include/cuda/` - Public header interface
- `lib/libcuda_impl.a` - Core implementation library

## Architectural Constraints

- **Threading:** Single-threaded event loop; thread-safety via mutexes in managers
- **Global state:** Singleton patterns in `NcclContext`, `MeshStreams`, `DeviceMesh`
- **Memory model:** Explicit GPU memory management; no unified memory assumed
- **Error handling:** Exception-based with `CUDA_CHECK()` macro; no error codes

## Error Handling

**Strategy:** Exception-based with comprehensive error context

**Patterns:**
- `CUDA_CHECK(cudaMalloc(...))` - Automatic exception on CUDA errors
- `NCCL_CHECK(ncclCommInitAll(...))` - NCCL-specific error checking
- Custom exception classes: `CudaException`, `CublasException`, `NcclException`

## Cross-Cutting Concerns

**Logging:** NVTX extensions for profiling (`include/cuda/observability/nvtx_extensions.h`)

**Validation:** Configuration validation in constructors; range checking in accessors

**Performance:**
- Memory pooling for allocation reuse
- CUDA graphs for kernel launch optimization
- Continuous batching for GPU utilization

---

*Architecture analysis: 2026-04-30*
