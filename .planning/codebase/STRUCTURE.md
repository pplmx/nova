# Codebase Structure

**Analysis Date:** 2026-04-30

## Directory Layout

```
nova/
├── include/                    # Public header interface
│   ├── cuda/                   # CUDA library (main library)
│   │   ├── memory/             # Layer 0: Buffer, unique_ptr, MemoryPool, KVCache
│   │   ├── device/             # Layer 1: Kernels, device utils, error handling
│   │   ├── algo/               # Algorithms: reduce, sort, flash_attention
│   │   ├── linalg/             # Linear algebra operations
│   │   ├── numeric/            # Numerical operations
│   │   ├── signal/             # Signal processing
│   │   ├── stream/             # Stream management
│   │   ├── async/              # Async copy, pinned memory, stream manager
│   │   ├── fft/                # FFT operations
│   │   ├── graph/              # Graph algorithms (BFS, PageRank, CSR)
│   │   ├── raytrace/           # Ray tracing (BVH, primitives)
│   │   ├── neural/             # Neural network primitives
│   │   │   ├── fusion/         # Fused kernels
│   │   │   ├── transformer/    # Attention mechanisms
│   │   │   ├── loss/           # Loss functions
│   │   │   └── optimizers/     # Optimizer implementations
│   │   ├── inference/          # LLM inference (scheduler, block manager)
│   │   ├── distributed/        # Distributed primitives
│   │   ├── mesh/               # Device mesh, peer copy
│   │   ├── nccl/               # NCCL collective operations
│   │   ├── pipeline/           # Pipeline parallelism
│   │   ├── mpi/                # MPI context
│   │   ├── topology/           # Device topology mapping
│   │   ├── multinode/          # Multi-node coordination
│   │   ├── checkpoint/         # Checkpoint management
│   │   ├── comm/               # Communication error recovery
│   │   ├── memory_error/       # Memory error handling
│   │   ├── preemption/         # Preemption handling
│   │   ├── memory_opt/         # Memory optimization
│   │   ├── performance/        # Profiling, autotuning, metrics
│   │   ├── production/         # Production utilities
│   │   ├── observability/      # NVTX extensions
│   │   ├── benchmark/          # Benchmarking utilities
│   │   ├── tools/              # Development tools
│   │   ├── error/              # Error handling (timeout, retry, degrade)
│   │   └── api/                # High-level API layer
│   ├── image/                  # Image processing (blur, sobel, morphology)
│   ├── parallel/               # Parallel primitives (scan, sort, histogram)
│   ├── matrix/                 # Matrix operations (add, mult, ops)
│   └── convolution/            # 2D convolution
├── src/                        # Implementation files
│   ├── cuda/                   # CUDA implementations
│   │   ├── memory/             # Memory pool implementations
│   │   ├── device/             # Device kernel implementations
│   │   ├── algo/               # Algorithm implementations
│   │   ├── linalg/             # Linear algebra implementations
│   │   ├── numeric/            # Numerical implementations
│   │   ├── signal/             # Signal processing implementations
│   │   ├── neural/             # Neural network implementations
│   │   │   ├── fusion/         # Fused kernel implementations
│   │   │   ├── transformer/    # Attention implementations
│   │   │   ├── loss/           # Loss function implementations
│   │   │   └── optimizers/     # Optimizer implementations
│   │   ├── inference/          # Inference implementations
│   │   ├── distributed/        # Distributed operation implementations
│   │   ├── mesh/               # Mesh implementations
│   │   ├── nccl/               # NCCL implementations
│   │   ├── pipeline/           # Pipeline implementations
│   │   ├── mpi/                # MPI implementations
│   │   ├── topology/           # Topology implementations
│   │   ├── multinode/          # Multi-node implementations
│   │   ├── checkpoint/         # Checkpoint implementations
│   │   ├── comm/               # Comm error recovery implementations
│   │   ├── memory_error/       # Memory error handler implementations
│   │   ├── preemption/         # Preemption handler implementations
│   │   ├── memory_opt/         # Memory optimizer implementations
│   │   ├── performance/        # Performance implementations
│   │   ├── production/         # Production utility implementations
│   │   ├── tools/              # Tool implementations
│   │   ├── error/              # Error handling implementations
│   │   ├── async/              # Async operations
│   │   ├── fft/                # FFT implementations
│   │   ├── raytrace/           # Ray tracing implementations
│   │   └── graph/              # Graph algorithm implementations
│   ├── image/                  # Image processing implementations
│   ├── parallel/               # Parallel primitive implementations
│   ├── matrix/                 # Matrix implementations
│   ├── convolution/            # Convolution implementations
│   └── main.cpp                # Demo application
├── tests/                      # Test suite
│   ├── *_test.cpp              # C++ unit tests (Google Test)
│   ├── *_test.cu               # CUDA tests (Google Test)
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── property/               # Property-based tests
│   ├── fuzz/                   # Fuzz tests
│   └── [domain]/               # Domain-specific tests
│       ├── inference/          # Inference tests
│       ├── neural/             # Neural network tests
│       ├── distributed/        # Distributed tests
│       ├── memory/             # Memory tests
│       ├── performance/        # Performance tests
│       ├── production/         # Production tests
│       └── ...
├── examples/                   # Example applications
│   ├── neural_net.cpp
│   ├── distributed_training.cpp
│   ├── graph_algorithms.cpp
│   └── image_processing.cpp
├── benchmark/                  # Benchmark suite
│   ├── benchmark_kernels.cu
│   └── CMakeLists.txt
├── cmake/                      # CMake modules
├── .github/                    # GitHub configuration
│   └── workflows/              # CI/CD pipelines
├── data/                       # Test data files
├── docs/                       # Documentation
├── .planning/                  # Planning documents
│   └── codebase/               # Codebase analysis docs
├── CMakeLists.txt              # Build configuration
├── CMakePresets.json           # CMake presets
└── Makefile                    # Build targets
```

## Key File Locations

### Memory Layer

| File | Purpose |
|------|---------|
| `include/cuda/memory/buffer.h` | Buffer<T> RAII wrapper |
| `include/cuda/memory/unique_ptr.h` | Smart pointer for device memory |
| `include/cuda/memory/memory_pool.h` | Memory pool with metrics |
| `include/cuda/memory/kv_cache_allocator.h` | KV-cache block allocator |
| `src/cuda/memory/` | Memory pool implementations |

### Device Layer

| File | Purpose |
|------|---------|
| `include/cuda/device/error.h` | CUDA_CHECK macro, exceptions |
| `include/cuda/device/device_utils.h` | Warp/block reduction primitives |
| `include/cuda/device/reduce_kernels.h` | Kernel declarations |
| `include/cuda/device/cublas_context.h` | cuBLAS context management |
| `src/cuda/device/` | Kernel implementations |

### Inference Layer

| File | Purpose |
|------|---------|
| `include/cuda/inference/scheduler.h` | Sequence scheduling |
| `include/cuda/inference/block_manager.h` | Paged KV-cache management |
| `include/cuda/algo/flash_attention.h` | Flash attention implementation |
| `src/cuda/inference/` | Inference implementations |
| `tests/inference/` | Inference tests |

### Neural Layer

| File | Purpose |
|------|---------|
| `include/cuda/neural/matmul.h` | Matrix multiplication |
| `include/cuda/neural/activations.h` | Activation functions |
| `include/cuda/neural/softmax.h` | Softmax operations |
| `include/cuda/neural/layer_norm.h` | Layer normalization |
| `include/cuda/neural/transformer/attention.h` | Multi-head attention |
| `include/cuda/neural/loss/loss_functions.h` | Loss functions |
| `include/cuda/neural/optimizers/optimizers.h` | Optimizers |
| `include/cuda/neural/fusion/fused_matmul_bias_act.h` | Fused kernels |
| `include/cuda/neural/tensor_parallel_*.h` | Tensor parallelism |
| `src/cuda/neural/` | Neural implementations |
| `tests/neural/` | Neural tests |

### Distributed Layer

| File | Purpose |
|------|---------|
| `include/cuda/nccl/nccl_context.h` | NCCL context (singleton) |
| `include/cuda/nccl/nccl_ops.h` | Collective operations |
| `include/cuda/distributed/common.h` | Distributed utilities |
| `include/cuda/mpi/mpi_context.h` | MPI context |
| `include/cuda/multinode/multi_node_context.h` | Multi-node coordination |
| `src/cuda/nccl/` | NCCL implementations |
| `src/cuda/distributed/` | Distributed implementations |
| `tests/cuda/nccl/` | NCCL tests |
| `tests/distributed/` | Distributed tests |

### Production Layer

| File | Purpose |
|------|---------|
| `include/cuda/production/graph_executor.h` | CUDA graphs |
| `include/cuda/performance/profiler.h` | Performance profiling |
| `include/cuda/performance/autotuner.h` | Autotuning |
| `include/cuda/production/health_metrics.h` | Health monitoring |
| `src/cuda/production/` | Production implementations |
| `tests/production/` | Production tests |

## Naming Conventions

### Files

| Pattern | Example |
|---------|---------|
| C++ headers | `*.h` |
| CUDA headers | `*.h`, `*.cuh` |
| CUDA sources | `*.cu` |
| C++ sources | `*.cpp` |
| Test files | `*_test.cpp`, `*_test.cu`, `*_edge_test.cpp` |
| Edge case tests | `*_edge_test.cpp` |
| Integration tests | `*_integration_test.cpp` |

### Classes and Types

| Pattern | Example |
|---------|---------|
| Classes | `PascalCase` (e.g., `MemoryPool`, `BlockManager`) |
| Structs | `PascalCase` (e.g., `SchedulerConfig`) |
| Enums | `PascalCase` (e.g., `SequenceState`) |
| Type aliases | `PascalCase` or `_t` suffix |
| Namespaces | `snake_case` (e.g., `cuda::memory`) |

### Functions and Variables

| Pattern | Example |
|---------|---------|
| Functions | `snake_case` (e.g., `reduce_sum`, `create_sequence`) |
| Member variables | `snake_case_` with trailing underscore (e.g., `config_`, `mutex_`) |
| Static variables | `snake_case_` with trailing underscore |
| Constants | `kPascalCase` (e.g., `kMaxBatchSize`) |

## Where to Add New Code

### New Algorithm/Operation

1. **Header:** `include/cuda/{domain}/operation_name.h`
2. **Implementation:** `src/cuda/{domain}/operation_name.cu` (or `.cpp`)
3. **Test:** `tests/{domain}/operation_name_test.cpp` or `*_test.cu`
4. **Registration:** Add to `src/cuda/{domain}/` CMakeLists or parent CMakeLists.txt

### New Neural Network Component

1. **Header:** `include/cuda/neural/{submodule}/component.h`
2. **Implementation:** `src/cuda/neural/{submodule}/component.cu`
3. **Test:** `tests/neural/{submodule}/component_test.cpp`
4. **CMake:** Add source to `NEURAL_SOURCES` in root CMakeLists.txt

### New Distributed Primitive

1. **Header:** `include/cuda/{nccl,distributed,mpi}/primitive.h`
2. **Implementation:** `src/cuda/{nccl,distributed,mpi}/primitive.cpp`
3. **Test:** `tests/{nccl,distributed,mpi}/test_primitive.cpp`

### New Inference Component

1. **Header:** `include/cuda/inference/component.h`
2. **Implementation:** `src/cuda/inference/component.cpp`
3. **Test:** `tests/inference/component_test.cpp`

## Source vs Test Separation

```
tests/
├── *_test.cpp                    # Unit tests (co-located by domain)
├── *_test.cu                     # CUDA unit tests
├── *_edge_test.cpp               # Edge case tests
├── *_integration_test.cpp        # Integration tests
├── *_fuzz.cpp                    # Fuzz tests
├── unit/                         # Additional unit tests
├── integration/                  # Additional integration tests
├── property/                     # Property-based tests
├── production/                   # Production tests
├── inference/                    # Inference-specific tests
├── neural/                       # Neural network tests
├── distributed/                  # Distributed tests
├── memory/                       # Memory tests
├── performance/                  # Performance tests
└── [other domains]/              # Domain-specific tests
```

## Build System Structure

### Core Libraries

```
cmake_target          description                  sources
─────────────────────────────────────────────────────────────────
cuda_memory           Memory management            INTERFACE
cuda_device           Device kernels               INTERFACE
cuda_algo             Algorithms                   INTERFACE
cuda_nccl             NCCL collectives             STATIC
cuda_mpi              MPI support                  STATIC (optional)
cuda_topology         Device topology              STATIC
cuda_multinode        Multi-node support           STATIC
cuda_checkpoint       Checkpoint management        STATIC
cuda_comm             Comm error recovery          STATIC
cuda_memory_error     Memory error handling        STATIC
cuda_preemption       Preemption support           STATIC
cuda_impl             Main implementation          STATIC (all sources)
```

### Build Configuration

```bash
# Basic build
cmake -G Ninja -B build
cmake --build build --parallel

# With NCCL (default)
cmake -G Ninja -B build -DNOVA_ENABLE_NCCL=ON

# With MPI
cmake -G Ninja -B build -DNOVA_ENABLE_MPI=ON

# Run tests
ctest --parallel 16
```

### Test Execution

```bash
# All tests
ctest

# Specific test
./build/bin/nova-tests --gtest_filter="*Scheduler*"

# With coverage
cmake -B build -DNOVA_COVERAGE=ON
cmake --build build
ctest
```

## Special Directories

**`data/`**
- Purpose: Test patterns, STB image library
- Generated: No
- Committed: Yes

**`.planning/`**
- Purpose: Planning and codebase analysis documents
- Contains: ARCHITECTURE.md, STRUCTURE.md, ROADMAP.md, etc.

**`build*/`**
- Purpose: Build output directories
- Generated: Yes
- Committed: No (gitignored)

**`docs/superpowers/`**
- Purpose: Design specs and implementation plans
- Contains: RFCs, design documents

---

*Structure analysis: 2026-04-30*
