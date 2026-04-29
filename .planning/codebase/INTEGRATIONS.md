# External Integrations

**Analysis Date:** 2026-04-30

## NVIDIA CUDA Libraries

**cuBLAS (Dense Linear Algebra):**
- Purpose: Matrix multiplication, BLAS operations
- Target: `CUDA::cublas`
- Usage: `cuda_algo` library in `include/cuda/algo/`
- Key operations: GEMM, GEMV, batched matmul

**cuSOLVER (Dense Linear Solvers):**
- Purpose: Linear system solvers, eigenvalue problems
- Target: `CUDA::cusolver`
- Linked: `cuda_impl` library

**cuRAND (Random Number Generation):**
- Purpose: GPU random number generation
- Target: `CUDA::curand`
- Linked: `cuda_impl` library

**cuFFT (Fast Fourier Transform):**
- Purpose: FFT computations on GPU
- Target: `CUDA::cufft`
- Sources: `src/cuda/fft/fft.cu`
- Headers: `include/cuda/fft/`
- Tests: `tests/fft/` (fft_plan_test, fft_inverse_test, fft_accuracy_test)

## NCCL (Multi-GPU Communication)

**Purpose:** Optimized multi-GPU collective operations

**Version Requirement:** NCCL 2.25+

**Detection:** Custom CMake finder in `cmake/FindNCCL.cmake`

**Search Paths (in priority order):**
1. `NCCL_DIR` environment variable
2. `/usr/local/nccl`
3. `/usr/local/cuda/nccl`
4. `/opt/nccl`

**Collective Operations:**
- `nccl_all_reduce.cpp` - All-reduce across GPUs
- `nccl_broadcast.cpp` - Broadcast to all GPUs
- `nccl_barrier.cpp` - Synchronization barrier
- `nccl_all_gather.cpp` - All-gather operation
- `nccl_reduce_scatter.cpp` - Reduce-scatter operation
- `nccl_group.cpp` - Group communication support

**Compile Definition:**
- `NOVA_NCCL_ENABLED=1` when NCCL is found
- `NOVA_NCCL_ENABLED=0` when NCCL is disabled

**Build Option:** `NOVA_ENABLE_NCCL` (default: ON)

**Files:**
- Headers: `include/cuda/nccl/`
- Sources: `src/cuda/nccl/`
- Tests: `tests/cuda/nccl/test_nccl_collectives.cpp`

## MPI (Multi-Node Communication)

**Purpose:** Distributed training across multiple nodes

**Version Requirement:** MPI 3.1+

**Supported Implementations:** OpenMPI, MPICH, Intel MPI

**Detection:** Custom CMake finder in `cmake/FindMPI.cmake`

**Search Paths:**
- `MPI_DIR` environment variable
- `/usr/lib/x86_64-linux-gnu`
- `/usr/lib64`
- `/usr/local/lib`

**Features:**
- MPI bootstrapping for NCCL
- Multi-node context management
- `mpi_context.cpp` - MPI initialization and cleanup

**Compile Definition:**
- `NOVA_MPI_ENABLED=1` when MPI is found
- `NOVA_MPI_ENABLED=0` when MPI is disabled

**Build Option:** `NOVA_ENABLE_MPI` (default: OFF)

**Files:**
- Headers: `include/cuda/mpi/`
- Sources: `src/cuda/mpi/`
- Tests: `tests/mpi/test_mpi_context.cpp`

## OpenSSL (Cryptography)

**Purpose:** Checkpoint encryption for model persistence

**Package:** `OpenSSL::Crypto` (OpenSSL cryptography library)

**Usage:** `cuda_checkpoint` library

**Features:**
- Checkpoint file encryption
- Secure serialization of model state

**Files:**
- Headers: `include/cuda/checkpoint/`
- Sources: `src/cuda/checkpoint/checkpoint_manager.cpp`

## Google Test (Testing Framework)

**Version:** v1.17.0

**Repository:** `https://github.com/google/googletest.git`

**CMake Setup:**
```cmake
FetchContent_Declare(googletest GIT_TAG v1.17.0)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
```

**Test Components:**
- `GTest::gtest_main` - Main test entry point
- `GTest::gmock` - Mocking framework

**Test Executables:**
- `nova-tests` - Main test suite (90+ test files)
- `test_patterns-tests` - Pattern validation tests

## Threading

**Package:** `Threads::Threads` (CMake's FindThreads)

**Usage:**
- Error recovery handlers (`cuda_comm`)
- Memory error handling (`cuda_memory_error`)
- Preemption handling (`cuda_preemption`)

## Docker

**Purpose:** Containerized build and deployment

**Base Images:**
- Builder: `rockylinux:9`
- Runtime: `rockylinux:9-minimal`

**Build Features:**
- Multi-stage build for minimal image size
- Static linking for portability
- Rocky Linux 9 for enterprise compatibility

**Image Target:**
```bash
docker image build -t nova .
```

## GitHub Actions (CI/CD)

**Workflows:** `.github/workflows/`

**Integrations:**
- Coverage reporting via Coveralls
- Badge: `https://coveralls.io/repos/github/pplmx/nova/badge.svg`

## Code Quality Services

**Coveralls:** Code coverage tracking and reporting

**Commitizen:** Conventional commit enforcement

**rumdl:** Markdown linting for documentation

## Third-Party CMake Finders

**Custom Find Modules:**
| File | Package | Purpose |
|------|---------|---------|
| `cmake/FindNCCL.cmake` | NCCL 2.25+ | NVIDIA NCCL detection |
| `cmake/FindMPI.cmake` | MPI 3.1+ | MPI implementation detection |
| `cmake/FindXXX.cmake` | (template) | Template for future integrations |

## Build Dependencies

**System Packages (Rocky Linux 9):**
```dockerfile
cmake gcc gcc-c++ make libstdc++-static glibc-static
```

**Optional Runtime:**
- CUDA Toolkit (driver and runtime)
- NCCL library
- MPI implementation

## Library Architecture

**Core Libraries:**
```
cuda_impl (STATIC)
├── cuda_device (INTERFACE)
├── cuda_algo (INTERFACE)
│   └── CUDA::cublas
├── cuda_nccl (STATIC, optional)
│   └── NCCL::nccl
├── cuda_mpi (STATIC, optional)
│   └── MPI::MPI_CXX
├── cuda_topology (STATIC)
├── cuda_multinode (STATIC)
├── cuda_checkpoint (STATIC)
│   └── OpenSSL::Crypto
├── cuda_comm (STATIC)
├── cuda_memory_error (STATIC)
├── cuda_preemption (STATIC)
└── CUDA::cudart, cublas, cusolver, curand, cufft
```

---

*Integration audit: 2026-04-30*
