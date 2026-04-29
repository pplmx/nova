# Technology Stack

**Analysis Date:** 2026-04-30

## Languages

**Primary:**
- **C++23** - Core library implementation
- **CUDA 20** - GPU kernel programming (nvcc)

**Build Configuration:**
- `CMAKE_CXX_STANDARD 23` - C++ standard requirement
- `CMAKE_CUDA_STANDARD 20` - CUDA standard requirement

## Compiler Toolchain

**Build System:**
- **CMake 4.0+** - Primary build system (minimum 3.28 for presets)
- **Ninja** - Recommended generator for faster builds
- **Make** - Alternative generator

**CUDA Compiler:**
- **NVIDIA nvcc** - CUDA compiler (detected via `CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"`)
- **GPU Parallel Jobs:** `--threads ${NCPU}` flag for parallel kernel compilation

**Host Compiler:**
- GCC or Clang (for C++ compilation)
- Detected via `CMAKE_CXX_COMPILER_ID`

## GPU Support

**Compute Capabilities:**
- `CMAKE_CUDA_ARCHITECTURES 60 70 80 90`
  - **60** - NVIDIA Pascal (GTX 10xx, Tesla P100)
  - **70** - NVIDIA Volta (Tesla V100)
  - **80** - NVIDIA Ampere (RTX 30xx, A100, H100)
  - **90** - NVIDIA Hopper (H100, H200)

**CUDA Toolkit Components:**
- `CUDA::cudart` - CUDA Runtime API
- `CUDA::cublas` - cuBLAS (dense linear algebra)
- `CUDA::cusolver` - cuSOLVER (dense linear solvers)
- `CUDA::curand` - cuRAND (random number generation)
- `CUDA::cufft` - cuFFT (Fast Fourier Transform)

## Key Dependencies

**Required:**
- **CUDAToolkit** - Found via `find_package(CUDAToolkit REQUIRED)`
- **OpenSSL** - Found via `find_package(OpenSSL REQUIRED)` for checkpoint encryption

**Optional (Multi-GPU/Multi-Node):**
- **NCCL 2.25+** - NVIDIA Collective Communications Library
  - Custom finder: `cmake/FindNCCL.cmake`
  - Enable: `NOVA_ENABLE_NCCL=ON` (default: ON)
- **MPI 3.1+** - Message Passing Interface
  - Custom finder: `cmake/FindMPI.cmake`
  - Enable: `NOVA_ENABLE_MPI=OFF` (default: OFF)
  - Supports: OpenMPI, MPICH, Intel MPI

**System Libraries:**
- **Threads::Threads** - POSIX threads (for error recovery, preemption handlers)

## Build Options

| Option | Default | Purpose |
|--------|---------|---------|
| `NOVA_ENABLE_NCCL` | ON | Multi-GPU collectives |
| `NOVA_ENABLE_MPI` | OFF | Multi-node support |
| `NOVA_ENABLE_UNITY_BUILD` | ON | Faster compilation |
| `NOVA_USE_CCACHE` | OFF | Compiler caching |
| `NOVA_COVERAGE` | OFF | Code coverage instrumentation |

## Testing Frameworks

**Google Test (googletest):**
- **Version:** v1.17.0
- **Setup:** FetchContent from `https://github.com/google/googletest.git`
- **Components:** `GTest::gtest_main`, `GTest::gmock`
- **Configuration:** `gtest_force_shared_crt ON`

**Test Types:**
- **Unit Tests** - `nova-tests` executable (90+ test files)
- **Pattern Tests** - `test_patterns-tests` executable
- **Fuzz Tests** - libFuzzer (Clang only, opt-in via `NOVA_BUILD_FUZZ_TESTS=ON`)
- **Property-Based Tests** - Custom framework in `tests/property/`

**Test Execution:**
```bash
ctest -j16                              # Parallel tests (capped at 16 for GPU memory)
./build/bin/nova-tests --gtest_filter='TestSuite.TestName'  # Single test
```

## Code Quality Tools

**Formatting:**
- **clang-format** - `.clang-format` file
  - Based on Google style
  - Standard: C++20
  - Column limit: 180
  - 4-space indentation

**Linting:**
- **clang-tidy** - `.clang-tidy` file
  - Checks enabled: `bugprone-*`, `readability-*`, `performance-*`, `modernize-*`, `cppcoreguidelines-*`, `clang-analyzer-*`, `portability-*`, `security-*`, `misc-*`, `google-*`
  - Cognitive complexity threshold: 35

**Pre-commit Hooks:**
- **commitizen** v4.13.10 - Conventional commits
- **rumdl** - Markdown linting
- Built-in: end-of-file-fixer, trailing-whitespace, check-toml, check-yaml, check-merge-conflict, mixed-line-ending

## Build Presets

**CMakePresets.json:**
| Preset | Build Type | NCCL | MPI | Unity | ccache |
|--------|------------|------|-----|-------|--------|
| `dev` | Debug | OFF | OFF | OFF | ON |
| `release` | Release | ON | ON | ON | OFF |
| `ci` | Release | OFF | OFF | ON | OFF |

## Containerization

**Docker:**
- Base image: `rockylinux:9` (builder), `rockylinux:9-minimal` (runtime)
- Multi-stage build for minimal runtime image
- Static linking for portable deployment

**CI/CD:**
- GitHub Actions - CI pipeline
- Coveralls - Coverage tracking

## Configuration Files

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Main build configuration |
| `CMakePresets.json` | Build presets |
| `cmake/FindNCCL.cmake` | NCCL detection |
| `cmake/FindMPI.cmake` | MPI detection |
| `.clang-format` | Code formatting |
| `.clang-tidy` | Static analysis |
| `.pre-commit-config.yaml` | Git hooks |
| `Dockerfile` | Container build |

---

*Stack analysis: 2026-04-30*
