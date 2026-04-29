# Technology Stack Additions for v2.7

**Analysis Date:** 2026-04-30
**Confidence:** HIGH
**Sources:** NVIDIA official documentation, GitHub releases

## Executive Summary

v2.7 adds three capability areas: robustness testing, performance profiling, and advanced algorithms. Most tooling is either already in the stack or available from NVIDIA at no additional cost. **One critical migration required:** CUB has been archived and moved to the unified CCCL repository.

---

## 1. What's Already in the Stack

The project already has significant tooling. Do NOT re-add:

| Existing | Version | Source | Already Provides |
|----------|---------|--------|------------------|
| Google Test | v1.17.0 | FetchContent | Unit tests |
| libFuzzer | Clang-only | Built-in | Fuzz testing |
| Property-based tests | Custom | In-repo | Property testing |
| NVTX | Header-only | CUDA toolkit | Profiling annotations |
| CUDA Events | Built-in | CUDA toolkit | Kernel timing |
| Google Benchmark | v1.9.0 | External | Microbenchmarks |
| Error injection framework | Custom | v2.4 shipped | Chaos testing |
| Memory pressure stress tests | Custom | v2.4 shipped | Stress testing |

---

## 2. Recommended Additions

### 2.1 Robustness Testing

#### Option A: NVIDIA Compute Sanitizer (Recommended)
**Status:** Part of CUDA Toolkit — no installation needed
**Confidence:** HIGH

Compute Sanitizer is a functional correctness checking suite included in CUDA 12+. It provides:

| Tool | Purpose | Use Case |
|------|---------|----------|
| `memcheck` | Out-of-bounds, misaligned access detection | Memory safety validation |
| `racecheck` | Shared memory data race detection | Concurrent kernel validation |
| `initcheck` | Uninitialized memory access detection | Initialization bug finding |
| `synccheck` | Invalid synchronization detection | Async operation validation |

**Integration:**
```bash
# Replace direct execution with sanitizer wrapper
cuda-compute-sanitizer --tool memcheck ./bin/nova-tests
```

**CMake Integration:**
```cmake
# Optional sanitizer builds
option(NOVA_ENABLE_SANITIZERS "Enable sanitizer builds for testing" OFF)
if(NOVA_ENABLE_SANITIZERS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()
```

**Why not AddressSanitizer:** ASAN does not work with CUDA kernels. Compute Sanitizer is the only option for GPU memory checking.

**Source:** [NVIDIA Compute Sanitizer Documentation](https://docs.nvidia.com/compute-sanitizer/)

#### Option B: Tracy Profiler (Optional - Open Source)
**Status:** v0.13.1 (Dec 2025), Apache 2.0
**Confidence:** MEDIUM

Open-source profiler with CUDA support, nanosecond resolution, real-time telemetry.

**Use if:** You want a free alternative to Nsight for continuous profiling in CI or local development.

**Not required if:** Nsight Systems provides sufficient capability (it usually does for CUDA development).

**Source:** [Tracy Profiler GitHub](https://github.com/wolfpld/tracy)

---

### 2.2 Performance Profiling

#### NVIDIA Nsight Compute CLI
**Status:** v13.2 (March 2026), Part of CUDA Toolkit
**Confidence:** HIGH

Kernel-level profiler with detailed metrics. Already partially referenced via NVBench headers in v2.4.

**Installation:** Included in CUDA Toolkit
```bash
# Profile a kernel
ncu --set full ./bin/nova-benchmark --benchmark_filter=BM_Matmul
```

**Python Report Interface for CI:**
```bash
# Generate JSON report for trend analysis
ncu --export json --output profile.ncu-rep ./bin/nova-benchmark
```

**CMake:**
```cmake
find_package(CUDAToolkit REQUIRED)
# ncu CLI available via CUDA_TOOLKIT_TARGET_DIR
```

**Source:** [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

#### NVIDIA Nsight Systems
**Status:** v2026.2 (March 2026), Part of CUDA Toolkit
**Confidence:** HIGH

System-wide timeline profiler. Excellent for understanding multi-stream interactions and GPU utilization.

**Installation:** Separate download from [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) (free registration required).

**Use for:**
- Multi-stream timeline visualization
- CPU-GPU interaction analysis
- Identifying pipeline bottlenecks

**Source:** [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)

---

### 2.3 Algorithm Libraries

#### Critical: CUB → CCCL Migration Required
**Status:** CUB 2.1.0 archived, now part of CCCL
**Confidence:** HIGH

**CUB has been moved to the unified [NVIDIA CCCL (CUDA C++ Core Libraries)](https://github.com/nvidia/cccl) repository.**

**Action Required:**
1. Update CMake from CUB to CCCL
2. Update includes from `<cub/cub.cuh>` to `<cub/cub.cuh>` (CCCL provides same headers)

**CMake Update:**
```cmake
# Old (will break)
find_package(CUB)

# New
include(FetchContent)
FetchContent_Declare(
  cccl
  GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
  GIT_TAG        2.6.0  # Check latest release
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cccl)
target_link_libraries(nova PRIVATE CCCL::cccl)
```

**Why:** Using archived CUB may work today but will become deprecated. CCCL is actively maintained and provides backward-compatible headers.

**Source:** [CCCL GitHub](https://github.com/nvidia/cccl)

#### Optional: cuCollections (Concurrent Data Structures)
**Status:** Header-only, requires NVCC 12.0+, C++17
**Confidence:** MEDIUM

GPU-accelerated concurrent data structures for advanced algorithms.

**Consider if building:**
- Hash table-based algorithms
- Concurrent set/map operations
- Lock-free data structure needs

**Not required for:**
- Basic sorting, scanning (CUB/CCCL handles)
- Graph algorithms (already implemented)
- Numerical methods (already implemented)

**CMake:**
```cmake
FetchContent_Declare(
  cuco
  GIT_REPOSITORY https://github.com/NVIDIA/cuCollections.git
  GIT_TAG        dev  # Active development
  OPTIONS        "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF"
)
FetchContent_MakeAvailable(cuco)
target_link_libraries(nova PRIVATE cuco)
```

**Requirements:** Volta or newer (sm_70+). **Does not support Pascal (sm_60)** — relevant if 6.0 architecture support is still needed.

**Source:** [cuCollections GitHub](https://github.com/NVIDIA/cuCollections)

---

## 3. What's NOT Needed

| Library | Why Avoid | Alternative |
|---------|-----------|-------------|
| Valgrind | Does not support CUDA | Compute Sanitizer |
| Intel VTune | x86-focused | Nsight Compute/Systems |
| Allinea/ARM DDT | Commercial, less CUDA-native | Nsight tools (free) |
| Extra property-based testing frameworks | Custom framework sufficient | Keep existing |
| Commercial profiling tools | NVIDIA tools are free and comprehensive | Nsight suite |

---

## 4. Complete Stack Additions for v2.7

### Required Migration

| Component | Current | Target | Rationale |
|-----------|---------|--------|-----------|
| CUB | Direct include | CCCL 2.6.0 | CUB archived, CCCL is maintained |

### Recommended Additions

| Component | Version | Integration | Purpose |
|-----------|---------|-------------|---------|
| Compute Sanitizer | CUDA 12+ built-in | CLI wrapper | Memory safety, race detection |
| Nsight Compute CLI | CUDA 12+ built-in | CMake detection | Kernel profiling |
| Nsight Systems | v2026.2 | Optional download | Timeline visualization |

### Optional Additions

| Component | Version | When Needed |
|-----------|---------|-------------|
| Tracy Profiler | v0.13.1 | If Nsight licensing/access is problematic |
| cuCollections | dev branch | If concurrent hash tables needed |

---

## 5. Version Compatibility

| Tool | Min CUDA | Min NVCC | Notes |
|------|----------|----------|-------|
| Compute Sanitizer | 11.0+ | — | Full feature set in CUDA 12+ |
| Nsight Compute | 11.0+ | — | v13.2 requires CUDA 12.0+ |
| Nsight Systems | 11.0+ | — | Supports all target archs |
| CCCL | 11.0+ | 11.0+ | Backward compatible |
| cuCollections | 12.0+ | 12.0+ | Dropped CUDA 11 support Feb 2026 |
| Tracy | 11.0+ | — | CUDA support varies by version |

**Note on sm_60 (Pascal):** cuCollections does not support Pascal. Compute Sanitizer and Nsight tools support Pascal. If maintaining sm_60 support, do not add cuCollections.

---

## 6. Installation Summary

### Minimal Additions (Required)

```bash
# No installation needed — all NVIDIA tools are part of CUDA Toolkit
# Just need to migrate CUB → CCCL in CMake

# Install Nsight Systems (optional, for GUI timeline analysis)
# Download from: https://developer.nvidia.com/nsight-systems/get-started
```

### CMake Changes for v2.7

```cmake
# 1. Replace CUB with CCCL
FetchContent_Declare(
  cccl
  GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
  GIT_TAG        2.6.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cccl)

# 2. Link CCCL instead of CUB
target_link_libraries(nova PUBLIC CCCL::cccl)

# 3. Add optional sanitizer support
option(NOVA_ENABLE_SANITIZER "Build with sanitizer support" OFF)
if(NOVA_ENABLE_SANITIZER)
    set_target_properties(nova PROPERTIES
        CXX_SANITIZERS "address,thread"
    )
endif()

# 4. Nsight Compute CLI detection
find_program(NCU_PATH ncu PATHS ENV PATH)
if(NCU_PATH)
    message(STATUS "Nsight Compute found: ${NCU_PATH}")
endif()
```

---

## 7. Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| CUB → CCCL migration | HIGH | Official migration path, well documented |
| Compute Sanitizer | HIGH | Part of CUDA toolkit, no version concerns |
| Nsight Compute/Systems | HIGH | Version 13.2/2026.2 verified current |
| cuCollections | MEDIUM | Active development, API may change |
| Tracy | MEDIUM | Good alternative if Nsight unavailable |

---

## Sources

- [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/)
- [Nsight Compute v13.2](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems v2026.2](https://docs.nvidia.com/nsight-systems/)
- [CCCL GitHub](https://github.com/nvidia/cccl)
- [cuCollections GitHub](https://github.com/NVIDIA/cuCollections)
- [Tracy Profiler v0.13.1](https://github.com/wolfpld/tracy)
- [Google Benchmark v1.9.5](https://github.com/google/benchmark)

---

*Stack research: 2026-04-30 for v2.7 Comprehensive Testing & Validation*
