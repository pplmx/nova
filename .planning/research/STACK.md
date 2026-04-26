# Stack Research: Nova v1.8 Developer Experience

**Domain:** CUDA C++ library developer experience tooling
**Researched:** 2026-04-26
**Confidence:** HIGH — CMake 4.3.2 docs verified, clangd official docs verified, ccache native CMake module confirmed

## Executive Summary

Nova v1.8 adds four DX layers on top of the existing CUDA/C++23/CMake 4.0+ stack. No framework changes are needed — only tooling additions. The core technology additions are: (1) a descriptive CUDA error wrapper header, (2) proper CMake install/export infrastructure using config-file packages, (3) CMakePresets.json and .clangd configuration for IDE support, and (4) ccache + Ninja integration for build performance. The project already exports `compile_commands.json` and has a unity build option — these need to be wired up properly rather than replaced.

## Recommended Stack

### Core Technologies (Already Present — No Change Needed)

| Technology | Version | Status | Notes |
|------------|---------|--------|-------|
| CMake | 4.0+ | Already in use | CMake 4.3.2 available in environment |
| CUDA Toolkit | 20 | Already in use | FindCUDAToolkit provides `CUDA::cudart`, `CUDA::cublas`, etc. |
| C++ standard | C++23 | Already in use | CMAKE_CXX_STANDARD 23 |
| CUDA standard | CUDA 20 | Already in use | CMAKE_CUDA_STANDARD 20 |
| CMAKE_EXPORT_COMPILE_COMMANDS | ON | Already enabled | Line 53 of CMakeLists.txt |
| Ninja generator | latest | Already supported | Documented in CMakeLists.txt header |

### New Additions: Error Messages

| Technology | Purpose | Why |
|------------|---------|-----|
| Custom `nova_error.hpp` | Descriptive CUDA errors with device context, API call, and recovery hints | `cudaGetErrorString()` is generic; Nova needs actionable guidance per error category |

**No external library needed.** The error wrapper is a single header file that:
- Maps `cudaError_t` codes to human-readable categories (memory, synchronization, launch, API usage)
- Appends device ordinal, stream, and memory usage at error site
- Provides recovery-action strings per error type (retry, reduce grid size, check memory, etc.)

### New Additions: CMake Integration

| Technology | Version | Purpose | Integration |
|------------|---------|---------|-------------|
| CMakePackageConfigHelpers | Built-in (CMake 3.15+) | Generate `*Config.cmake` and `*ConfigVersion.cmake` during install | Use `write_basic_package_version_file()` + `install(EXPORT)` |
| NovaTargets.cmake | Generated | IMPORTED targets for downstream `find_package(Nova)` | `install(EXPORT NovaTargets NAMESPACE Nova:: DESTINATION lib/cmake/Nova)` |
| CMakePresets.json | Schema version 10 | Shareable, versioned build configurations for developers and CI | Place in project root, version-controlled |
| CMakeUserPresets.json | Same schema | Developer-local build overrides (ignored by git) | Created by developers, .gitignore'd |

**Why config-file packages over find-modules:** CMake 4.0 ships with `FindCUDAToolkit` providing all `CUDA::*` imported targets natively. A config-file package exposes Nova's own library targets the same way, with proper `INTERFACE_INCLUDE_DIRECTORIES` propagation. This is the modern standard (used by fmt, spdlog, Eigen, GoogleBenchmark).

### New Additions: IDE Support

| Technology | Purpose | Integration |
|------------|---------|-------------|
| `.clangd` config file | Language server configuration for clangd | YAML file in project root, configures compile flags, includes, clang-tidy |
| `compile_commands.json` symlink | Make compilation database discoverable by clangd | Symlink `build/compile_commands.json` → project root (clangd searches parent dirs) |
| `.vscode/` settings | VS Code clangd integration | `settings.json` disabling built-in C++ extension, enabling clangd |
| CMakePresets.json `vendor` field | IDE-specific metadata | VS Code clangd extension reads `"vendor": {"llvm-vs-code-extensions.vscode-clangd": {...}}` |

**clangd CUDA support:** clangd does not natively understand CUDA `.cu` files as GPU code. Configuration via `.clangd` `CompileFlags` with `Add: ["-x", "cuda", "-std=c++23", "--cuda-gpu-arch=sm_80"]` tells clangd to treat files as CUDA and apply correct target architecture. This is the standard approach — no fork or patched clangd required.

**Fallback flags approach:** For headers and files clangd cannot fully parse as CUDA, use `CompileFlags: FallbackFlags: ["-std=c++23"]` so partial completions still work. See [clangd installation docs](https://clangd.llvm.org/installation).

### New Additions: Build Performance

| Technology | Purpose | Integration |
|------------|---------|-------------|
| ccache | Compiler cache for incremental builds | CMakeFindPackage support via `find_package(CCache)` or environment variable `CMAKE_C_COMPILER_LAUNCHER=ccache` |
| Ninja generator | Parallel build execution | `cmake -G Ninja -B build` (already documented in CMakeLists.txt) |
| CMAKE_UNITY_BUILD | Unity/Jumbo builds | Already partially implemented; needs `CMAKE_UNITY_BUILD ON` global variable + preset control |
| CMAKE_UNITY_BUILD_BATCH_SIZE | Files per unity source | Already has adaptive logic (32 for 32+ cores, 64 for 64+ cores) |

**ccache integration:**
```cmake
# Via CMake module (CMake 3.4+)
find_package(CCache)
if(CCache_FOUND)
    ccache_add_project()
endif()

# Or via environment / preset (no CMake change needed)
# cmake --preset dev  → CMakePresets.json sets CMAKE_CUDA_COMPILER_LAUNCHER:FILEPATH=ccache
```

**ccache version:** Latest stable is 4.10 (released 2025). Verify via `ccache --version`. Use `apt install ccache` or `brew install ccache`.

## Installation

```bash
# Developer tooling (Ubuntu/Debian)
sudo apt install ccache clangd ninja-build

# Verify versions
ccache --version    # Should be 4.10+
clangd --version    # Should be 18+ for best CUDA support
cmake --version     # 4.0+ (already satisfied)
nvcc --version      # CUDA 20 (already satisfied)

# macOS
brew install ccache llvm ninja

# After building with ccache-enabled preset
cmake --preset dev  # CMAKE_CUDA_COMPILER_LAUNCHER=ccache set by preset
```

## Alternatives Considered

| Category | Recommended | Alternative | When to Use Alternative |
|----------|-------------|-------------|-------------------------|
| Package type | Config-file packages | Find-modules | Only if Nova is not CMake-native (it is CMake-native, so config-file is correct) |
| Package discovery | CMAKE_PREFIX_PATH | User Package Registry | Package Registry is IDE-specific; CMAKE_PREFIX_PATH works everywhere |
| clangd integration | .clangd YAML | Bear + compile_flags.txt | Bear is build-system-specific; .clangd works for all clangd clients |
| Unity build control | CMake preset toggle | Hard-coded in CMakeLists.txt | Preset control lets developers opt-in/out without editing CMakeLists.txt |
| Build generator | Ninja (default preset) | Makefiles | Makefiles work but Ninja is faster for CUDA compilation parallelism |
| ccache invocation | CMake preset CMAKE_CUDA_COMPILER_LAUNCHER | find_package(CCache) + ccache_add_project | Both work; preset is more visible and portable |
| Error format | Custom header-only wrapper | CUDA error string only | Wrapper adds device context and recovery hints — actionable beyond raw error |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Find-modules for Nova itself | Outdated pattern for CMake-native projects; does not expose imported targets properly | `install(EXPORT)` + Config.cmake files |
| Bear for compile_commands | Requires running full clean build to generate; fragile for incremental use | CMake's built-in `CMAKE_EXPORT_COMPILE_COMMANDS=ON` (already enabled) + symlink |
| Clang-based nvcc wrapper (icx) | NVIDIA recommends nvcc for CUDA; clang as CUDA frontend is separate project | Keep nvcc, use clangd only for host-code analysis with CUDA flags |
| Global `CMAKE_UNITY_BUILD` ON in CMakeLists.txt | Can cause ODR issues with templates and anonymous namespaces in CUDA code | Make it a preset-controlled option (developers choose) or set per-target |
| Pre-built clangd CUDA patches | Fragmented, unmaintained forks; breaks with LLVM upgrades | Use upstream clangd 18+ with `.clangd` `CompileFlags` configuration |
| bear for compile_commands.json | Requires clean rebuild; not compatible with CMake configure step | `CMAKE_EXPORT_COMPILE_COMMANDS=ON` already generates it at configure time |

## Stack Patterns by Variant

**If developer uses `cmake --preset dev`:**
- Generator: Ninja
- ccache enabled via `CMAKE_CUDA_COMPILER_LAUNCHER`
- Unity build ON (fast iterative development)
- clangd index built via background-index

**If developer uses `cmake --preset release`:**
- Generator: Ninja Multi-Config
- ccache disabled (clean build, no cache pollution)
- Unity build ON
- Full opt+vectorize flags

**If CI runs `cmake --preset ci`:**
- Generator: Ninja
- No ccache (clean environment)
- Unity build ON
- Strict warnings as errors
- clang-tidy enabled

**If developer uses `cmake -G "Unix Makefiles"`:**
- ccache via `CC`/`CXX` wrappers in PATH
- Unity build optional (controlled by cache variable)
- No CMakePresets.json dependency

## Version Compatibility

| Component | Version | Compatible With | Notes |
|-----------|---------|-----------------|-------|
| CMake | 4.0+ | CUDA 20, C++23, Ninja | CMake 4.3.2 is latest; project already requires 4.0 |
| FindCUDAToolkit | CMake 3.17+ | CUDA 11+ | Provides all `CUDA::*` imported targets; no manual FindCUDA.cmake needed |
| CMAKE_UNITY_BUILD | CMake 3.16+ | C++, CUDA (added CMake 3.31) | Project already has CMake 4.0, so CUDA unity builds fully supported |
| CMakePackageConfigHelpers | CMake 3.15+ | All modern packages | Used for Nova's own package export |
| CMakePresets.json schema | Version 10 | CMake 3.23+ | Project already has CMake 4.0, schema v10 fully supported |
| clangd | 15+ | C++23, CUDA via flags | Use latest (18+) for best C++23 support |
| ccache | 4.0+ | All compilers including nvcc | ccache 4.10 recommended; verify with `ccache --version` |
| Ninja | 1.10+ | CMake all versions | Already recommended in CMakeLists.txt |

## Integration Points

### Existing CMake Points (Wire Up, Don't Replace)

| Existing Feature | Current State | What to Add |
|-----------------|---------------|-------------|
| `CMAKE_EXPORT_COMPILE_COMMANDS ON` | Line 53, always ON | Add symlink in CMakeLists.txt post-build or as a custom target |
| `NOVA_ENABLE_UNITY_BUILD` option | Line 75, default ON | Wire to `CMAKE_UNITY_BUILD ${NOVA_ENABLE_UNITY_BUILD}` |
| ProcessorCount + `--threads` | Lines 84-93, nvcc flags | Already good; add ccache launcher alongside |
| FindCUDAToolkit | Line 59, REQUIRED | Already optimal; use `CUDA::*` targets in exported package |
| Google Test FetchContent | Lines 481-487 | Already good; keep as-is |

### New CMake Targets to Add

```cmake
# In CMakeLists.txt additions:
# 1. Wire unity build option to CMAKE_UNITY_BUILD
set(CMAKE_UNITY_BUILD ${NOVA_ENABLE_UNITY_BUILD})

# 2. Export targets for find_package(Nova)
include(CMakePackageConfigHelpers)
export(EXPORT NovaTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/Nova/NovaTargets.cmake"
  NAMESPACE Nova::
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/Nova/NovaConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
configure_file(cmake/NovaConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/Nova/NovaConfig.cmake"
  COPYONLY
)
install(EXPORT NovaTargets
  FILE NovaTargets.cmake
  NAMESPACE Nova::
  DESTINATION lib/cmake/Nova
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/Nova/NovaConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/Nova/NovaConfigVersion.cmake"
  DESTINATION lib/cmake/Nova
)
```

### New Files to Create

| File | Purpose |
|------|---------|
| `cmake/NovaConfig.cmake.in` | Config-file template for `find_package(Nova)` |
| `.clangd` | clangd configuration (CUDA flags, include paths) |
| `CMakePresets.json` | Developer build presets (dev, release, ci) |
| `CMakeUserPresets.json` | Developer-local overrides (gitignored) |
| `.vscode/settings.json` | VS Code clangd integration |
| `include/nova/error.hpp` | Descriptive CUDA error wrapper |
| `src/nova/error.cpp` | Error detail implementation |

## Sources

- [CMake cmake-packages(7) — Config-file package layout and `install(EXPORT)`](https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html) — HIGH
- [CMake CMAKE_EXPORT_COMPILE_COMMANDS variable](https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html) — HIGH
- [CMake UNITY_BUILD property (CUDA support added CMake 3.31)](https://cmake.org/cmake/help/latest/prop_tgt/UNITY_BUILD.html) — HIGH
- [CMake FindCUDAToolkit — all CUDA::* imported targets](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html) — HIGH
- [CMake cmake-presets(7) — version 10 schema, environment, vendor fields](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) — HIGH
- [clangd Installation and Project Setup](https://clangd.llvm.org/installation.html) — HIGH
- [clangd Configuration Reference](https://clangd.llvm.org/config.html) — HIGH
- [CCache CMake integration](https://ccache.dev/manual/latest.html#_cmake) — MEDIUM (verified via CMake modules, FindCCache moved/deprecated)

---
*Stack research for: Nova v1.8 Developer Experience*
*Researched: 2026-04-26*
