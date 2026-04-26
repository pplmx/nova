# Project Research Summary

**Project:** Nova v1.8 Developer Experience
**Domain:** CUDA C++ Library Developer Experience Tooling
**Researched:** 2026-04-26
**Confidence:** HIGH

## Executive Summary

Nova v1.8 adds four developer experience layers on top of an already-solid CUDA/C++23/CMake 4.0+ foundation. The core additions are: (1) a descriptive CUDA error wrapper with device context and recovery hints, (2) proper CMake install/export infrastructure using config-file packages, (3) clangd and VS Code configuration for IDE support, and (4) ccache + Ninja integration for build performance. No framework changes are needed—only tooling additions and CMake infrastructure wiring.

The recommended approach prioritizes build correctness over speed: error handling first (Phase 1), then CMake packaging (Phase 2), IDE configuration (Phase 3), and build performance (Phase 4). Each phase builds on the previous without dependencies on future phases. Key risks are unity build correctness (can cause silent data corruption), ccache cache key divergence (architecture flags produce near-zero hit rates), and non-relocatable CMake packages (hardcoded absolute paths). All risks have documented prevention strategies in PITFALLS.md.

## Key Findings

### Recommended Stack

**Summary from STACK.md:** The project already has CMake 4.0+, CUDA 20, and C++23. Four new tooling layers need addition, all with well-documented integration patterns.

**Core technologies (no change needed):**
- CMake 4.0+ — Already in use; CMake 4.3.2 available
- CUDA Toolkit 20 — FindCUDAToolkit provides all `CUDA::*` imported targets
- C++23/CUDA 20 — Already set via `CMAKE_CXX_STANDARD` and `CMAKE_CUDA_STANDARD`
- `CMAKE_EXPORT_COMPILE_COMMANDS=ON` — Already enabled (line 53 of CMakeLists.txt)
- Ninja generator — Already documented in CMakeLists.txt

**New additions:**
- Custom `nova_error.hpp` — Header-only error wrapper with `std::error_code` categories, device context, and recovery hints. No external library needed.
- CMake config-file packages — Use `write_basic_package_version_file()` + `install(EXPORT)` via CMakePackageConfigHelpers (CMake 3.15+)
- `.clangd` configuration — YAML file telling clangd to treat `.cu` files as CUDA with `--cuda-gpu-arch=sm_80` and `-xcuda` flags
- CMakePresets.json — Shareable build configurations (dev, release, ci variants) with ccache and unity build options
- ccache — Set via `CMAKE_CUDA_COMPILER_LAUNCHER=ccache` in presets; version 4.10 recommended
- Ninja — Use `-G Ninja` generator for parallel builds (already recommended in CMakeLists.txt)

### Expected Features

**Summary from FEATURES.md:** Table stakes are CMake integration, compile_commands.json, and rich error messages. Differentiators are IDE zero-config and CMake presets.

**Must have (table stakes):**
- CMake integration with `find_package(nova)` support — Required for library adoption; export targets to `lib/cmake/nova/`
- `compile_commands.json` generation — Required for any IDE to function
- Rich CUDA error messages with file:line context and device info — Immediately useful debugging
- ccache detection and integration — Quick win for build time
- `.clangd` configuration file — Zero-config clangd support

**Should have (competitive differentiators):**
- Error recovery suggestions — "Try reducing block size" based on error type (P2)
- CMake presets (`dev`, `release`, `ci`) — Sensible defaults for common workflows (P2)
- Kernel-level error attribution — "Reduce kernel failed at thread (42,0)" (P2, HIGH complexity)

**Defer to v2+:**
- Interactive debugger helpers — Advanced users use ncu/nsight directly
- CI build matrix — Valuable but not blocking; depends on resource availability
- pkg-config support — Only needed for non-CMake build systems
- Python/Rust bindings — Separate repo scope

### Architecture Approach

**Summary from ARCHITECTURE.md:** The DX improvements integrate as a cross-cutting layer above the five-layer CUDA stack (memory → device → algo → api). These are infrastructure additions, not feature additions, so they touch the build system, library layer boundaries, and developer tooling.

**Major components:**
1. **Error Layer (`include/cuda/error/`)** — Structured `std::error_code` with CUDA error categories, device context capture, and recovery hints. RAII `cuda_error_guard` pattern.
2. **CMake Package Exports** — Modern CMake `export()` + `install(EXPORT)` generating `novaTargets.cmake` and `novaConfig.cmake` for relocatable `find_package()` support.
3. **IDE Configuration (`.clangd`, `.vscode/`)** — YAML and JSON configs that tell clangd how to parse CUDA files and configure VS Code with CUDA IntelliSense.
4. **Build Performance Layer** — ccache + Ninja + unity builds, controlled via CMakePresets.json with environment-aware settings.

### Critical Pitfalls

1. **Generic Error Messages Mask the Root Cause** — `cudaGetErrorString()` returns the same text NVIDIA's driver returns. Prevention: Extend `CUDA_CHECK` with operation context, add error-category-specific recovery hints, map cuBLAS status codes to readable names.

2. **CMake Exports Produce Non-Relocatable Packages** — Hard-coded `${CMAKE_SOURCE_DIR}` paths break when the package is installed to a different location. Prevention: Use `$<BUILD_INTERFACE:>` and `$<INSTALL_INTERFACE:>` generator expressions in all `target_include_directories` calls.

3. **clangd Reports Thousands of Errors for CUDA Files** — clangd uses libclang which cannot parse nvcc-specific flags. Prevention: Create `.clangd/config.yaml` with `CompileFlags:` mapping that filters nvcc flags and adds `-xcuda`, `--cuda-gpu-arch=sm_80`.

4. **ccache Has Near-Zero Cache Hit Rate for CUDA** — Architecture flags (`-gencode`) and `--threads` in global `CMAKE_CUDA_FLAGS` produce different cache keys per compilation. Prevention: Remove `--threads` from global flags, set `CCACHE_BASEDIR`, configure `sloppiness = include_file_mtime`.

5. **Unity Builds Produce Silent Data Corruption** — Symbol collisions and macro collisions in concatenated `.cu` files cause non-deterministic behavior. Prevention: Audit for non-namespaced device symbols, start with `UNITY_BUILD_BATCH_SIZE=4`, run full test suite with unity builds enabled.

## Implications for Roadmap

Based on research, the recommended phase structure follows build order from ARCHITECTURE.md and maps to pitfall prevention from PITFALLS.md:

### Phase 1: Error Message Framework

**Rationale:** Error handling is foundational—all other DX features benefit from users being able to understand failures. The existing `CUDA_CHECK` macro is minimal and must be extended before CMake packaging (which may surface new error paths).

**Delivers:**
- `include/cuda/error/cuda_error.hpp` — Structured error types with categories
- `cuda_error_guard` RAII wrapper — Translates `cudaError_t` → `nova::error`
- Recovery hint mapping per error type
- cuBLAS status code → readable name mapping
- Tests for error translation

**Addresses:** Rich error messages with context (FEATURES.md P1)

**Avoids:** Pitfall 1 — Generic error messages mask root cause

### Phase 2: CMake Package Export

**Rationale:** CMake packaging is needed for library adoption. Must be validated with real downstream projects. Depends on error layer (Phase 1) for consistent error translation in exported targets.

**Delivers:**
- `cmake/novaConfig.cmake.in` — Config-file template
- `install(EXPORT novaTargets ...)` rules in CMakeLists.txt
- Generator expressions for relocatable paths
- Feature matrix in CMake configure output (NCCL, MPI status)
- Documentation: `find_package(nova)` usage

**Addresses:** CMake integration with exported targets (FEATURES.md P1)

**Avoids:** Pitfall 2 — Non-relocatable CMake packages; Pitfall 6 — Silent dependency failures

### Phase 3: IDE Configuration

**Rationale:** IDE support requires `compile_commands.json` (already generated) and clangd configuration. Once `.clangd` exists, VS Code settings can reference the same flags. Phase 4 (build performance) can use the same IDE infrastructure.

**Delivers:**
- `.clangd/config.yaml` — clangd configuration with CUDA flags
- `.vscode/settings.json` — VS Code clangd integration
- `.vscode/c_cpp_properties.json` — CUDA IntelliSense paths
- `.vscode/extensions.json` — Recommended extensions
- `compile_commands.json` symlink to project root
- `docs/ide-setup.md` — Developer onboarding guide

**Addresses:** `.clangd` configuration (FEATURES.md P1), compile_commands.json generation (FEATURES.md P1)

**Avoids:** Pitfall 3 — clangd fails on CUDA files; Pitfall 7 — IDE config not version-controlled

### Phase 4: Build Performance

**Rationale:** ccache and unity builds are the highest-impact performance optimizations, but must be validated with full test suite before shipping. CI configuration depends on developer presets being stable.

**Delivers:**
- `CMakePresets.json` — `dev`, `release`, `ci` presets with ccache and unity build
- `NOVA_USE_CCACHE` CMake option — ccache detection and configuration
- `CMAKE_UNITY_BUILD` wired to `NOVA_ENABLE_UNITY_BUILD`
- Unity build batch size adaptive to CPU/memory
- CI-specific presets with cache limits and no ccache
- Full test suite validation with `NOVA_ENABLE_UNITY_BUILD=ON`

**Addresses:** ccache integration (FEATURES.md P1), CMake presets (FEATURES.md P2)

**Avoids:** Pitfall 4 — ccache zero cache hits; Pitfall 5 — Unity build correctness failures; Pitfall 8 — CI build performance unchanged

### Phase Ordering Rationale

- **Phase 1 before 2:** Error handling is foundational; CMake exports may surface new error paths that need proper translation
- **Phase 2 before 3:** IDE include paths come from CMake; once CMake exports are correct, IDE configs can reference them
- **Phase 3 before 4:** IDE configuration is independent of build speed; validate correctness first, then optimize
- **Phase 4 last:** Build performance is additive—it enhances all previous phases but doesn't change their correctness guarantees

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (Build Performance):** Unity build + CUDA interaction has edge cases; may need `/gsd-research-phase` for NCCL symbol collision patterns
- **Phase 1 (Error Framework):** cuBLAS error code mapping may need NVIDIA documentation validation for all status codes

Phases with standard patterns (skip research-phase):
- **Phase 2 (CMake Export):** Modern CMake `export()` + `install(EXPORT)` is well-documented with official CMake docs
- **Phase 3 (IDE Configuration):** `.clangd` YAML configuration is standard with clangd official docs

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All technology versions verified against official CMake 4.3.2 and clangd documentation; ccache native CMake module confirmed |
| Features | HIGH | Feature landscape verified against CMake integration patterns and competitor analysis (cuBLAS, Thrust) |
| Architecture | MEDIUM | Phase ordering and component responsibilities are well-reasoned from research but represent inference from patterns rather than a single authoritative source |
| Pitfalls | HIGH | 8 pitfalls documented with specific warning signs, prevention strategies, and recovery steps; all verified against CMake and clangd documentation |

**Overall confidence:** HIGH

### Gaps to Address

- **cuBLAS status code exhaustiveness:** Research identified common cuBLAS errors but did not enumerate all status codes. Validate coverage during Phase 1 implementation.
- **NCCL symbol collision patterns:** Unity builds + NCCL interaction may have specific patterns not documented. Test with `NOVA_ENABLE_UNITY_BUILD=ON` in Phase 4.
- **clangd CUDA parsing coverage:** clangd CUDA support has known limitations. May need to document unsupported patterns or fallback to nvim-lspconfig for some files.

## Sources

### Primary (HIGH confidence)
- [CMake cmake-packages(7)](https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html) — Config-file package layout and `install(EXPORT)`
- [CMake FindCUDAToolkit](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html) — All `CUDA::*` imported targets
- [CMake cmake-presets(7)](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) — Version 10 schema, environment, vendor fields
- [clangd Installation and Configuration](https://clangd.llvm.org/installation.html) — .clangd YAML configuration reference
- [CCache CMake integration](https://ccache.dev/manual/latest.html#_cmake) — Compiler cache configuration

### Secondary (MEDIUM confidence)
- [ARCHITECTURE.md Phase Dependencies](./ARCHITECTURE.md) — Build order rationale from architectural patterns
- [PITFALLS.md Recovery Strategies](./PITFALLS.md) — Recovery cost estimates from integration experience

### Research Files
- [STACK.md](./STACK.md) — Complete tooling inventory and integration patterns
- [FEATURES.md](./FEATURES.md) — Feature prioritization matrix and MVP definition
- [ARCHITECTURE.md](./ARCHITECTURE.md) — Component responsibilities and data flows
- [PITFALLS.md](./PITFALLS.md) — 8 critical pitfalls with prevention and recovery

---
*Research completed: 2026-04-26*
*Ready for roadmap: yes*
