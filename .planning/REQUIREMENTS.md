# Nova v1.8 Developer Experience — Requirements

**Milestone:** v1.8 Developer Experience
**Research:** Complete (STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md)
**Status:** Draft

## Overview

This milestone adds four developer experience layers to Nova:

1. **Error Message Framework** — Descriptive CUDA errors with device context and recovery hints
2. **CMake Package Export** — Relocatable `find_package(nova)` support via config-file packages
3. **IDE Configuration** — clangd and VS Code settings for CUDA support
4. **Build Performance** — ccache, Ninja, and CMakePresets.json for fast builds

## Requirements

### Phase 1: Error Message Framework

- [ ] **ERR-01**: Developer can receive descriptive error messages that include the CUDA function name, file:line location, and device context
- [ ] **ERR-02**: Developer receives error-category-specific recovery hints (e.g., "Try reducing block size" for shared memory errors)
- [ ] **ERR-03**: cuBLAS status codes are translated to human-readable names (e.g., `CUBLAS_STATUS_NOT_SUPPORTED` instead of `7`)
- [ ] **ERR-04**: Error types integrate with `std::error_code` for idiomatic C++ error handling

### Phase 2: CMake Package Export

- [ ] **CMK-01**: Library can be found via `find_package(nova REQUIRED)` after installation
- [ ] **CMK-02**: Exported targets (`Nova::nova`, `Nova::cuda`) are relocatable using generator expressions
- [ ] **CMK-03**: Feature matrix is displayed during CMake configure (NCCL, MPI status)
- [ ] **CMK-04**: Version file matches installed version via `cmake_package_registry`

### Phase 3: IDE Configuration

- [ ] **IDE-01**: `.clangd/config.yaml` enables clangd to parse `.cu` files with correct CUDA flags
- [ ] **IDE-02**: `.vscode/settings.json` integrates clangd for VS Code users
- [ ] **IDE-03**: `compile_commands.json` symlink exists at project root for IDE discovery
- [ ] **IDE-04**: Documentation exists for IDE setup in `docs/ide-setup.md`

### Phase 4: Build Performance

- [ ] **BLD-01**: CMakePresets.json provides `dev`, `release`, and `ci` presets with sensible defaults
- [ ] **BLD-02**: `NOVA_USE_CCACHE` CMake option enables ccache with correct configuration
- [ ] **BLD-03**: `NOVA_ENABLE_UNITY_BUILD` CMake option enables unity builds validated against full test suite
- [ ] **BLD-04**: Build performance documentation exists with ccache and preset usage instructions

## Future Requirements (Deferred)

- Interactive debugger helpers with ncu/nsight integration
- Python/Rust bindings
- pkg-config support
- CI build matrix with multiple CUDA versions

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time error monitoring | Advanced observability, not core DX |
| Interactive debugger helpers | Advanced users use ncu/nsight directly |
| Python/Rust bindings | Separate repository scope |
| pkg-config support | Only needed for non-CMake build systems |
| Cross-vendor support (HIP) | Nova is CUDA-specific by design |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ERR-01 | Phase 33 | Pending |
| ERR-02 | Phase 33 | Pending |
| ERR-03 | Phase 33 | Pending |
| ERR-04 | Phase 33 | Pending |
| CMK-01 | Phase 34 | Pending |
| CMK-02 | Phase 34 | Pending |
| CMK-03 | Phase 34 | Pending |
| CMK-04 | Phase 34 | Pending |
| IDE-01 | Phase 35 | Pending |
| IDE-02 | Phase 35 | Pending |
| IDE-03 | Phase 35 | Pending |
| IDE-04 | Phase 35 | Pending |
| BLD-01 | Phase 36 | Pending |
| BLD-02 | Phase 36 | Pending |
| BLD-03 | Phase 36 | Pending |
| BLD-04 | Phase 36 | Pending |

---

*Requirements defined: 2026-04-26*
*Total: 16 requirements (ERR-01..ERR-04, CMK-01..CMK-04, IDE-01..IDE-04, BLD-01..BLD-04)*
