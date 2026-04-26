# Nova v1.8 Developer Experience — Roadmap

**Milestone:** v1.8 Developer Experience
**Created:** 2026-04-26
**Status:** Planning
**Granularity:** Standard (4 phases)

## Overview

This milestone adds four developer experience layers to Nova:

1. **Error Message Framework** — Descriptive CUDA errors with device context and recovery hints
2. **CMake Package Export** — Relocatable `find_package(nova)` support via config-file packages
3. **IDE Configuration** — clangd and VS Code settings for CUDA support
4. **Build Performance** — ccache, Ninja, and CMakePresets.json for fast builds

## Phases

- [ ] **Phase 33: Error Message Framework** — Descriptive CUDA errors with device context and recovery hints
- [ ] **Phase 34: CMake Package Export** — Relocatable find_package support with exported targets
- [ ] **Phase 35: IDE Configuration** — clangd and VS Code settings for CUDA support
- [ ] **Phase 36: Build Performance** — ccache, unity builds, and CMakePresets.json

---

## Phase Details

### Phase 33: Error Message Framework

**Goal:** Developers can understand and recover from CUDA errors quickly

**Depends on:** Nothing (first phase)

**Requirements:** ERR-01, ERR-02, ERR-03, ERR-04

**Success Criteria** (what must be TRUE):

1. Developer sees CUDA error message that includes the CUDA function name, file:line location, and device context (e.g., "Reduce kernel failed at device 0, stream 1, file:line /path/to/kernel.cu:42")
2. Developer sees error-category-specific recovery hints in error output (e.g., "Try reducing block size" for shared memory errors, "Check memory allocation" for OOM errors)
3. Developer sees cuBLAS status codes translated to human-readable names (e.g., `CUBLAS_STATUS_NOT_SUPPORTED` instead of numeric code `7`)
4. Developer can catch and handle Nova errors using `std::error_code` and `std::error_category` idioms in C++

**Plans:** TBD

### Phase 34: CMake Package Export

**Goal:** Downstream projects can discover and link Nova via CMake

**Depends on:** Phase 33 (error framework provides consistent errors in exported targets)

**Requirements:** CMK-01, CMK-02, CMK-03, CMK-04

**Success Criteria** (what must be TRUE):

1. Developer can run `find_package(nova REQUIRED)` successfully after running `cmake --install`
2. Developer can link `Nova::nova` and `Nova::cuda` imported targets in their CMake project
3. Developer sees feature matrix output during CMake configure showing NCCL, MPI status (e.g., "Nova 1.8.0: NCCL [ON], MPI [ON], CUDA 20")
4. Developer can relocate installed package to different directory without breaking `find_package` resolution

**Plans:** TBD

### Phase 35: IDE Configuration

**Goal:** Developers can use clangd or VS Code with full CUDA support

**Depends on:** Phase 34 (IDE include paths come from CMake exports)

**Requirements:** IDE-01, IDE-02, IDE-03, IDE-04

**Success Criteria** (what must be TRUE):

1. Developer can open Nova in any editor with clangd and see zero spurious errors for `.cu` files (correct CUDA parsing with `--cuda-gpu-arch` flags)
2. Developer using VS Code sees clangd integration working with real-time diagnostics and code completion for CUDA code
3. Developer finds `compile_commands.json` symlinked at project root without manual setup
4. Developer can follow `docs/ide-setup.md` to configure their IDE in under 5 minutes

**Plans:** TBD

### Phase 36: Build Performance

**Goal:** Developers can build Nova quickly using CMake presets and ccache

**Depends on:** Phase 35 (presets reference IDE flags from Phase 35)

**Requirements:** BLD-01, BLD-02, BLD-03, BLD-04

**Success Criteria** (what must be TRUE):

1. Developer can configure and build using `cmake --preset dev` / `cmake --build --preset dev` with sensible defaults (Ninja, parallel jobs)
2. Developer can enable ccache with `-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache` and see >50% cache hit rate on rebuilds
3. Developer can enable unity builds with `-DNOVA_ENABLE_UNITY_BUILD=ON` and all 444 tests pass
4. Developer can follow build performance documentation to set up ccache and understand preset usage

**Plans:** TBD

---

## Coverage

| Requirement | Phase | Description |
|-------------|-------|-------------|
| ERR-01 | Phase 33 | Descriptive error messages with CUDA function name, file:line, device context |
| ERR-02 | Phase 33 | Error-category-specific recovery hints |
| ERR-03 | Phase 33 | cuBLAS status code translation to readable names |
| ERR-04 | Phase 33 | std::error_code integration for idiomatic error handling |
| CMK-01 | Phase 34 | find_package(nova REQUIRED) support |
| CMK-02 | Phase 34 | Relocatable exported targets with generator expressions |
| CMK-03 | Phase 34 | Feature matrix display during CMake configure |
| CMK-04 | Phase 34 | Version file matching installed version |
| IDE-01 | Phase 35 | .clangd/config.yaml for clangd CUDA parsing |
| IDE-02 | Phase 35 | .vscode/settings.json for VS Code clangd integration |
| IDE-03 | Phase 35 | compile_commands.json symlink at project root |
| IDE-04 | Phase 35 | docs/ide-setup.md documentation |
| BLD-01 | Phase 36 | CMakePresets.json with dev/release/ci presets |
| BLD-02 | Phase 36 | NOVA_USE_CCACHE CMake option for ccache |
| BLD-03 | Phase 36 | NOVA_ENABLE_UNITY_BUILD CMake option |
| BLD-04 | Phase 36 | Build performance documentation |

**Coverage:** 16/16 requirements mapped

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 33. Error Message Framework | 0/4 | Not started | — |
| 34. CMake Package Export | 0/4 | Not started | — |
| 35. IDE Configuration | 0/4 | Not started | — |
| 36. Build Performance | 0/4 | Not started | — |

---

## Previous Milestone

**v1.7 Benchmarking & Testing** — SHIPPED 2026-04-26

- Phase 29: Benchmark Infrastructure Foundation
- Phase 30: Comprehensive Benchmark Suite
- Phase 31: CI Regression Testing
- Phase 32: Performance Dashboards

---

*Roadmap created: 2026-04-26*
*Next: /gsd-plan-phase 33*
