# Feature Research

**Domain:** CUDA Library Developer Experience
**Researched:** 2026-04-26
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist in a production CUDA library. Missing these = product feels incomplete or unprofessional.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| CMake integration with `find_package()` support | Users expect standard CMake workflows: `find_package(nova)`, `target_link_libraries(myapp PRIVATE nova::nova)` | MEDIUM | Requires `novaTargets.cmake`, `novaConfig.cmake`, version file. Build should export targets to `lib/cmake/nova/` |
| `compile_commands.json` generation | Required for clangd, VS Code intellisense, and other tooling to work | LOW | Add `CMAKE_EXPORT_COMPILE_COMMANDS=ON` and set output path to project root |
| CUDA error messages with context | Raw `cudaError_t` codes are unhelpful ("cudaErrorIllegalAddress: an illegal memory access was encountered") | MEDIUM | Wrap with file:line, device info, kernel name if available, suggested fixes |
| ccache integration | Dramatically speeds rebuilds; users expect it to work out of box | LOW | Detect ccache, set `CMAKE_CUDA_COMPILER_LAUNCHER`, document in README |
| Parallel compilation support | Users expect `-j$(nproc)` to work effectively | LOW | Ensure CMake targets have no serialization bottlenecks; use unity builds if needed |
| pkg-config support | Some build systems prefer `.pc` files | LOW | Generate via CMake's `export()` with pkg_check_modules |

### Differentiators (Competitive Advantage)

Features that set the library apart. Not required, but valuable for adoption and retention.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Kernel-level error attribution | "Reduce kernel failed at thread (42,0) on device 1" instead of just "illegal address" | HIGH | Requires capturing kernel launch info, threading through async error paths |
| Interactive debugger helpers | `NOVA_DEBUG_KERNEL=1` env var to inject breakpoints, logging | MEDIUM | Conditional compilation, GPU debugging macros |
| CUDA error recovery suggestions | "Try reducing block size" or "Check memory alignment" based on error type | MEDIUM | Error-specific hints; maps common errors to actionable advice |
| IDE config presets (.clangd, compile_flags.txt) | "Works in VS Code immediately" - zero config for users | LOW | Ship `.clangd` with compilation flags, include paths |
| Build performance diagnostics | `nova::benchmark_build` target or cmake --preset with build timing | MEDIUM | Useful for users debugging their own CUDA build performance |
| CMake presets | `cmake --preset developer` with sensible defaults for common workflows | LOW | Include: dev, release, ndebug, unity builds |
| Header-only diagnostic layer | Optional `nova/diagnostics/errors.hpp` for users who want rich errors | MEDIUM | Opt-in via CMake flag `NOVA_ENABLE_RICH_ERRORS` |
| Continuous integration build matrix | `.github/workflows/build.yml` with multiple CUDA/GCC versions | MEDIUM | Validates compatibility; builds serve as working examples |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Automatic error recovery/resilience | "Just make it work even when CUDA fails" | Masks real bugs, creates silent data corruption, user loses awareness of issues | Explicit `try_recover()` API; let errors propagate by default |
| Language bindings (Python, Rust) in same repo | "I want to use it from Python" | Massive scope, ABI complexity, release cycle overhead | Separate `nova-python` repo with clear versioning |
| Runtime CUDA version detection/switching | "Support both CUDA 11 and 12" | Massive complexity, feature fragmentation, testing matrix explosion | Document minimum CUDA version (already 20); recommend upgrade path |
| GUI debugging tools | "Make it easier to debug" | Custom GUI dev is not library work; distracts from core value | Document ncu,Nsight usage; provide integration scripts |
| Cross-vendor support (AMD HIP) | "Also work on AMD GPUs" | Entirely different runtime, not a feature add, fundamental rearchitecture | Document as out-of-scope; suggest rocAL for AMD |

## Feature Dependencies

```
CMake Integration
    └──requires──> Version file generation
    └──requires──> Target export configuration
    └──requires──> FindModule compatibility

IDE Support (clangd/VS Code)
    └──requires──> compile_commands.json
    └──requires──> CMake integration (for include paths)

Better Error Messages
    └──requires──> CUDA error wrapper infrastructure
    └──requires──> Kernel launch context capture

Build Performance
    └──enhances──> CMake Integration (faster iteration)
    └──enhances──> IDE Support (faster reparse)
```

### Dependency Notes

- **CMake integration requires version file generation:** The `novaConfigVersion.cmake` file is needed for `find_package()` to work with version constraints
- **IDE support requires compile_commands.json:** Clangd and VS Code both rely on this JSON file for code intelligence
- **Better error messages require CUDA error wrapper:** A centralized error handling system that enriches `cudaError_t` with context
- **Build performance enhances everything:** Faster builds improve iteration speed for all DX features

## MVP Definition

### Launch With (v1.8)

Minimum viable developer experience — what's needed to validate the approach.

- [ ] CMake integration with exported targets and `find_package()` support — essential for library adoption
- [ ] `compile_commands.json` generation — required for any IDE to function
- [ ] Rich CUDA error messages with file:line context and device info — immediately useful debugging
- [ ] ccache detection and integration — quick win for build time
- [ ] `.clangd` configuration file — zero-config clangd support

### Add After Validation (v1.8.x)

Features to add once core is working.

- [ ] CMake presets for common workflows (dev, release) — trigger: user feedback about build configuration
- [ ] Error recovery suggestions — trigger: documented common error patterns from users
- [ ] Kernel-level error attribution — trigger: users debugging async kernel failures
- [ ] pkg-config support — trigger: user requests from non-CMake build systems

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] Interactive debugger helpers — low priority; advanced users use ncu/nsight
- [ ] CI build matrix — valuable but not blocking; depends on resource availability
- [ ] Header-only diagnostic layer — nice-to-have; requires API stability
- [ ] Build performance diagnostics target — low priority for library users

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| CMake integration with exported targets | HIGH | MEDIUM | P1 |
| compile_commands.json generation | HIGH | LOW | P1 |
| Rich error messages with context | HIGH | MEDIUM | P1 |
| .clangd configuration | HIGH | LOW | P1 |
| ccache integration | MEDIUM | LOW | P1 |
| Error recovery suggestions | MEDIUM | MEDIUM | P2 |
| CMake presets | MEDIUM | LOW | P2 |
| Kernel-level error attribution | MEDIUM | HIGH | P2 |
| pkg-config support | LOW | LOW | P3 |
| Interactive debugger helpers | LOW | MEDIUM | P3 |
| CI build matrix | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for v1.8 launch
- P2: Should have, add in v1.8.x
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | cuBLAS | Thrust | Our Approach |
|---------|--------|--------|--------------|
| CMake integration | Yes, via CMake | Yes, via CMake | Same - export targets to `lib/cmake/nova/` |
| compile_commands.json | Not prominent | Not prominent | Ship `.clangd` for zero-config IDE support |
| Error messages | Raw CUDA errors | Exceptions with basic context | Wrap with device info, kernel context, recovery hints |
| ccache support | Not documented | Not documented | Document in README, auto-detect in CMake |
| IDE config | Not shipped | Not shipped | Ship `.clangd`, `.vscode/settings.json` |

## Sources

- CMake 3.20+ `find_package()` documentation and import/export patterns
- NVIDIA CUDA Toolkit documentation for error codes and diagnostics
- clangd documentation for `.clangd` configuration format
- ccache documentation for CMake integration (`CMAKE_CUDA_COMPILER_LAUNCHER`)
- GoogleTest/GoogleBenchmark patterns for CMake integration (used by project already)
- libcu++ error handling patterns

---
*Feature research for: CUDA Library Developer Experience*
*Researched: 2026-04-26*
