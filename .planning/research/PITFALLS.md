# Pitfalls Research

**Domain:** CUDA Library Developer Experience — Error Messages, CMake Integration, IDE Support, Build Performance
**Researched:** 2026-04-26
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Generic Error Messages Mask the Root Cause

**What goes wrong:**
CUDA errors surface as opaque codes like `CUDA error: cudaErrorLaunchFailure` with only a file and line number. Users cannot determine which kernel failed, what inputs caused the failure, or how to recover. The error message provides no actionable guidance.

**Why it happens:**
The existing `CUDA_CHECK` macro in `include/cuda/device/error.h` only captures `cudaGetErrorString(err)`, which is the same text NVIDIA's driver returns. It does not capture:
- The kernel or operation that was being performed
- Input dimensions, device count, or memory state at the time of failure
- Recovery suggestions specific to the error type
- The call stack leading to the failure

The `OperationContext` struct exists but is not used by `CUDA_CHECK` — only by the `CUDA_VALIDATE_SIZE` macro. cuBLAS errors print only the integer status code, not the meaningful name.

**How to avoid:**
- Extend `CUDA_CHECK` to accept optional operation context and capture it in the exception
- Add error-category-specific recovery hints: `cudaErrorMemoryAllocation` → suggest reducing batch size, checking for memory leaks; `cudaErrorLaunchFailure` → suggest checking kernel parameters and device compatibility
- Map cuBLAS status codes to readable names and recovery actions (e.g., `CUBLAS_STATUS_NOT_INITIALIZED` → "cuBLAS handle not created. Call cublasCreate() before this operation.")
- Include device ID, stream ID, and operation name in every error
- Use structured error output: `[nova] ERROR in <kernel_name> on device <N>: <error> — suggest: <recovery>`

**Warning signs:**
- Errors say "CUDA error" without naming the operation
- Users report errors without knowing which kernel triggered them
- Error messages are identical for different failure modes
- No recovery guidance in any error message

**Phase to address:**
Phase 1: Error Message Framework — establish structured error taxonomy and context propagation before adding new error types.

---

### Pitfall 2: CMake Exports Produce Non-Relocatable Packages

**What goes wrong:**
Downstream projects that `find_package(nova)` get hard-coded include paths like `/home/user/repos/nova/include` and library paths like `/home/user/repos/nova/build/libnova.a`. The package works on the original machine but breaks when the build directory moves or the project is installed to a system path.

**Why it happens:**
The current `CMakeLists.txt` uses raw paths in `target_include_directories` and does not use generator expressions like `$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>`. When targets are exported, the absolute paths from the build tree are embedded. Additionally, there is no `install(EXPORT)` declaration — the library has no install rules beyond `add_subdirectory` usage.

The existing custom `FindNCCL.cmake` and `FindMPI.cmake` modules create imported targets but do not set `IMPORTED_LOCATION` with generator expressions for relocatable packaging.

**How to avoid:**
- Use `$<BUILD_INTERFACE:>` and `$<INSTALL_INTERFACE:>` generator expressions in all `target_include_directories` and `target_link_directories` calls
- Add `install(TARGETS ... EXPORT novaTargets)` with `install(EXPORT novaTargets FILE novaTargets.cmake NAMESPACE nova:: DESTINATION lib/cmake/nova)`
- Add `include(CMakePackageConfigHelpers)` with `configure_package_config_file()` for a versioned package config
- Replace custom FindXXX modules with CMake's built-in FindNCCL/FindMPI where available, or ensure custom FindXXX modules produce truly relocatable imported targets
- Export only interface libraries and properly-configured static/shared libraries, not OBJECT libraries with absolute paths

**Warning signs:**
- `target_include_directories` uses `${CMAKE_SOURCE_DIR}` without BUILD_INTERFACE generator expression
- No `cmake/novaConfig.cmake.in` or `cmake/novaTargets.cmake` files
- No `install()` commands for the library targets
- Custom FindXXX modules set `IMPORTED_LOCATION` to absolute paths

**Phase to address:**
Phase 2: CMake Package Export — use modern CMake export patterns with generator expressions and install rules.

---

### Pitfall 3: compile_commands.json Missing for CUDA Source Files

**What goes wrong:**
clangd reports thousands of errors for `.cu` files, claims CUDA headers cannot be found, and provides no code completions for CUDA APIs. IDE features work for `.cpp` files but are completely broken for `.cu` files — the files that make up the majority of the codebase.

**Why it happens:**
CMake generates `compile_commands.json` entries for `.cu` files using nvcc's full command line, but clangd uses libclang which cannot parse nvcc-specific flags or CUDA syntax. The entries include flags like `--threads`, `--expt-relaxed-constexpr`, and CUDA architecture flags that confuse the C++ language server. Additionally, CMake's `CMAKE_EXPORT_COMPILE_COMMANDS=ON` creates the file in the build directory but clangd does not search parent directories of symlinked build directories.

The project has no `.clangd` config file to tell clangd how to handle CUDA files, and no `compile_flags.txt` fallback.

**How to avoid:**
- Create `.clangd/config.yaml` with `CompileFlags:` mapping that filters nvcc-specific flags and adds CUDA include paths:
  ```yaml
  CompileFlags:
    Add:
      - "-xcuda"
      - "--cuda-gpu-arch=sm_80"
      - "-I${CMAKE_SOURCE_DIR}/include"
      - "-I${CUDAToolkit_INCLUDE_DIRS}"
    Remove:
      - "-*"
      - "--threads*"
      - "--expt*"
  ```
- Set `CompilationDatabase` in `.clangd/config.yaml` to the build directory path
- Consider using `clangd-vscode` extension with CUDA LSP support, or configure VS Code's `C_Cpp.default.compilerPath` to nvcc with appropriate args
- For VS Code: configure `.vscode/c_cpp_properties.json` with CUDA-specific `compilerPath` (path to nvcc) and `cppStandard`/`cudaPath`
- Ensure `compile_commands.json` is symlinked or copied to project root so clangd finds it without extra configuration

**Warning signs:**
- clangd shows red squiggles on every CUDA kernel definition
- Code completion offers no CUDA runtime API suggestions
- `#include <cuda_runtime.h>` shows "file not found" in IDE
- `.clangd` config file does not exist in project root

**Phase to address:**
Phase 3: IDE Configuration — establish clangd config, VS Code settings, and compile_commands.json root placement.

---

### Pitfall 4: ccache Misses nvcc Cache Keys Due to Architecture Flags

**What goes wrong:**
`ccache` reports a near-0% cache hit rate for CUDA compilation. Every build recompiles all `.cu` files, even when only C++ headers change. The build takes just as long as without ccache, and disk cache grows without providing speedups.

**Why it happens:**
ccache computes the cache key from compiler command line arguments. With `CMAKE_CUDA_ARCHITECTURES` set to `60 70 80 90`, CMake generates four separate nvcc invocations for each `.cu` file (one per SM). The architecture list produces different compiler flags that ccache interprets as different compilations. When architecture list order changes, or when mixing Debug and Release builds, ccache treats them as entirely separate compilations.

Additionally, `--threads ${NCPU}` in `CMAKE_CUDA_FLAGS` injects the host CPU core count into the compiler flags, making every machine produce different cache keys.

**How to avoid:**
- Remove `--threads` from global `CMAKE_CUDA_FLAGS` — use `CMAKE_CUDA_FLAGS_<CONFIG>` or per-target `CUDA_NVCC_FLAGS` instead, with separate handling for build type
- Normalize ccache key for architecture: avoid having multiple near-identical nvcc invocations by letting CMake handle per-architecture compilation rather than baking arch flags into CMAKE_CUDA_FLAGS
- Configure ccache with `sloppiness = include_file_mtime` to ignore minor timestamp changes
- Consider `sccache` instead of ccache — sccache supports distributed caching and has better handling for CUDA's multi-pass compilation, plus the `[cache.disk.preprocessor_cache_mode]` section can cache preprocessor output separately
- Set `CCACHE_BASEDIR` to the source directory to produce relative paths in cache keys

**Warning signs:**
- `ccache -s` shows high cache miss rate (>90%)
- `ccache -s` shows zero hits despite repeated builds of unchanged files
- Build times don't improve on second build
- Cache size grows without bound with no reuse

**Phase to address:**
Phase 4: Build Performance — fix ccache key computation, consider sccache, and optimize unity build configuration.

---

### Pitfall 5: Unity Builds Produce Binary Incompatibilities

**What goes wrong:**
After enabling unity builds with large batch sizes, a previously working CUDA kernel starts producing wrong results or silent data corruption. The issue only manifests with `NOVA_ENABLE_UNITY_BUILD=ON`, and only when batch size exceeds a threshold. Debugging is extremely difficult because the failure is non-deterministic across batch sizes.

**Why it happens:**
Unity builds concatenate multiple `.cu` files into a single compilation unit before passing to nvcc. This causes:
- **Symbol collisions**: if two `.cu` files define identically-named `__device__` functions or static variables, nvcc silently deduplicates or picks one arbitrarily, breaking kernels that expected the other
- **Template bloat amplification**: when each source file includes templates, concatenating N files multiplies template instantiation work, causing nvcc to exceed memory limits
- **Macro collision**: device-side macros with identical names but different definitions get the last definition applied to all concatenated files
- The current `UNITY_BUILD_BATCH_SIZE` only applies to `CMAKE_CUDA_ARCHITECTURES` compilation but not to C++ host-code compilation, creating inconsistent behavior

**How to avoid:**
- Audit all `.cu` files for non-namespaced device symbols before enabling unity builds
- Use `__device__` and `__global__` functions only inside anonymous namespaces or with explicit `__forceinline__`
- Start with `UNITY_BUILD_BATCH_SIZE=4` and verify correctness before increasing
- Add a unity-build-specific test that runs the full test suite: all 444 tests must pass with unity builds before increasing batch size
- Keep `.cu` files with complex device-side symbol interactions (like NCCL internals) excluded from unity builds using `target_sources(nova_impl PRIVATE ...)` with explicit exclusions

**Warning signs:**
- Test suite passes without unity builds but fails with them
- Different results on different runs with same inputs when unity builds enabled
- nvcc memory usage spikes during compilation (visible in `nvidia-smi` on build machine)
- Linker warnings about duplicate symbols that were not present before

**Phase to address:**
Phase 4: Build Performance — validate unity build correctness with full test suite before shipping.

---

### Pitfall 6: No CMake Configuration Validation for Optional Dependencies

**What goes wrong:**
The build succeeds but users discover at runtime that NCCL support was silently disabled because CMake's FindNCCL.cmake could not locate the library. Or MPI is enabled in CMake but the actual `mpirun` binary is missing. Features appear to be present based on CMake output but are non-functional.

**Why it happens:**
The current CMakeLists.txt uses `find_package(NCCL)` but `NCCL_FOUND` is checked only conditionally when creating the `NCCL::nccl` target. When NCCL is not found, `NOVA_NCCL_ENABLED=0` is set as a compile definition, but there is no fatal error for REQUIRED configurations and no user-visible warning that explains what is missing and how to install it.

The `FindNCCL.cmake` custom module uses `find_package_handle_standard_args` with a helpful FAIL_MESSAGE, but the CMakeLists.txt does not call the FindXXX modules in the right order to surface these messages early.

**How to avoid:**
- After all `find_package` calls, print a clear feature matrix: `message(STATUS "NCCL support: ${NCCL_FOUND} (${NCCL_VERSION})")` and `message(STATUS "MPI support: ${MPI_FOUND} (${MPI_VERSION})")`
- For optional features, emit `WARNING` (not FATAL_ERROR) when not found, explaining the trade-off
- Add a CMake option `NOVA_WARN_ABOUT_MISSING_DEPS` that elevates optional-dependency warnings to errors in CI
- Verify the exported CMake config files include dependency status for downstream consumers

**Warning signs:**
- CMake configure step completes with no warnings about missing dependencies
- `NOVA_NCCL_ENABLED` is used at runtime without checking if NCCL was actually found
- No feature matrix in CMake output showing which optional components are available

**Phase to address:**
Phase 2: CMake Package Export — add dependency reporting and validate optional feature configuration.

---

### Pitfall 7: IDE Configuration Files Not Version-Controlled or Tested

**What goes wrong:**
VS Code settings are configured locally but never committed. clangd config is written once and never updated when CMake flags change. New developers get a broken IDE experience and spend hours debugging their setup instead of writing code.

**Why it happens:**
IDE configuration files (`.vscode/`, `.clangd/`, `compile_flags.txt`) are often treated as personal preferences rather than project artifacts. They are excluded via `.gitignore` or simply never created. When CMake flags change, the clangd config becomes stale and reports false errors. VS Code extensions are not specified, so different developers use different tooling with different behaviors.

**How to avoid:**
- Version-control all IDE configuration: `.clangd/config.yaml`, `.vscode/c_cpp_properties.json`, `.vscode/settings.json`, `.vscode/extensions.json`
- Add a CI check that validates `compile_commands.json` is generated and contains entries for all `.cu` and `.cpp` files
- Use `cmake --build build --target help` to verify all expected targets exist
- Include a `DEVELOPERS.md` or `docs/ide-setup.md` that explains required VS Code extensions and clangd configuration for new contributors
- Test IDE configuration by running clangd on the codebase in CI with `--check` mode

**Warning signs:**
- `.vscode/` and `.clangd/` directories are gitignored
- No documentation for IDE setup in the project
- New contributors report "clangd doesn't work" within the first day
- compile_commands.json entries are missing for newly added source files

**Phase to address:**
Phase 3: IDE Configuration — commit IDE files, document setup, and add CI validation.

---

### Pitfall 8: Build Performance Improvements Negatively Impact CI Reproducibility

**What goes wrong:**
ccache and parallel builds are optimized for the developer's machine (128-core build server) but CI runs on 2-core containers with minimal cache. Build times in CI are unchanged from baseline, and the performance tuning provides zero benefit where it matters most. The `--threads ${NCPU}` flag causes nvcc to spawn 128 threads on a machine with 4 cores, degrading performance.

**Why it happens:**
`ProcessorCount(NCPU)` in the current CMakeLists.txt detects CPU cores at configure time and sets `--threads` globally. On a high-core-count developer machine this helps; on a containerized CI runner with CPU limits, it either causes resource exhaustion or uses a stale NCPU value from the host. Unity build batch sizes of 64 and 32 assume abundant memory, which may not be available in CI.

**How to avoid:**
- Remove the global `--threads` nvcc flag or make it conditional on available memory, not just CPU count
- Use CMake's `CMAKE_BUILD_PARALLEL_LEVEL` instead of hardcoding `--parallel` in CMake flags
- Make unity build batch size adaptive: `if(NCPU GREATER_EQUAL 64 AND CMAKE_SYSTEM_MEMORY GREATER 16GB)` — not just CPU count
- Configure ccache size limits appropriate for CI: `CCACHE_MAXSIZE=500M` in CI vs. `CCACHE_MAXSIZE=10G` for developer machines
- Add CI-specific CMake presets that prioritize cache hit rate over absolute build speed

**Warning signs:**
- CI build times don't improve despite ccache being enabled
- CI jobs run out of memory during unity build compilation
- Build outputs differ between developer machines and CI
- `--threads` flag causes nvcc to crash or exceed container memory limits

**Phase to address:**
Phase 4: Build Performance — make performance settings environment-aware and validate CI behavior separately.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoded absolute paths in CMake | Simpler initial setup | Breaks on any other machine or directory move | Never |
| Skip `install(EXPORT)` for internal projects | Less CMake code | Cannot be used as a proper dependency | Only for one-off internal tools |
| Unity builds without testing | Faster local iteration | Silent correctness failures in CI | Only if full test suite runs after enabling |
| ccache without config validation | "Build faster" marketing | Zero cache hits, wasted disk space | Only if cache hit rate is monitored |
| Local-only IDE config | Developer freedom | Onboarding nightmare, inconsistent tooling | Only if docs and CI validation exist |
| Generic error messages | Less code to maintain | Hours lost debugging production errors | Never in production library |
| `CUBLAS_STATUS_SUCCESS` printed as integer | No work needed | Useless for debugging | Never |
| Using `--threads` in global CUDA flags | Parallel host compilation | Cross-machine cache key divergence | Only with explicit per-machine configuration |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| FindCUDAToolkit + custom FindXXX | Duplicate search paths, inconsistent CUDA_ROOT | Use `CUDAToolkit_*` variables from FindCUDAToolkit in custom FindXXX modules |
| CMake export + interface libraries | Interface-only targets can't be installed as binaries | Use `install(TARGETS ... INCLUDES DESTINATION ...)` for INTERFACE targets |
| clangd + nvcc | clangd cannot parse nvcc flags, reports false errors | Filter nvcc-specific flags in `.clangd/config.yaml`, use `-xcuda` flag |
| ccache + nvcc | Cache key includes absolute paths, architecture flags | Set `CCACHE_BASEDIR`, normalize architecture flags, consider sccache |
| sccache + distributed CI | sccache server not available in CI environment | Fall back to local ccache in CI, use sccache only for developer workflows |
| VS Code + CUDA | Default C++ IntelliSense doesn't know CUDA paths | Configure `c_cpp_properties.json` with `compilerPath: nvcc` and CUDA include dirs |
| compile_commands.json + symlinks | clangd doesn't follow symlinks to find compile_commands.json | Copy or symlink compile_commands.json to project root |
| Unity builds + NCCL | NCCL internal symbols collide in concatenated compilation | Keep NCCL-related `.cu` files out of unity build batch |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| `--threads` on low-core machines | nvcc thread spawning overhead exceeds benefit, slower build | Make `--threads` conditional on NCPU >= 16 | On CI containers, laptops, constrained environments |
| Large unity batch sizes | nvcc OOM during compilation, link-time bloat | Start with batch size 4, increase only after testing | On machines with <32GB RAM |
| ccache with no max size | Disk fills up over time, cache becomes ineffective | Set `CCACHE_MAXSIZE` explicitly, prune regularly | In CI environments with limited disk |
| Precompiled header misuse | CUDA headers not precompiled correctly, longer build | Test PCH with unity builds — they interact unexpectedly | When adding precompiled headers to `.cu` files |
| Parallel test + GPU memory | Tests crash with out-of-memory on parallel run | Current `TEST_PARALLEL_LEVEL=16` cap is appropriate, but verify | With new GPU-intensive tests (v1.8+) |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| CMake exported paths from untrusted source tree | If CMakeLists.txt is modified by untrusted party, exported config could contain malicious paths | Validate all `find_package` paths with `file(DOWNLOAD)` or checksum verification for installed packages |
| Exported CMake config leaks credentials | If custom FindXXX modules read from environment variables for credentials, these get baked into config | Never put credential-bearing env vars in exported CMake configs; use `find_package` discovery patterns |
| VS Code settings download extensions from marketplace | Malicious extension could be substituted | Pin extension versions in `.vscode/extensions.json`, use verified publishers |
| compile_commands.json in CI artifacts | Build artifacts with absolute paths leak internal directory structures | CI should sanitize or exclude compile_commands.json from artifacts if security-sensitive |

*Note: This library's DX surface is primarily local build tooling. Primary concerns are CMake package integrity and credential handling in FindXXX modules.*

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No error recovery guidance | Users hit an error and have no idea how to fix it | Every error includes: what went wrong, why it might have happened, concrete recovery steps |
| cuBLAS errors as raw integers | `CublasException: cuBLAS error: -17` means nothing to users | Map status codes to human-readable names: `CUBLAS_STATUS_INVALID_VALUE (-17)` with meaning |
| CMake feature matrix missing | Users don't know which optional features are enabled | Print feature status table at CMake configure time: `nova 0.1.0: NCCL 2.25 [ON], MPI [OFF], Unity Builds [ON]` |
| No developer onboarding docs | New contributors waste time setting up IDE and build environment | `docs/DEVELOPERS.md` with step-by-step: clone, cmake, build, test, IDE setup |
| clangd false errors on CUDA code | Developers ignore or disable clangd entirely | `.clangd/config.yaml` must be tested in CI against the full codebase |
| Build errors without suggested fixes | A failed cmake configure leaves user stranded | CMake errors should suggest specific actions: `NCCL not found — set NCCL_DIR or install NCCL 2.25+` |

---

## "Looks Done But Isn't" Checklist

- [ ] **Error messages:** All CUDA_CHECK and CUBLAS_CHECK macro invocations produce contextual error messages — verify by grepping for non-contextual error sites
- [ ] **Error recovery hints:** Every error type has at least one recovery suggestion — verify coverage for all `cudaError*` codes used in the codebase
- [ ] **CMake export:** `cmake --install build --prefix /tmp/nova-install` produces a relocatable package — verify `find_package(nova)` works from the install prefix
- [ ] **CMake package config:** `novaConfig.cmake` and `novaTargets.cmake` exist in install tree — verify paths are relative via generator expressions
- [ ] **clangd config:** `.clangd/config.yaml` exists and clangd reports zero errors on a clean build — verify with `clangd --check` in CI
- [ ] **clangd CUDA detection:** `#include <cuda_runtime.h>` resolves in clangd — verify include path in compile_commands.json entries
- [ ] **VS Code settings:** `.vscode/c_cpp_properties.json` exists with correct CUDA paths — verify `IntelliSense` mode shows CUDA syntax highlighting
- [ ] **ccache hit rate:** Second build with no changes achieves >80% cache hit rate — verify with `ccache -s` output
- [ ] **sccache (if used):** sccache distributed cache is configured and functional for CUDA — verify cache hits across machines
- [ ] **Unity build correctness:** All 444 tests pass with unity builds enabled — verify in CI with `NOVA_ENABLE_UNITY_BUILD=ON`
- [ ] **compile_commands.json coverage:** All `.cu` and `.cpp` files appear in compile_commands.json — verify entry count matches file count
- [ ] **Feature matrix:** CMake configure output shows NCCL/MPI/UnityBuild status — verify output includes each optional feature

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Generic error messages | MEDIUM | Refactor CUDA_CHECK macro, add OperationContext parameter, audit all call sites; ~2-3 days |
| Non-relocatable CMake packages | HIGH | Rewrite target_include_directories with generator expressions, add install rules; ~1 week |
| clangd not working for CUDA | LOW | Create `.clangd/config.yaml` with CUDA flags, symlink compile_commands.json; ~1 day |
| ccache 0% hit rate | MEDIUM | Remove --threads from global flags, set CCACHE_BASEDIR, adjust sloppiness settings; ~2 hours |
| Unity build correctness failures | MEDIUM | Reduce batch size to 4, run full test suite, identify conflicting symbols; ~2-3 days |
| Missing IDE config files | LOW | Create version-controlled `.vscode/` and `.clangd/` directories with CI validation; ~1 day |
| CI build performance unchanged | MEDIUM | Add CCACHE_MAXSIZE to CI, set environment-aware parallelism; ~1 day |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Generic error messages | Phase 1: Error Message Framework | Unit tests that verify error context is captured; grep all CUDA_CHECK calls for context |
| Non-relocatable CMake packages | Phase 2: CMake Package Export | `find_package(nova)` from install prefix; verify all paths use generator expressions |
| clangd fails on CUDA files | Phase 3: IDE Configuration | `clangd --check` in CI; VS Code extensions validated; `.clangd/config.yaml` committed |
| ccache zero cache hits | Phase 4: Build Performance | `ccache -s` shows >80% hit rate on second build; `--threads` not in cache key |
| Unity build correctness | Phase 4: Build Performance | All 444 tests pass with `NOVA_ENABLE_UNITY_BUILD=ON`; no symbol collision warnings |
| Silent dependency failures | Phase 2: CMake Package Export | Feature matrix in CMake output; optional deps emit WARNING, not silence |
| IDE config not version-controlled | Phase 3: IDE Configuration | `.vscode/` and `.clangd/` in git; CI validates compile_commands.json coverage |
| CI build perf regressions | Phase 4: Build Performance | CI build times tracked; no regressions >10% vs. baseline; adaptive batch sizing |

---

## Sources

- CMake 4.3.2 Documentation — FindCUDAToolkit imported targets, generator expressions, install/export patterns
- CMake 4.3.2 Documentation — cmake-buildsystem(7) for PUBLIC/PRIVATE/INTERFACE propagation rules
- NVIDIA CUDA Documentation — cudaGetErrorString and error recovery recommendations
- NVIDIA cuBLAS Documentation — cublasStatus_t error code meanings
- clangd documentation (clangd.llvm.org) — compile_commands.json configuration and CUDA-specific flags
- sccache (github.com/mozilla/sccache) — distributed CUDA compilation caching, preprocessor cache mode
- NVIDIA Developer Blog — "CUDA Pro Tips" series on error handling and debugging
- CMake Discourse — "Modern CMake and CUDA" patterns for exported packages
- GitHub NVIDIA/CUDA-Samples — CMakeLists.txt patterns for clangd-compatible compile_commands.json

---

*Pitfalls research for: CUDA Library Developer Experience (Error Messages, CMake, IDE Support, Build Performance)*
*Researched: 2026-04-26*
