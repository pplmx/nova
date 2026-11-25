# Continuous Integration

Nova uses GitHub-hosted runners only. Those runners have no GPU, and the
project has no self-hosted runner, so anything that needs to touch a device
is verified locally rather than in CI. This page says which checks run where,
so a green CI run is not mistaken for "everything is verified".

## What runs in CI

The checks run on GitHub-hosted runners: `ubuntu-latest`, plus `windows-2022`
for the build in `ci.yml`.

| Check                | Workflow          | Tool                                      | Needs GPU |
| -------------------- | ----------------- | ----------------------------------------- | --------- |
| Format Check         | Testing & Quality | `clang-format` via `make fmt-check`       | no        |
| Fuzz Corpus Baseline | Testing & Quality | corpus vs. base commit diff               | no        |
| Build (Linux/Win)    | CI                | CMake + Ninja + nvcc                      | no        |
| Quick Benchmarks     | Benchmark         | file presence, Python parse, CMake wiring | no        |

None of these compile the benchmark or link a device binary, so none of them
substitute for a local run.

`ci.yml` deliberately stops after the build step. Compiling needs no GPU, but
the demo and the test suite call `cudaMalloc` and abort with
"no CUDA-capable device is detected" on a GPU-less runner (verified by
running the built demo with `CUDA_VISIBLE_DEVICES=""`: exit 134). A step that
always fails is not a signal, so running the binary and `ctest` are local
tasks instead.

## What does NOT run in CI

These are the checks that matter most, and they are local-only:

- **clang-tidy static analysis**
- **Sanitizers (ASan + UBSan)**
- **Unit tests, property tests, fuzz tests**
- **Code coverage**

They were previously defined as jobs on a `[self-hosted, gpu]` runner that
never existed, so the workflow either failed to parse or sat queued forever.
They have been removed rather than left disabled, because a job that can never
run is worse than no job: it hides whether the check passes.

Run them locally before pushing anything that touches device code:

```bash
# Everything that CI does run, plus the gates, from a clean tree
make fmt-check
prek run --all-files

# The full build and test suite (needs a GPU)
cmake -G Ninja -B build -DNOVA_ENABLE_NCCL=OFF -DNOVA_ENABLE_MPI=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure -j16

# Static analysis
# (see the note below: this target does not currently complete on this tree)
make lint

# Compiler warnings, as errors
cmake -G Ninja -B build -DNOVA_ENABLE_NCCL=OFF -DNOVA_ENABLE_MPI=OFF \
    -DNOVA_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel

# ASan + UBSan
make sanitize

# ThreadSanitizer (separate build: TSan cannot share one with ASan)
make tsan
```

### What `make tsan` covers, and what it cannot

It builds its own tree with `-fsanitize=thread` and runs the suite at `-j2`.
Scope matters for reading the result: TSan only reports races in code it
instruments, and a test linked against `libcuda_impl.a` would check the test body
alone. This target compiles the library with the sanitizer as well, which is the
reason it exists -- an earlier attempt that instrumented only the tests reported
nothing while a race inside `MemoryPool` was live.

`-j2` rather than the `-j8` the other targets use, because
`MemoryPoolMetricsTest`'s context-gone case calls `cudaDeviceReset()` on purpose
and a parallel run let that reset land on another case's device.

It needs the GPU to itself. Running it beside `ctest` in another build directory
makes both runs meaningless -- the device is shared, and cases that synchronise or
reset it interfere. Run the suites one at a time.

### `make lint` does not complete

`.clang-tidy` enables `readability-*`, `cppcoreguidelines-*` and `modernize-*` in full, which
produces roughly 22,600 diagnostics on this tree -- magic numbers, uppercase literal suffixes,
identifier length -- and `ninja` stops at the first translation unit that fails. Measured: the target
exits 2 without ever reporting anything a developer can act on.

Of those diagnostics, 30 are `clang-analyzer-*` (the passes that find real bugs). Two of them are in
product code and both are false positives:

- `include/cuda/error/timeout.hpp` -- `message(int ev)` casts a runtime value to `timeout_error_code`
  and has a `default` arm. Out-of-range is the intended input, not a defect.
- `src/cuda/quantize/calibrator.cpp` -- a `float` loop counter (`candidate_scale *= 2.0f`). Doubling
  is exact in binary floating point and the loop always terminates (worst case 138 iterations), so
  there is no drift and no hang.

Rather than leave the command above as an instruction that cannot succeed, this note records the
state. Narrowing the check set is a project decision about what static analysis should mean here, not
a fix to make quietly: it changes which diagnostics are considered an error, and the answer belongs
with whoever owns that policy.

`NOVA_ENABLE_WARNINGS` defaults on and applies `-Wall -Wextra` to host code and, through
`-Xcompiler`, to CUDA translation units as well. Two warnings are exempt for reasons
that are permanent here: unused parameters, because the NCCL sources compile to stubs
when `NOVA_ENABLE_NCCL=OFF`, and deprecated declarations, because the CUDA runtime still
marks the shared-memory bank-size calls deprecated although a test guard legitimately
saves and restores them. `NOVA_WARNINGS_AS_ERRORS=ON` turns the rest into errors; the
whole tree builds clean with it.

Under ASan, `protect_shadow_gap=0` is required. Without it no CUDA context can
be created on this host: `cudaGetDeviceCount` returns
`cudaErrorMemoryAllocation`, so a device test either skips itself or fails at its
first `cudaMalloc` with "out of memory" — measured on a fresh sanitizer binary,
`BufferTest.ConstructionWithSize` failed at `buffer-inl.h:43` and all 9
`DeviceInfoTest` cases failed. `make sanitize` now sets the flag itself through
`NOVA_SANITIZE_ASAN_OPTIONS`, so the target works as written; override that
variable on a host that does not need the workaround.

`make sanitize` also applies `scripts/sanitize/lsan.supp`. `libnccl.so` keeps
480 bytes of calloc'd bookkeeping per communicator until process exit, with no
nova frame in the stack, so without the suppression every process that touches a
communicator fails the run on a leak this project cannot release. The suppression
file states the reason; keep it to third-party libraries only, or the run stops
meaning "something nova did".

One thing to know before adding a replacement for `operator new` to a test
helper. C++ has several forms of it, and the `operator delete` that runs is
chosen by the form used to allocate. Replacing only `operator new(std::size_t)`
leaves `operator new(std::size_t, std::nothrow_t)` going to the runtime's
allocator while the delete for it resolves to the replacement.
`std::get_temporary_buffer` uses the nothrow form and googletest calls it once
per test suite, so ASan then aborts every process with
`alloc-dealloc-mismatch (operator new vs free)` from inside
`testing::TestSuite::Run()`, before any case runs -- a report with no nova frame,
2639 of 2672 cases failing, and nothing pointing at the helper that caused it.
Replace all six points (throwing, nothrow, array, each with its matching delete)
through one pair of helpers. `tests/cuda_alloc_injection.cpp` does this, and a
case in `BufferTest` fails if a future change covers only one form.

```bash
# Coverage
cmake -G Ninja -B build-coverage -DNOVA_COVERAGE=ON
cmake --build build-coverage --parallel
ctest --test-dir build-coverage -j16
./scripts/coverage/generate_coverage.sh build-coverage
bash ./scripts/coverage/coverage_summary.sh build-coverage
```

## Reading a failing local run

A parallel `ctest` run on a machine with other GPU work reports failures that
say nothing about the code. Measured on this host, 2506 tests:

| Run                                    | Failures |
| -------------------------------------- | -------- |
| `-j32`, four GPUs, all shared          | 30       |
| `-j6`, three GPUs with no other tenant | 10       |
| those 10, alone on one GPU             | 0        |

The ten are all in three inference suites (`ChunkedPrefillTest`,
`LongPromptIntegrationTest`, `BeamSpeculativeIntegrationTest`), and they pass
17/17 when nothing else is allocating on the device. Their assertion output is
a `cudaMalloc` failure from `include/cuda/memory/buffer-inl.h`, not a wrong
value. Before treating a failure as a defect, re-run that one case alone:

```bash
CUDA_VISIBLE_DEVICES=N ./bin/nova-tests --gtest_filter='Suite.Case'
```

Cases that compare device-wide memory across time cannot be made reliable this
way, so they are excluded from the parallel discovery set and registered with
`RUN_SERIAL` instead: `RobustnessProfileSerialTest`, `TensorParallelSerialTest`,
`BufferSerialTest`, `UniquePtrSerialTest`, `MemoryMetricsSerialTest`. They still
need a device with no other tenant, because `cudaMemGetInfo` aggregates every
process; a neighbour moving free memory between two samples is indistinguishable
from a leak. Nothing in the build can detect that condition, so if one of them
fails, check what else is on the device before reading anything into it.

### Resetting the CUDA context in a test

Do not call `cudaDeviceReset()` from a test fixture. It tears down the entire process
context, not just the case's own resources, and several process-wide singletons cache
handles that live on that context (`MeshStreams`, `NcclContext`, `CusparseContext`).
Those handles do not become null; they stay as non-null values pointing at a destroyed
context, and the runtime offers no way to detect that from the value:

```cpp
cudaEvent_t e;
cudaEventCreate(&e);
cudaDeviceReset();
cudaGetDevice(&dev);        // succeeds, reports device 0
cudaGetLastError();         // clean
cudaEventDestroy(e);        // SIGSEGV inside libcuda, not an error return
```

Measured exactly as shown, with no sanitizer and no CUPTI involved. The failure mode
depends on the case's memory layout, which is why it shows up under ASan on cases that
pass in a normal build -- `MeshStreamsPartialInitTest.ResetReturnsToTheUninitialisedState`
crashed there in `cudaEventDestroy` and passes alone.

What a fixture usually wants from the reset is only that no work from a previous case is
still in flight. `cudaDeviceSynchronize()` does that and invalidates nothing. All 42 call
sites were converted; the two suites that need a clean context for a different reason
construct their own resources instead of resetting the process.

`cudaDeviceSynchronize` also costs nothing when the intent was to release memory: the
device allocator already reclaims on `cudaFree`, and a reset was never needed for it.

### When a reset is what a test needs: register the handles

`execute_isolated()` does reset the context, so the singletons that own CUDA handles have to
release them first. That is what the reset registry is for, and it is not optional: a handle
from a destroyed context is not merely invalid, it is *silently* invalid, and worse, it
poisons handles created afterwards.

```cpp
cublasCreate(&h);  cudaDeviceReset();  cublasDestroy(h);      // SIGSEGV inside libcuda
cublasCreate(&h);  cudaDeviceReset();  cublasSgemm(h, ...);   // CUBLAS_STATUS_EXECUTION_FAILED
cudaStream_t s;    cudaStreamCreate(&s);  cudaDeviceReset();  // s is now dead,
cublasCreate(&fresh); cublasSetStream(fresh, s);              // ...and this poisons `fresh`
cublasSgemm(fresh, ...);                                      // SIGSEGV inside libcuda
cudaStreamSynchronize(dead_s);  cudaStreamDestroy(dead_s);    // both report cudaSuccess
```

The last two lines are why nothing notices until much later: the stream calls that would
reveal a dead handle both claim success, so the failure surfaces in whichever unrelated call
binds the handle next.

Measured on this host, this cost two things on one suite run: the multi-GPU matmul failed
with `CUBLAS_STATUS_EXECUTION_FAILED` when it ran after `execute_isolated()`, and the process
then died at exit inside `cublasDestroy` after reporting **every test passed** (exit 139,
after the GoogleTest summary). The exit crash came from a static destructor running
`cublasDestroy` during `__run_exit_handlers`, when the driver is already tearing the thread's
context down.

Two rules follow:

1. **A cache of CUDA handles registers a resetter** with `register_singleton_resetter()`,
   from the translation unit that owns the handles. `MeshStreams`, `NcclContext` and the
   per-device cuBLAS handle cache in `distributed/matmul.cu` all do.
2. **Never destroy CUDA handles from a static destructor.** `MeshStreams`, `NcclContext`,
   `CusparseContext` and the cuBLAS cache are all intentionally leaked for this reason; the
   driver reclaims their resources at exit, and a process-wide object whose destructor calls
   into libcuda is a crash waiting for the right teardown order.

The registry itself is reached through function-local accessors, not file-scope statics.
Registration happens in the constructors of static objects in *other* translation units, and
one of those running before the registry's own constructor is a crash before `main()` — which
also breaks `gtest_discover_tests`, so the build fails rather than a test.

`ContextResetRecoveryTest` drives the sequence work → `execute_isolated()` → work again for
each of the three caches. It needs two GPUs and NCCL to reach the multi-GPU path, and skips
for the same environmental reasons as `DistributedMatmulTest`.

## Release workflows

Three workflows respond to a version tag, and all three were dead until the
tag filter was fixed.

The filter was `v*.*.*`, which requires two dots. Every tag in this repository
is `vMAJOR.MINOR` (`v2.12`, `v2.3`, `v2.4`), so nothing ever matched and the
three workflows had a combined recorded run count of 0. GitHub tag filters are
glob patterns, not regular expressions; `v[0-9]*` is used now, which matches
both the existing two-part form and a future `v2.14.0`.

| Workflow      | Does                                                      | Permissions needed |
| ------------- | --------------------------------------------------------- | ------------------ |
| `release.yml` | GitHub Release with notes from `cliff.toml`               | `contents: write`  |
| `docker.yml`  | Builds the image from `Dockerfile`, pushes to GHCR        | `packages: write`  |
| `labeler.yml` | Applies `.github/configs/labeler.yml` to PRs and issues   | `issues: write`    |

The repository default for `GITHUB_TOKEN` is read-only, so each workflow
declares its own `permissions:` block. Without them the release and the image
push fail with 403 while the run still looks like it started correctly.

`cd.yml` was removed. Its only step was `actions/checkout`, so on the first
tag it would have reported a successful deployment having deployed nothing —
the same "green check that proves nothing" the GPU jobs had. Delivery here is
the GitHub Release and the container image.

Notes on the two files that were broken:

- `cliff.toml` had been deleted by accident in `46b22d2`, a commit about tests.
  `git-cliff` only warns when its `--config` is missing and falls back to the
  built-in defaults, so nothing failed; the notes would silently have used the
  wrong style. It is restored, and it now names `pplmx/nova` rather than the
  pre-rename `pplmx/cu` that its commit and compare links are built from.
- `Dockerfile` started from `rockylinux:9`, which has no CUDA toolkit, against
  a project that declares `LANGUAGES ... CUDA` and requires CMake 4.0 (Rocky 9
  ships 3.31.8). It now uses a two-stage CUDA build: `devel` to compile,
  `runtime` — which carries the cuBLAS, cuSOLVER and cuFFT the demo links
  against — to run. The image needs a GPU and the NVIDIA container runtime, so
  `docker.yml` no longer runs it as a smoke test; a hosted runner cannot.

`CHANGELOG.md` is maintained by hand in Keep a Changelog format (see
CONTRIBUTING.md, "Updating the changelog"). `cliff.toml` output is a different
shape, so it is used only for release notes and never written back over that
file. The previous `release.yml` would have replaced it and pushed the result
to `main`.

## Windows

`ci.yml` builds on `windows-2022` as well as `ubuntu-24.04`, and it compiles
with MSVC. That job never reached a compiler: its only recorded build failure
was `CMake 4.0 or higher is required. You are running version 3.31.6`
(run 44473381489), the same CMake pin `ubuntu-24.04` has. With CMake installed
from pip it can configure, so the platform-specific code had to be made to
compile before that job could be anything but red.

MSVC has no `<unistd.h>`, `<fcntl.h>`, `<dirent.h>` or `<sys/statfs.h>`, and no
`/tmp`. Four headers and six tests used them. Each is now either guarded or
routed through a portable call:

| File                                | Was                                 | Now                                                                 |
| ----------------------------------- | ----------------------------------- | ------------------------------------------------------------------- |
| `observability/timeline.h`          | `<unistd.h>` + `::getpid()`         | `<process.h>`/`_getpid()` behind `_WIN32`                           |
| `topology/topology_map.cpp`         | `<unistd.h>`, `setenv`, Linux sysfs | `winsock2.h`, `_putenv_s`, sysfs enumeration Linux-only             |
| `nccl/nccl_validation.cpp`          | `<sys/statfs.h>` unconditionally    | guarded; no-statfs platforms report "not checked"                   |
| `preemption/preemption_handler.cpp` | `sigaction` + `pipe` + `SIGUSR1`    | `signal()` + event; `SIGBREAK`/`SIGINT` on Windows                  |
| six tests                           | `<unistd.h>`, `::getpid()`, `/tmp`  | `tests/test_temp_paths.h`, `std::filesystem::temp_directory_path()` |

The preemption handler is the one with a behavioural difference rather than a
mechanical one. POSIX needs a self-pipe because a handler may only call
async-signal-safe functions; Windows has no such contract and no `pipe()`, so
the handler sets a manual-reset event and a dispatcher thread waits on it. Both
platforms run the callback from that thread, so what callers observe is the
same.

Verified here: each changed translation unit cross-compiles to a Windows target
with the POSIX headers replaced by failing stubs, which proves they are not
reached on `_WIN32`; the Linux build still compiles (full tree) and the
topology, timeline, logger, and signal-handler suites pass (the pre-existing
failures are the `no CUDA-capable device is detected` ones on this GPU-less
machine, and `ProfilingIntegrationTest.MemorySafetyWithTimeline` was confirmed
to fail identically at the previous commit). Not verified: an actual MSVC
build. Cross-compiling with mingw finds portability mistakes but is not the
same compiler, and no MSVC or Windows machine was available here. Until a
`windows-2022` run is green, treat that matrix entry as unproven.

## Adding a GPU runner later

If a machine with the `[self-hosted, gpu]` labels is ever registered, the
removed jobs can be restored from git history (the commit that removed them
carries the full job definitions in its diff). Until then, do not re-add them:
a job whose runner does not exist never reports, and a permanently queued run
looks the same as a passing one in the Actions list.

## Environment notes

- `CMakeLists.txt` requires CMake 4.0 and no runner image ships it: the
  runner-images toolset pins 3.31.6 onto `PATH` for `ubuntu-24.04` and
  `windows-2022` alike, and keeps 4.x only for the Android SDK where the build
  cannot reach it. `ci.yml` therefore installs CMake through pip on both
  platforms and then asserts that `cmake --version` really is 4.x, rather than
  trusting the PATH order to work out.
- `clang-format` is pinned to 18 and the two options whose defaults changed
  between releases are explicit in `.clang-format`; see the comments there.
  Without that pinning the format check passed or failed depending on which
  patch release the runner happened to install.
- The fuzz corpus check diffs the corpus against the base commit, so it needs
  full history: `actions/checkout` is given `fetch-depth: 0`. A shallow clone
  would make the base commit unreadable, and an unreadable base must fail the
  job rather than produce an empty diff that reads as "nothing was deleted".
- Workflow files are validated by `actionlint` in the pre-commit hook, because
  `check-yaml` only proves the file parses: GitHub also rejects valid YAML whose
  expressions are used where the context is unavailable.
