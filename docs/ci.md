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
make lint

# ASan + UBSan
make sanitize
```

Under ASan, `protect_shadow_gap=0` is required. Without it no CUDA context can
be created on this host: `cudaGetDeviceCount` returns
`cudaErrorMemoryAllocation`, so a device test either skips itself or fails at its
first `cudaMalloc` with "out of memory" — measured on a fresh sanitizer binary,
`BufferTest.ConstructionWithSize` failed at `buffer-inl.h:43` and all 9
`DeviceInfoTest` cases failed. `make sanitize` now sets the flag itself through
`NOVA_SANITIZE_ASAN_OPTIONS`, so the target works as written; override that
variable on a host that does not need the workaround.

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
