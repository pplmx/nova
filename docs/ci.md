# Continuous Integration

Nova uses GitHub-hosted runners only. Those runners have no GPU, and the
project has no self-hosted runner, so anything that needs to touch a device
is verified locally rather than in CI. This page says which checks run where,
so a green CI run is not mistaken for "everything is verified".

## What runs in CI

All three workflows run on GitHub-hosted runners: `ubuntu-latest`, plus
`windows-2022` for the build in `ci.yml`.

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

# Coverage
cmake -G Ninja -B build-coverage -DNOVA_COVERAGE=ON
cmake --build build-coverage --parallel
ctest --test-dir build-coverage -j16
./scripts/coverage/generate_coverage.sh build-coverage
bash ./scripts/coverage/coverage_summary.sh build-coverage
```

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
