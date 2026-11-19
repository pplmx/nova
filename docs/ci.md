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

# Coverage
cmake -G Ninja -B build-coverage -DNOVA_COVERAGE=ON
cmake --build build-coverage --parallel
ctest --test-dir build-coverage -j16
./scripts/coverage/generate_coverage.sh build-coverage
bash ./scripts/coverage/coverage_summary.sh build-coverage
```

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
