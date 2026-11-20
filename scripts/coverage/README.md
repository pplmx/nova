# Nova Coverage Infrastructure

This directory contains scripts and configuration for generating code coverage reports.

## Quick Start

```bash
# Build with coverage
cmake -B build-coverage -DNOVA_COVERAGE=ON
cmake --build build-coverage

# Run tests
ctest --test-dir build-coverage

# Generate coverage report
./scripts/coverage/generate_coverage.sh build-coverage
# Output: build-coverage/coverage/index.html
```

## Scripts

| Script                 | Purpose                       |
| ---------------------- | ----------------------------- |
| `generate_coverage.sh` | Generate full coverage report |
| `coverage_gaps.sh`     | Show untested functions       |
| `coverage_summary.sh`  | Per-module breakdown          |

## CMake Options

| Option              | Description                             |
| ------------------- | --------------------------------------- |
| `NOVA_COVERAGE`     | Enable coverage instrumentation         |
| `NOVA_COVERAGE_MIN` | Minimum required coverage (default: 80) |

## Coverage Report Structure

```text
build/coverage/
├── index.html          # Main HTML report
├── lcov/               # Raw lcov data files
│   ├── app.info
│   └── total.info
├── module/             # Per-module reports
│   ├── memory.info
│   ├── algo.info
│   └── neural.info
└── gaps/               # Coverage gaps
    └── untested_functions.txt
```

## Requirements

| ID      | Requirement           | Provided by                                    |
| ------- | --------------------- | ---------------------------------------------- |
| COVR-01 | HTML coverage report  | `generate_coverage.sh` (genhtml)               |
| COVR-02 | Coverage gap analysis | `coverage_gaps.sh`                             |
| COVR-03 | Per-module breakdown  | `generate_coverage.sh` + `coverage_summary.sh` |

## Dependencies

- lcov (Linux: `apt install lcov`)
- genhtml (part of lcov)
- Clang or GCC with coverage support

## Threshold

`coverage_summary.sh` exits non-zero when line coverage is below the minimum, so
it can be the gate in a script. The minimum comes from, in order: an exported
`NOVA_COVERAGE_MIN`, the `NOVA_COVERAGE_MIN` value in the build's CMakeCache.txt
(what `-DNOVA_COVERAGE_MIN=90` writes), then a default of 80.

The cache lookup matters: the option is a CMake CACHE variable and is never
exported, so the script used to enforce 80 no matter what the build was
configured with.

```bash
cmake -B build-coverage -DNOVA_COVERAGE=ON -DNOVA_COVERAGE_MIN=90
# ... build and run tests ...
./scripts/coverage/generate_coverage.sh build-coverage
bash ./scripts/coverage/coverage_summary.sh build-coverage   # fails below 90%
```

## CI

Coverage does not run in CI: it needs a GPU to execute the suite, and the project
has no self-hosted runner. See the "What does NOT run in CI" section of
[docs/ci.md](../../docs/ci.md) for the local command sequence.
