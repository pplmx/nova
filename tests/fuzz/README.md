# Fuzz Testing

This directory contains fuzzing infrastructure for Nova using libFuzzer.

## Directory Structure

- `corpus/` - Seed corpus for each fuzz target (committed to git)
    - `memory_pool/` - Seeds for memory pool fuzzing
    - `algorithm/` - Seeds for algorithm fuzzing
    - `matmul/` - Seeds for matmul fuzzing
- `crashes/` - Crash artifacts (gitignored, for local analysis)

The committed corpus is the baseline: CI compares against the base commit
rather than against a checked-in copy, so there is no `baseline/` directory
to maintain.

## Running Fuzz Tests

### Build with Fuzzing

```bash
cmake -B build -DNOVA_BUILD_FUZZ_TESTS=ON
cmake --build build
```

### Run Individual Fuzz Targets

```bash
# Memory pool fuzzing
./build/bin/memory_pool_fuzz tests/fuzz/corpus/memory_pool -max_total_time=60

# Algorithm fuzzing
./build/bin/algorithm_fuzz tests/fuzz/corpus/algorithm -max_total_time=60

# Matmul fuzzing
./build/bin/matmul_fuzz tests/fuzz/corpus/matmul -max_total_time=60
```

## CMake Targets

When `NOVA_BUILD_FUZZ_TESTS=ON`:

- `make fuzz_memory_pool` - Run memory pool fuzzing
- `make fuzz_algorithms` - Run algorithm fuzzing
- `make fuzz_matmul` - Run matmul fuzzing
- `make fuzz_all` - Run all fuzzing targets

## CI Fuzzing

CI does not run a fuzzer. The only fuzz-related job is a corpus integrity
check in `testing-quality.yml`, which fails when a pull request deletes a
committed seed; adding seeds is fine. It compares against the base commit, so
there is no checked-in baseline to keep in sync.

Two caveats on the numbers: the committed corpus has a handful of seeds, not
the 1000 entries an earlier version of this file claimed, and the targets are
built only when `NOVA_BUILD_FUZZ_TESTS=ON` (see `CMakeFuzz.txt`). What each
target is actually worth fuzzing is recorded in `README_STATUS.md` — two of
the three should not be driven at all.

## Fuzzer Details

### Memory Pool Fuzzing (FUZZ-01)

Tests allocation/deallocation patterns and edge cases in memory management.

### Algorithm Fuzzing (FUZZ-02)

Tests reduce, scan, and sort operations with varied input sizes and values.

### Matmul Fuzzing (FUZZ-03)

Tests matrix multiplication with varied tensor shapes and precision modes.
