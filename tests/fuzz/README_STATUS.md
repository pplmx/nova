# Source files not built by CMake

A scan for `.cpp`/`.cu` files under `src/` and `tests/` that no `CMakeLists.txt` mentions
found nine. A source file that is never compiled cannot rot visibly: nothing fails when its
APIs change. This round checked each one and fixed what was broken.

## Fixed

| File | Was | Now |
|---|---|---|
| `tests/benchmark/inference_benchmark.cpp` | 15 compile errors: `namespace cuda::inference::benchmark` shadowed the Google Benchmark library, so `benchmark::State` resolved to a namespace member | `::benchmark::` qualified, `BENCHMARK_MAIN()` moved out of the namespace; also sizes its attention buffers from the benchmark's own config |
| `tests/fuzz/algorithm_fuzz.cpp` | `cuda/algo/reduce.hpp`, `cuda/algo/scan.hpp`, `cuda/algo/sort.hpp` — none of those paths exist | corrected to `cuda/algo/reduce.h`, `parallel/scan.h`, `parallel/sort.h` |
| `tests/fuzz/memory_pool_fuzz.cpp` | `cuda/memory/memory_pool.hpp` — the header is `.h` | corrected |

All three now compile.

## Compiles, not registered

| File | Entry point |
|---|---|
| `tests/fuzz/algorithm_fuzz.cpp` | `LLVMFuzzerTestOneInput` |
| `tests/fuzz/matmul_fuzz.cpp` | `LLVMFuzzerTestOneInput` |
| `tests/fuzz/memory_pool_fuzz.cpp` | `LLVMFuzzerTestOneInput` |
| `tests/property/algorithmic_tests.cpp` | `int main()` |
| `tests/property/mathematical_tests.cpp` | `int main()` |
| `tests/property/numerical_tests.cpp` | `int main()` |

The three fuzz targets are libFuzzer entries: they need a fuzzer driver to run and cannot be
ctest cases as written. The three property tests each define their own `main` and are
standalone programs, so registering them means three separate executables and three ctest
entries, not adding them to nova-tests.

## Dead

| File | Why |
|---|---|
| `tests/test_matrix.cpp` | calls `addMatricesOnGPU`, which no longer exists anywhere in the tree |
| `tests/test_my_lib.cpp` | includes `my_lib.h`, which is not in the tree; a leftover scaffold |

Removing these, registering the working ones, and giving the fuzz targets a driver are
decisions for the maintainer. This file records the state rather than making those choices
silently.
