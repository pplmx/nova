# Source files not built by CMake

A scan for `.cpp`/`.cu` files under `src/` and `tests/` that no `CMakeLists.txt` mentions
found nine. A source file that is never compiled cannot rot visibly: nothing fails when its
APIs change. This round checked each one and fixed what was broken.

## Fixed

| File                                      | Was                                                                                                                                                   | Now                                                                                                                                        |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `tests/benchmark/inference_benchmark.cpp` | 15 compile errors: `namespace cuda::inference::benchmark` shadowed the Google Benchmark library, so `benchmark::State` resolved to a namespace member | `::benchmark::` qualified, `BENCHMARK_MAIN()` moved out of the namespace; also sizes its attention buffers from the benchmark's own config |
| `tests/fuzz/algorithm_fuzz.cpp`           | `cuda/algo/reduce.hpp`, `cuda/algo/scan.hpp`, `cuda/algo/sort.hpp` — none of those paths exist                                                        | corrected to `cuda/algo/reduce.h`, `parallel/scan.h`, `parallel/sort.h`                                                                    |
| `tests/fuzz/memory_pool_fuzz.cpp`         | `cuda/memory/memory_pool.hpp` — the header is `.h`                                                                                                    | corrected                                                                                                                                  |

All three now compile.

## Fuzz targets: build state

All three build and run under `-DNOVA_BUILD_FUZZ_TESTS=ON` with a Clang toolchain, driven
by libFuzzer. That was not true before round 5: the option silently skipped the targets on
any compiler without Clang, the runtime check searched for a file name no distribution
ships, and the targets never linked the sanitizer runtime. The details are in the commit
that fixed them.

"Builds and runs" is not the same as "tests the library". See below for what each one
actually does.

The three property tests are registered now: each has its own `main` and is built as its own
executable with its own ctest entry (`property_algorithmic`, `property_mathematical`,
`property_numerical`).

## Fuzz targets: what each one tests

Two of the three are worth a driver.

| File                              | State                                                                                                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/fuzz/algorithm_fuzz.cpp`   | Calls the library (`cuda::algo::reduce_sum`, scan, sort) and checks each result.                                                                              |
| `tests/fuzz/matmul_fuzz.cpp`      | Calls nothing from nova — it multiplies with its own triple loop and checks that loop's output. Fuzzing it fuzzes this file.                                  |
| `tests/fuzz/memory_pool_fuzz.cpp` | Calls `MemoryPool` (`allocate` / `deallocate` / `defragment` / `get_metrics`) over a fuzzed sequence of operations, and checks its invariants after each one. |

`matmul_fuzz.cpp` carries a note at the top saying the same thing.

`algorithm_fuzz.cpp` caps its input at 1024 elements, which is exactly `MAX_SCAN_SIZE`:
raising it would only make the scan branch reject, and bitonicSort's own 1024-element limit
was raised in round 5 by changing its launch shape. The cap is a property of the harness, not
of the three operations (reduce and bitonic sort both accept more).

## Dead

| File                    | Why                                                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_matrix.cpp` | two of its four cases call `addMatricesOnGPU`, removed from the tree; its multiply cases do match the current `multiplyMatricesOnGPU` |
| `tests/test_my_lib.cpp` | includes `my_lib.h`, which is not in the tree; a gtest tutorial template                                                              |

All four files carry a note at the top saying so. Removing them is a decision for the
maintainer. This file records the state rather than making those choices silently.
