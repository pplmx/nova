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

## Compiles, not registered

| File                              | Entry point              |
| --------------------------------- | ------------------------ |
| `tests/fuzz/algorithm_fuzz.cpp`   | `LLVMFuzzerTestOneInput` |
| `tests/fuzz/matmul_fuzz.cpp`      | `LLVMFuzzerTestOneInput` |
| `tests/fuzz/memory_pool_fuzz.cpp` | `LLVMFuzzerTestOneInput` |

"Compiles" is the whole claim — see the Fuzz targets section below for what each one does when
it runs, which is not the same thing.

The three property tests are registered now: each has its own `main` and is built as its own
executable with its own ctest entry (`property_algorithmic`, `property_mathematical`,
`property_numerical`).

## Fuzz targets

Only one of the three is worth a driver.

| File                              | State                                                                                                                            |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `tests/fuzz/algorithm_fuzz.cpp`   | Calls the library (`cuda::algo::reduce_sum`, scan, sort). Needs a libFuzzer driver.                                              |
| `tests/fuzz/matmul_fuzz.cpp`      | Calls nothing from nova — it multiplies with its own triple loop and checks that loop's output. Fuzzing it fuzzes this file.     |
| `tests/fuzz/memory_pool_fuzz.cpp` | Broken: `memset()` on a `cudaMalloc` pointer from the host. Would crash the harness, not the pool. Also never uses `MemoryPool`. |

Each of the two has a note at the top of the file saying the same thing.

## Dead

| File                    | Why                                                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_matrix.cpp` | two of its four cases call `addMatricesOnGPU`, removed from the tree; its multiply cases do match the current `multiplyMatricesOnGPU` |
| `tests/test_my_lib.cpp` | includes `my_lib.h`, which is not in the tree; a gtest tutorial template                                                              |

All four files carry a note at the top saying so. Removing them, and giving
`algorithm_fuzz.cpp` a driver, are decisions for the maintainer. This file records the state
rather than making those choices silently.
