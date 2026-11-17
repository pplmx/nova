# Source files not built by CMake

A scan for `.cpp`/`.cu` files under `src/` and `tests/` that no `CMakeLists.txt` mentions
found nine. They are listed here because a source file that is never compiled cannot rot
visibly: nothing fails when its APIs change.

| File | Compiles? | Notes |
|---|---|---|
| `tests/benchmark/inference_benchmark.cpp` | yes, after this round's fix | The namespace `cuda::inference::benchmark` shadowed the Google Benchmark library, so `benchmark::State` resolved to a namespace member. Fixed by qualifying `::benchmark::` and moving `BENCHMARK_MAIN()` outside the namespace. Still not registered in CMake. |
| `tests/property/algorithmic_tests.cpp` | yes | Not registered. |
| `tests/property/mathematical_tests.cpp` | not checked | Not registered. |
| `tests/property/numerical_tests.cpp` | not checked | Not registered. |
| `tests/fuzz/algorithm_fuzz.cpp` | not checked | Not registered. |
| `tests/fuzz/matmul_fuzz.cpp` | yes | Not registered. |
| `tests/fuzz/memory_pool_fuzz.cpp` | not checked | Not registered. |
| `tests/test_matrix.cpp` | **no** | Calls `addMatricesOnGPU`, which no longer exists. Stale. |
| `tests/test_my_lib.cpp` | **no** | Includes `my_lib.h`, which does not exist in the tree. A leftover scaffold. |

## What this means

The four that compile could be registered and run. The two that do not are dead files whose
removal is a decision for the maintainer rather than something this audit should do silently.

Anything relying on these files is relying on unchecked code.
