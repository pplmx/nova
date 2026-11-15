#pragma once

/**
 * @file nvbench_integration.h
 * @brief Optional NVBench glue, compiled only when NVBench is available
 *
 * This header is NOT part of the default build: nvbench is not a dependency of
 * this project (benchmarks use Google Benchmark, see benchmark/benchmark_kernels.cu),
 * and <nvbench/nvbench.cuh> is not on the include path. Nothing in the tree
 * includes this file, which is why it had never been compiled.
 *
 * Every call in the previous version named an API that does not exist in any
 * released NVBench:
 *
 *   .set_timeout_ms(5000)                       -> set_timeout() takes SECONDS
 *                                                  as a float64, not a
 *                                                  millisecond integer
 *   .add_device_memory(name, args)              -> does not exist; the real call
 *                                                  is add_device(device_info)
 *   state.collect_keeps_bandwidth_throughput_true(...)  -> does not exist
 *   state.add_float64_metric("GFLOPS")          -> does not exist
 *
 * The surviving helpers below use only APIs that NVBench actually provides
 * (NVBENCH_BENCH, add_float64_axis, set_timeout, collect_dram_throughput), and
 * the whole file is guarded so including it without NVBench present is a no-op
 * rather than a hard error.
 */

#include <cstddef>

#if defined(NOVA_NVBENCH_ENABLED) && NOVA_NVBENCH_ENABLED

#include <nvbench/nvbench.cuh>

/**
 * @def NOVA_BENCHMARK_KERNEL(name)
 * @brief Declare an NVBench benchmark and let the kernel size axes come from the
 *        benchmark itself; NVBench owns device memory via add_device().
 */
#define NOVA_BENCHMARK_KERNEL(name) NVBENCH_BENCH(name)

/**
 * @def NOVA_BENCHMARK_MEMORY_BANDWIDTH(name, timeout_seconds)
 * @brief Declare a bandwidth benchmark with a measurement timeout
 *
 * The timeout is in SECONDS, matching nvbench::benchmark_base::set_timeout.
 */
#define NOVA_BENCHMARK_MEMORY_BANDWIDTH(name, timeout_seconds) \
    NVBENCH_BENCH(name).set_timeout(timeout_seconds)

namespace cuda::benchmark {

/**
 * @brief Report the bytes moved by one kernel launch
 *
 * NVBench derives bandwidth from this; there is no collector named
 * "collect_keeps_bandwidth_throughput" and no add_float64_metric, so callers
 * add the axis themselves with add_float64_axis.
 */
template <typename T>
constexpr double bytes_processed(std::size_t element_count) {
    return static_cast<double>(element_count) * static_cast<double>(sizeof(T));
}

}  // namespace cuda::benchmark

#else

// Without NVBench, the macros still have to be usable (so a source file that
// mentions them can be listed in CMake and simply pit) and must not expand into
// expressions that would need the missing types.
#define NOVA_BENCHMARK_KERNEL(name)
#define NOVA_BENCHMARK_MEMORY_BANDWIDTH(name, timeout_seconds)

namespace cuda::benchmark {

/**
 * @brief Bytes moved by one kernel launch
 *
 * Kept available in both configurations so callers can compute and print a
 * throughput figure without NVBench.
 */
template <typename T>
constexpr double bytes_processed(std::size_t element_count) {
    return static_cast<double>(element_count) * static_cast<double>(sizeof(T));
}

}  // namespace cuda::benchmark

#endif
