/**
 * @file tensor_parallel_profile_test.cpp
 * @brief Contracts for the tensor-parallel sizing and memory queries
 *
 * src/cuda/neural/tensor_parallel_profile.cpp implements the profiler that sizes
 * a TP shard and recommends a degree. Two of its five functions were referenced
 * from a test, and one of them was wrong:
 *
 *   size_t total_mem = 0;
 *   if (cudaMemGetInfo(nullptr, &total_mem) != cudaSuccess) return 0;
 *   return total_mem;
 *
 * cudaMemGetInfo(&free, &total) puts FREE memory in its first out-parameter and
 * the device CAPACITY in its second. Passing nullptr for free and returning
 * total therefore handed back the card's total size — always 100% of the device,
 * whatever was actually allocated. Measured: it returned 85094825984 on a device
 * whose free memory was 84649377792.
 *
 * CUDA exposes no high-water mark, so the function now reports current usage and
 * the header says so.
 */

#include <cuda/neural/tensor_parallel_profile.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda::neural::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

    }  // namespace

    // Regression: this returned the device total, not its usage.
    // Same constraint as AvailableMemoryIsFreeMemory: it compares two device-wide
    // snapshots, which a neighbour's allocations can shift between reads.
    TEST(TensorParallelSerialTest, PeakMemoryUsageReportsUsageNotCapacity) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        size_t free_mem = 0;
        size_t total_mem = 0;
        ASSERT_EQ(cudaMemGetInfo(&free_mem, &total_mem), cudaSuccess);

        const size_t reported = TensorParallelProfiler::peak_memory_usage();

        // The defect being pinned is returning the device CAPACITY, which is always
        // ~100% regardless of allocation. Exact equality with total-free cannot be
        // required: peak_memory_usage reads its own snapshot, and under the parallel
        // test runner another process may allocate between the two calls.
        EXPECT_NE(reported, total_mem) << "reporting the device capacity makes the number independent of what "
                                          "was allocated";

        if (total_mem > free_mem) {
            const size_t expected_usage = total_mem - free_mem;
            // A little more than the whole device allows for a snapshot that moved
            // in either direction without letting a capacity-sized answer through.
            const size_t slack = total_mem / 20;  // 5%
            EXPECT_LE(reported, expected_usage + slack) << "usage cannot exceed what the snapshots accounted for";
            EXPECT_GE(reported + slack, expected_usage) << "usage must not read as near-zero when memory is held";
        }
    }

    // available_memory was always correct; this pins its contract rather than
    // guarding a past defect. The peak_memory_usage regression is the separate test
    // above, which is where the swapped-argument bug actually lived.
    // Device-wide snapshots cannot be compared under a parallel test run: cudaMemGetInfo
    // aggregates every process, so a neighbour allocating between the two reads looks
    // like a wrong answer. A 1% margin failed first, then 20%; the right fix is to give
    // the measurement the device to itself, so it lives in the serial suite registered
    // with RUN_SERIAL rather than being loosened until it can no longer detect anything.
    TEST(TensorParallelSerialTest, AvailableMemoryIsFreeMemory) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        size_t free_mem = 0;
        size_t total_mem = 0;
        ASSERT_EQ(cudaMemGetInfo(&free_mem, &total_mem), cudaSuccess);

        const size_t available = TensorParallelProfiler::available_memory(0);

        // Another process can take memory between the two calls. Under `ctest -j16`
        // with ~1900 tests, other GPU tests allocate and free freely, so the drift is
        // far larger than a serial run suggests: a 1% margin failed in the full suite
        // while passing in isolation. The property being pinned is still meaningful at
        // a wide margin — available_memory must reflect FREE memory, not the device
        // capacity — because returning `total` would be off by the entire allocation.
        ASSERT_LE(available, total_mem);
        const size_t slack = total_mem / 5;  // 20%
        EXPECT_LE(available, free_mem);
        EXPECT_GE(available + slack, free_mem);
    }

    TEST(TensorParallelProfilerTest, ProfileSizesTheShardAndActivation) {
        const int batch = 1, seq = 128, hidden = 512, tp = 2;

        const auto profile = TensorParallelProfiler::profile(batch, seq, hidden, tp);

        // Weight shard: (3 + 8) * hidden^2 * 4 bytes / tp.
        const size_t expected_weight = static_cast<size_t>(hidden) * hidden * 11 * sizeof(float) / tp;
        EXPECT_EQ(profile.weight_shard_bytes, expected_weight);

        // Activation: batch * seq * hidden * 4 bytes, not divided by tp.
        const size_t expected_activation = static_cast<size_t>(batch) * seq * hidden * sizeof(float);
        EXPECT_EQ(profile.activation_bytes, expected_activation);

        // Gradients are twice the activation in this model.
        EXPECT_EQ(profile.gradient_bytes, expected_activation * 2);

        EXPECT_EQ(profile.total_bytes, profile.weight_shard_bytes + profile.activation_bytes + profile.gradient_bytes);
    }

    TEST(TensorParallelProfilerTest, HigherDegreeShrinksTheWeightShard) {
        const auto two = TensorParallelProfiler::profile(1, 128, 512, 2);
        const auto four = TensorParallelProfiler::profile(1, 128, 512, 4);

        EXPECT_LT(four.weight_shard_bytes, two.weight_shard_bytes) << "splitting across more devices must shrink each shard";
        EXPECT_EQ(four.weight_shard_bytes * 2, two.weight_shard_bytes) << "a doubled degree halves the shard";

        // Activations are replicated, not sharded.
        EXPECT_EQ(four.activation_bytes, two.activation_bytes);
    }

    TEST(TensorParallelProfilerTest, RecommendedDegreeIsAtLeastOne) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        EXPECT_GE(TensorParallelProfiler::recommend_tp_degree(1, 128, 512), 1) << "the recommendation must describe a usable configuration";
    }

    TEST(TensorParallelProfilerTest, RecommendedDegreeFitsTheDevice) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        int device_count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
        if (device_count < 1) {
            GTEST_SKIP() << "no devices";
        }

        const int recommended = TensorParallelProfiler::recommend_tp_degree(1, 128, 512);

        EXPECT_LE(recommended, device_count) << "cannot recommend more shards than there are devices";
    }

    TEST(TensorParallelProfilerTest, MaxTpForBudgetRespectsTheLimit) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // max_tp_for_budget takes (hidden_dim, memory_budget_bytes). A budget of a
        // single byte can only support one device.
        const auto tiny = TensorParallelProfiler::max_tp_for_budget(512, 1);
        EXPECT_GE(tiny, 1) << "a budget must still yield a usable configuration";
        EXPECT_EQ(tiny, 1) << "1 byte cannot support more than one shard";
    }

    TEST(TensorParallelProfilerTest, MaxTpForHugeBudgetIsPositive) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const auto generous = TensorParallelProfiler::max_tp_for_budget(512, static_cast<size_t>(1) << 40);
        EXPECT_GE(generous, 1);
    }

}  // namespace cuda::neural::test
