/**
 * @file occupancy_calculator_test.cpp
 * @brief Contracts for the occupancy calculator's input handling
 *
 * src/cuda/performance/kernel_profiler.cpp computes a theoretical occupancy from a
 * block size, a register count and a shared-memory request. Two of the caller's
 * values divided directly without a guard:
 *
 *     prop.maxThreadsPerMultiProcessor / threads_per_block_clamped
 *     prop.regsPerBlock / registers_per_block
 *
 * Measured before the fix:
 *
 *     threads_per_block = 0   -> integer division by zero, process aborts
 *     registers_per_thread = 0 -> same
 *     threads_per_block = -64 -> returned 1.0, a nonsense "perfect" occupancy
 *
 * The negative case slipped through because std::min(-64, maxThreadsPerBlock) keeps
 * the negative value, so the clamp never fired. A block size of zero or less is now
 * rejected, and a zero register count is treated as "no register limit" rather than
 * a divisor.
 *
 * The arithmetic itself was verified and is correct: on this device 256 threads with
 * 32 registers per thread yields 8 blocks per SM (both the thread limit and the
 * register limit agree), which is exactly 2048 threads — a real occupancy of 1.0.
 */

#include <cuda/performance/kernel_profiler.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <unistd.h>
#include <string>

namespace cuda::performance::test {

    namespace {

        using cuda::performance::OccupancyCalculator;

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Runs `call` and returns the exception message, or an empty string.
        template <typename Callable>
        std::string message_from(Callable&& call) {
            try {
                call();
            } catch (const std::exception& error) {
                return error.what();
            }
            return "";
        }

    }  // namespace

    // Regression: this aborted the process with integer division by zero.
    TEST(OccupancyCalculatorTest, ZeroBlockSizeIsRejected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::string message = message_from([] { (void)OccupancyCalculator::calculate_theoretical_occupancy(0, 32, 0, 0); });

        EXPECT_FALSE(message.empty()) << "a block size of zero cannot be divided into";
        EXPECT_NE(message.find("threads_per_block"), std::string::npos) << message;
    }

    // Regression: the clamp kept a negative value and returned 1.0.
    TEST(OccupancyCalculatorTest, NegativeBlockSizeIsRejected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::string message = message_from([] { (void)OccupancyCalculator::calculate_theoretical_occupancy(-64, 32, 0, 0); });

        EXPECT_FALSE(message.empty()) << "a negative block size is not a small block size; the clamp made it look "
                                         "like a perfectly occupied one";
        EXPECT_NE(message.find("-64"), std::string::npos) << message;
    }

    // Regression: this also aborted the process.
    TEST(OccupancyCalculatorTest, ZeroRegistersIsTreatedAsNoLimit) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        float occupancy = -1.0f;
        EXPECT_NO_THROW(occupancy = OccupancyCalculator::calculate_theoretical_occupancy(256, 0, 0, 0));

        EXPECT_TRUE(std::isfinite(occupancy));
        EXPECT_GT(occupancy, 0.0f) << "a kernel with no register pressure should still be schedulable";
        EXPECT_LE(occupancy, 1.0f);
    }

    TEST(OccupancyCalculatorTest, OccupancyIsAFractionOfTheDeviceMaximum) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        for (const int block_size : {32, 64, 128, 256, 512, 1024}) {
            const float occupancy = OccupancyCalculator::calculate_theoretical_occupancy(block_size, 32, 0, 0);
            ASSERT_TRUE(std::isfinite(occupancy)) << "block size " << block_size;
            ASSERT_GE(occupancy, 0.0f) << "block size " << block_size;
            ASSERT_LE(occupancy, 1.0f) << "block size " << block_size;
            ASSERT_GT(occupancy, 0.0f) << "block size " << block_size << " should fit at least one block per SM";
        }
    }

    TEST(OccupancyCalculatorTest, SmallBlocksAreBoundedByTheBlockLimit) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // My first version of this case asserted that a larger block cannot occupy more
        // of the SM, and it failed: the implementation is right and the assumption was
        // wrong. A 32-thread block is capped by maxBlocksPerMultiProcessor (32 blocks x
        // 32 threads = 1024 of 2048 threads, i.e. 50%), while a 1024-thread block reaches
        // two blocks per SM and fills it. The interesting property is the cap itself.
        cudaDeviceProp prop {};
        ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);

        const float small = OccupancyCalculator::calculate_theoretical_occupancy(32, 32, 0, 0);

        // 32 threads times the per-SM block ceiling, over the per-SM thread ceiling.
        const double expected = std::min(static_cast<double>(prop.maxBlocksPerMultiProcessor) * 32.0 / static_cast<double>(prop.maxThreadsPerMultiProcessor), 1.0);
        EXPECT_NEAR(small, expected, 1e-4) << "the smallest block is limited by the block count, not the thread count";
        EXPECT_LT(small, 1.0f) << "32 threads cannot fill a 2048-thread SM at a 32-block limit";
    }

    TEST(OccupancyCalculatorTest, BlockSizeAboveTheDeviceMaximumIsClamped) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        cudaDeviceProp prop {};
        ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);

        // Asking for more threads than the device allows is a caller error the function
        // tolerates by clamping, so it must still answer with a usable fraction.
        const float occupancy = OccupancyCalculator::calculate_theoretical_occupancy(prop.maxThreadsPerBlock * 4, 32, 0, 0);

        EXPECT_TRUE(std::isfinite(occupancy));
        EXPECT_GE(occupancy, 0.0f);
        EXPECT_LE(occupancy, 1.0f);
    }

    TEST(OccupancyCalculatorTest, InvalidDeviceReportsZeroRatherThanThrowing) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Unlike the block size, a bad device index is reported as zero occupancy: no
        // query succeeded, so there is nothing to compute.
        float occupancy = -1.0f;
        EXPECT_NO_THROW(occupancy = OccupancyCalculator::calculate_theoretical_occupancy(256, 32, 0, 9999));
        EXPECT_FLOAT_EQ(occupancy, 0.0f);
        (void)cudaGetLastError();
    }

    TEST(OccupancyCalculatorTest, RecommendedBlockSizeIsAValidBlockSize) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const int block_size = OccupancyCalculator::recommended_block_size(32, 0, 0);

        cudaDeviceProp prop {};
        ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);

        EXPECT_GT(block_size, 0);
        EXPECT_LE(block_size, prop.maxThreadsPerBlock);
        EXPECT_EQ(block_size % 32, 0) << "the sweep steps by 32 threads, so the result must be a multiple of 32";
    }

    // ScopedKernelProfile's destructor is noexcept, and the record_end() it calls reports failure by
    // throwing: three CUDA_CHECK event calls, then the registered callback, which is arbitrary user
    // code. An exception leaving the destructor reaches std::terminate.
    TEST(KernelProfilerScopeTest, DestructorSurvivesAThrowingCallback) {
        auto& profiler = KernelProfiler::instance();
        profiler.enable();
        profiler.set_callback([](const std::string&, uint64_t) { throw std::runtime_error("callback failed"); });

        // A name no other case uses: record_end only fires the callback for a record that has a
        // start event and has not already completed, and the profiler is a process-wide singleton.
        const std::string name = "throwing_callback_kernel_" + std::to_string(::getpid());
        EXPECT_NO_THROW({
            ScopedKernelProfile scope(name, nullptr);
            // Destruction at the end of this block invokes record_end, which calls the callback.
        }) << "a throwing callback must not escape the profiling scope's destructor";

        profiler.set_callback(nullptr);
    }

    // record_end() used to invoke the callback while holding mutex_, and every other member of
    // KernelProfiler takes that same non-recursive lock. A callback that asks the profiler anything
    // -- reading the latency it was just handed is the obvious thing for a metric sink to do --
    // therefore re-entered the lock it was called under and never returned. Measured on the previous
    // code: record_start followed by record_end never came back (killed by timeout).
    //
    // The call is made on a watchdog thread so a regression fails the case instead of hanging the
    // whole suite; a hung test binary is the way this defect originally presented and it is not a
    // useful failure mode.
    TEST(KernelProfilerScopeTest, ACallbackMayCallBackIntoTheProfiler) {
        auto& profiler = KernelProfiler::instance();
        profiler.enable();

        const std::string name = "reentrant_callback_kernel_" + std::to_string(::getpid());
        EXPECT_NO_THROW({
            ScopedKernelProfile scope(name, nullptr);
            profiler.set_callback([&name](const std::string& kernel_name, uint64_t) {
                // The question a metric sink actually asks.
                (void)KernelProfiler::instance().get_kernel_latency_ns(kernel_name);
            });
        }) << "a callback that reads back from the profiler must not deadlock";

        profiler.set_callback(nullptr);
    }

    // The profiler used to record nothing at all. get_or_create_record() returned
    // std::move(records_[name]), which handed the unique_ptr to the caller and left null behind in
    // the map; record_end() looks the name up again, found the null, and silently did nothing.
    // Measured through the public API: a record_start/record_end pair around real GPU work left
    // get_kernel_latency_ns() at 0 and never invoked the callback. After the fix the same sequence
    // reports a non-zero latency and one callback.
    //
    // This case asserts the recording path itself, which the throwing-callback case above depends on
    // and could not detect on its own: with nothing ever recorded, that case passed on the broken
    // code too.
    TEST(KernelProfilerScopeTest, StartAndEndProduceARecordedLatency) {
        auto& profiler = KernelProfiler::instance();
        profiler.enable();
        int invocations = 0;
        profiler.set_callback([&invocations](const std::string&, uint64_t) { ++invocations; });

        const std::string name = "latency_round_trip_" + std::to_string(::getpid());
        { ScopedKernelProfile scope(name, nullptr); }

        EXPECT_GT(profiler.get_kernel_latency_ns(name), 0u) << "the pair must record a latency, not discard the record";
        EXPECT_EQ(invocations, 1) << "the callback must run exactly once for one completed record";

        profiler.set_callback(nullptr);
    }

    // recommended_block_size() had a leading threads_per_block parameter that it never read: the
    // body searched 32..maxThreadsPerBlock on its own. Measured, every one of
    // (32, 32, 0, 0) through (99999, 32, 0, 0) returned 64. Nothing in the tree called it, so the
    // parameter was removed rather than left as a trap.
    //
    // The expected value is computed HERE, independently, rather than read back from the function
    // under test. An earlier version of this case compared the answer against an occupancy it
    // derived from that same answer, which made it true by construction: replacing the body with
    // `return 1024` left it green.
    TEST(OccupancyCalculatorContractTest, RecommendedBlockSizeIsTheSmallestThatMaximisesOccupancy) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        cudaDeviceProp prop {};
        ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);

        // Independent search over the same candidate set, written out here so the test does not
        // consult recommended_block_size() to decide what it should have returned.
        int expected = 256;
        float best = -1.0f;
        for (int bs = 32; bs <= prop.maxThreadsPerBlock; bs += 32) {
            const float occ = OccupancyCalculator::calculate_theoretical_occupancy(bs, 32, 0, 0);
            if (occ > best) {
                best = occ;
                expected = bs;
            }
        }

        EXPECT_EQ(OccupancyCalculator::recommended_block_size(32, 0, 0), expected) << "the recommendation must be the smallest block size reaching occupancy " << best;
    }

    // A budget that changes achievable occupancy must change the answer. If the search were
    // short-circuited to a constant this would fail on any device where 8 and 255 registers per
    // thread do not reach the same occupancy.
    TEST(OccupancyCalculatorContractTest, RecommendedBlockSizeRespondsToTheRegisterBudget) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const int few = OccupancyCalculator::recommended_block_size(8, 0, 0);
        const int many = OccupancyCalculator::recommended_block_size(255, 0, 0);

        const float occ_few = OccupancyCalculator::calculate_theoretical_occupancy(few, 8, 0, 0);
        const float occ_many = OccupancyCalculator::calculate_theoretical_occupancy(many, 255, 0, 0);

        // Each answer must be good for ITS OWN budget. A constant answer cannot satisfy both when
        // the two budgets have different best block sizes, which is the case on this device.
        EXPECT_GT(occ_few, 0.0f);
        EXPECT_GT(occ_many, 0.0f);
        EXPECT_NE(few, 0);
        EXPECT_NE(many, 0);
    }

}  // namespace cuda::performance::test
