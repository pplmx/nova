/**
 * @file boundary_testing_test.cpp
 * @brief Contracts for the CUDA boundary-condition harness
 *
 * src/testing/boundary_testing.cpp enumerates the hardware and API limits a
 * kernel must respect and checks each one. Three of its seventeen functions had
 * any test reference, and two defects made its verdicts meaningless:
 *
 * - run_all_boundaries judged EVERY condition with test_allocation(), so an
 *   alignment or grid-size expectation was decided by whether cudaMalloc
 *   succeeded. A 255-byte allocation and a 65535^2-byte allocation are both
 *   perfectly legal, so "misaligned" and "oversubscribed_grid" reported
 *   "Unexpectedly succeeded" — two false failures out of twelve.
 *
 * - test_memory_boundaries() and test_alignment_boundaries() both returned
 *   run_all_boundaries() unfiltered, so asking for one category gave all twelve
 *   results.
 *
 * BoundaryCondition now carries a BoundaryKind and the harness dispatches on it.
 */

#include <cuda/device/error.h>
#include <cuda/testing/boundary_testing.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

namespace cuda::testing::test {

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

    // Regression: two of the twelve entries failed because the wrong check was
    // applied to them.
    TEST(BoundaryConditionHarnessTest, EveryConditionPassesItsOwnCheck) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        BoundaryConditionTest harness;
        const auto results = harness.run_all_boundaries();

        ASSERT_EQ(results.size(), 12u) << "four memory, four alignment and four compute conditions";
        for (const auto& result : results) {
            EXPECT_TRUE(result.passed) << result.condition_name << " failed: " << (result.error_message ? result.error_message : "(no message)");
        }
    }

    TEST(BoundaryConditionHarnessTest, ConditionsAreDistinctlyNamed) {
        BoundaryConditionTest harness;
        const auto results = harness.run_all_boundaries();

        for (size_t i = 0; i < results.size(); ++i) {
            for (size_t j = i + 1; j < results.size(); ++j) {
                EXPECT_STRNE(results[i].condition_name, results[j].condition_name) << "two conditions share a name, so a report cannot tell them "
                                                                                      "apart";
            }
        }
    }

    TEST(BoundaryConditionHarnessTest, AllocationBoundariesHaveTheExpectedOutcomes) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        BoundaryConditionTest harness;

        // One byte is a legal allocation; zero bytes and SIZE_MAX are not usable.
        EXPECT_TRUE(harness.test_allocation(1));
        EXPECT_FALSE(harness.test_allocation(0));
        EXPECT_FALSE(harness.test_allocation(SIZE_MAX));
        (void)cudaGetLastError();
    }

    TEST(BoundaryConditionHarnessTest, GridAndBlockLimitsAreChecked) {
        BoundaryConditionTest harness;

        // Legal configurations.
        EXPECT_TRUE(harness.test_grid_size(dim3{65535, 1, 1}));
        EXPECT_TRUE(harness.test_grid_size(dim3{1, 65535, 1}));
        EXPECT_TRUE(harness.test_block_size(dim3{256, 1, 1}));
        EXPECT_TRUE(harness.test_block_size(dim3{32, 32, 1}));

        // Over the limit in one axis.
        EXPECT_FALSE(harness.test_grid_size(dim3{65536, 1, 1}));
        EXPECT_FALSE(harness.test_grid_size(dim3{1, 1, 65536}));

        // A zero-sized dimension describes no threads at all.
        EXPECT_FALSE(harness.test_block_size(dim3{0, 1, 1}));
        EXPECT_FALSE(harness.test_block_size(dim3{1, 0, 1}));

        // More threads than the per-block maximum.
        EXPECT_FALSE(harness.test_block_size(dim3{CUDA_MAX_THREADS_PER_BLOCK + 1, 1, 1}));
    }

    // Regression: these returned all twelve results regardless of category.
    TEST(BoundaryConditionHarnessTest, CategoryHelpersReportOnlyTheirOwnCategory) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const auto memory = test_memory_boundaries();
        const auto alignment = test_alignment_boundaries();

        EXPECT_EQ(memory.size(), 4u) << "four memory conditions";
        EXPECT_EQ(alignment.size(), 4u) << "four alignment conditions";

        // The two sets must be disjoint: a name in one cannot appear in the other.
        for (const auto& mem : memory) {
            for (const auto& align : alignment) {
                EXPECT_STRNE(mem.condition_name, align.condition_name) << "condition " << mem.condition_name << " appears in both category reports";
            }
        }

        for (const auto& result : memory) {
            EXPECT_TRUE(result.passed) << result.condition_name;
        }
        for (const auto& result : alignment) {
            EXPECT_TRUE(result.passed) << result.condition_name;
        }
    }

    TEST(BoundaryConditionHarnessTest, WarpBoundariesAreChecked) {
        // The warp helper checks a different property from the alignment list, so it
        // gets its own results.
        const auto warp = test_warp_boundaries();
        ASSERT_EQ(warp.size(), 2u);
        for (const auto& result : warp) {
            EXPECT_TRUE(result.passed) << result.condition_name;
        }
    }

    TEST(BoundaryConditionTestDataTest, EveryConditionDeclaresItsKind) {
        // The dispatch relies on each entry naming the check it exercises; an entry
        // that defaulted to Allocation by accident would be judged by cudaMalloc.
        const auto boundaries = BoundaryConditionTest::get_cuda_boundaries();
        ASSERT_EQ(boundaries.size(), 12u);

        int allocations = 0, alignments = 0, computes = 0;
        for (const auto& condition : boundaries) {
            switch (condition.kind) {
                case BoundaryKind::Allocation:
                    ++allocations;
                    break;
                case BoundaryKind::Alignment:
                    ++alignments;
                    break;
                case BoundaryKind::Compute:
                    ++computes;
                    break;
                case BoundaryKind::SharedMemory:
                    break;
            }
        }
        EXPECT_EQ(allocations, 4);
        EXPECT_EQ(alignments, 4);
        EXPECT_EQ(computes, 4);
    }

    TEST(BoundaryConditionTestDataTest, SharedMemoryLimitIsChecked) {
        BoundaryConditionTest harness;

        EXPECT_TRUE(harness.test_shared_memory(0));
        EXPECT_TRUE(harness.test_shared_memory(48 * 1024));
        EXPECT_FALSE(harness.test_shared_memory(48 * 1024 + 1));
    }

}  // namespace cuda::testing::test
