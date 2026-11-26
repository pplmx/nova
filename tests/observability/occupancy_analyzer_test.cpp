#include <gtest/gtest.h>

#include "cuda/observability/occupancy_analyzer.h"

namespace cuda::observability::test {

    class OccupancyAnalyzerTest : public ::testing::Test {
    protected:
        void SetUp() override { analyzer_ = std::make_unique<OccupancyAnalyzer>(0); }

        std::unique_ptr<OccupancyAnalyzer> analyzer_;
    };

    TEST_F(OccupancyAnalyzerTest, ConstructorQueriesDeviceInfo) {
        EXPECT_GE(analyzer_->device(), 0);
        EXPECT_GE(analyzer_->sm_count(), 1);
        EXPECT_GE(analyzer_->max_threads_per_sm(), 1);
    }

    TEST_F(OccupancyAnalyzerTest, AnalyzeNullptrReturnsZeroOccupancy) {
        auto analysis = analyzer_->analyze(nullptr, 256, 0);

        EXPECT_EQ(analysis.theoretical_occupancy, 0.0);
        EXPECT_EQ(analysis.active_blocks_per_sm, 0);
    }

    TEST_F(OccupancyAnalyzerTest, RecommendReturnsValidBlockSize) {
        auto rec = analyzer_->recommend(nullptr, 0);

        EXPECT_GE(rec.recommended_block_size, 32);
        EXPECT_LE(rec.recommended_block_size, 1024);
    }

    TEST_F(OccupancyAnalyzerTest, OccupancyInValidRange) {
        auto analysis = analyzer_->analyze(nullptr, 256, 0);

        EXPECT_GE(analysis.theoretical_occupancy, 0.0);
        EXPECT_LE(analysis.theoretical_occupancy, 1.0);
    }

    TEST_F(OccupancyAnalyzerTest, AnalyzeRangeReturnsMultipleResults) {
        auto results = analyzer_->analyze_range(nullptr, 32, 128, 0);

        EXPECT_GE(results.size(), 1);
        for (const auto& result : results) {
            EXPECT_GE(result.max_threads_per_block, 32);
            EXPECT_LE(result.max_threads_per_block, 128);
        }
    }

    TEST_F(OccupancyAnalyzerTest, BlockSizeFeedbackToString) {
        BlockSizeFeedback fb;
        fb.block_size = 256;
        fb.occupancy = 0.75;
        fb.is_optimal = true;

        std::string str = fb.to_string();
        EXPECT_TRUE(str.find("256") != std::string::npos);
        EXPECT_TRUE(str.find("75") != std::string::npos);
        EXPECT_TRUE(str.find("OPTIMAL") != std::string::npos);
    }

    TEST_F(OccupancyAnalyzerTest, AnalyzeBlockSizesReturnsResults) {
        auto results = analyze_block_sizes(nullptr, 0, 0);

        EXPECT_GE(results.size(), 1);

        double max_occupancy = 0.0;
        int optimal_count = 0;
        for (const auto& fb : results) {
            max_occupancy = std::max(max_occupancy, fb.occupancy);
            if (fb.is_optimal) {
                optimal_count++;
            }
        }

        EXPECT_EQ(optimal_count, 1);
    }

    TEST_F(OccupancyAnalyzerTest, MinBlocksForFullOccupancy) {
        auto analysis = analyzer_->analyze(nullptr, 256, 0);

        // analyze() answers a null kernel_func with an all-zero result — it cannot ask the driver
        // about a kernel it was not given. My first attempt at this case asserted a high occupancy
        // and failed for that reason; the zero result is the contract, and the conditional guard
        // that used to be here meant the contract was never stated in either direction.
        EXPECT_DOUBLE_EQ(analysis.theoretical_occupancy, 0.0) << "a null kernel has no occupancy to report";
        EXPECT_EQ(analysis.min_blocks_for_full_occupancy, 0);
        EXPECT_EQ(analysis.active_blocks_per_sm, 0);

        // What the call does report is the geometry it was asked about.
        EXPECT_EQ(analysis.max_threads_per_block, 256);
    }

    TEST_F(OccupancyAnalyzerTest, RecommendedGridSizeCalculation) {
        auto rec = analyzer_->recommend(nullptr, 0);

        // Same null-kernel contract as analyze(). The comment above this line used to say the
        // assertion "used to be `>= 0` on a size, true by construction and checking nothing" --
        // and the assertion right below it still was exactly that. recommended_grid_size is a
        // size_t, so `>= 0` holds for every value, including a grid computed from a null kernel.
        //
        // recommend() with no kernel returns early with a fixed record, so the contract is an
        // exact zero rather than a bound.
        EXPECT_DOUBLE_EQ(rec.expected_occupancy, 0.0) << "a null kernel has no occupancy to recommend against";
        EXPECT_EQ(rec.recommended_grid_size, 0u) << "a null kernel cannot be given a grid to launch";
        EXPECT_EQ(rec.recommended_block_size, 256u) << "the documented default block size for the empty recommendation";
        EXPECT_EQ(rec.shared_memory_bytes, 0u);
    }

}  // namespace cuda::observability::test
