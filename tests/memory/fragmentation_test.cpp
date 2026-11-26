#include <gtest/gtest.h>
#include "cuda/memory/kv_cache_allocator.h"

namespace cuda::memory {

    class FragmentationTest : public ::testing::Test {
    protected:
        void SetUp() override {
            KVCacheAllocatorConfig config {
                .num_heads = 32,
                .head_dim = 128,
                .block_size_tokens = 16,
                .num_blocks = 256,
                .num_layers = 32,
            };
            allocator = std::make_unique<KVCacheAllocator>(config);
        }

        std::unique_ptr<KVCacheAllocator> allocator;
    };

    TEST_F(FragmentationTest, AnalyzeFragmentation) {
        auto report = allocator->analyze_fragmentation();

        // Both assertions used to be `>= 0` on a freshly built allocator: num_free_blocks is an int
        // and ratio a float, so the pair held for every value the implementation could return --
        // including the all-zero report produced by a ratio branch that never runs or fields that
        // are never filled. That is the same shape as the defect that makes needs_compaction()
        // permanently false, so the case was blind to the failure it most needed to catch.
        //
        // The fixture builds 256 blocks and allocates none, so the answers are exact.
        EXPECT_EQ(report.num_free_blocks, 256) << "every block is free in a fresh allocator";
        EXPECT_FLOAT_EQ(report.ratio, 100.0f) << "all 256 blocks free is 100% fragmentation by this report's definition";
        // The report measures free space in block_size_tokens units, not bytes (see
        // analyze_fragmentation: total_free accumulates config_.block_size_tokens per free block).
        EXPECT_FLOAT_EQ(report.avg_free_block_size, 16.0f) << "one free run of uniform blocks, measured in tokens";
        EXPECT_FLOAT_EQ(report.largest_free_block_size, 16.0f);
    }

    TEST_F(FragmentationTest, NeedsCompactionThreshold) {
        // analyze_fragmentation().ratio is the FREE percentage (pinned by
        // FragmentationReportEmptyPool/FullPool), and needs_compaction(t) triggers
        // when free% > t (pinned by NeedsCompactionBelowThreshold/AboveThreshold and
        // NeedsCompactionEdgeCases). A fresh allocator is 100% free, so it reports
        // "needs compaction" for any threshold below 100.
        EXPECT_TRUE(allocator->needs_compaction(30.0f));

        EXPECT_TRUE(allocator->needs_compaction(0.0f));
    }

    TEST_F(FragmentationTest, FragmentationAfterAllocations) {
        for (int i = 0; i < 10; ++i) {
            allocator->allocate(i, 32);
        }

        auto report_before = allocator->analyze_fragmentation();
        EXPECT_LT(report_before.num_free_blocks, 256);

        for (int i = 0; i < 5; ++i) {
            allocator->free(i);
        }

        auto report_after = allocator->analyze_fragmentation();
        EXPECT_GT(report_after.num_free_blocks, report_before.num_free_blocks);
    }

    TEST_F(FragmentationTest, CompactReducesHoles) {
        for (int i = 0; i < 50; ++i) {
            allocator->allocate(i, 16);
        }

        for (int i = 0; i < 25; ++i) {
            allocator->free(i);
        }

        auto report_before = allocator->analyze_fragmentation();

        allocator->compact();

        auto report_after = allocator->analyze_fragmentation();

        EXPECT_LE(report_after.num_free_blocks, report_before.num_free_blocks);
    }

}  // namespace cuda::memory
