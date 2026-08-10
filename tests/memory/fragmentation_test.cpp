#include <gtest/gtest.h>
#include "cuda/memory/kv_cache_allocator.h"

namespace cuda::memory {

class FragmentationTest : public ::testing::Test {
protected:
    void SetUp() override {
        KVCacheAllocatorConfig config{
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

    EXPECT_GE(report.num_free_blocks, 0);
    EXPECT_GE(report.ratio, 0.0f);
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
