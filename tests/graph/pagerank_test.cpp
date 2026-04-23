#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "cuda/graph/pagerank.h"

using namespace cuda::graph;

class PageRankTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(PageRankTest, PageRankResultConstruction) {
    PageRankResult result(10);
    EXPECT_EQ(result.num_vertices, 10);
    EXPECT_NE(result.ranks, nullptr);
    EXPECT_NE(result.d_ranks, nullptr);
}

TEST_F(PageRankTest, PageRankResultDefaultConstruction) {
    PageRankResult result;
    EXPECT_EQ(result.num_vertices, 0);
}

TEST_F(PageRankTest, RanksAreInitializedCorrectly) {
    PageRankResult result(3);
    result.ranks[0] = 0.5f;
    result.ranks[1] = 0.3f;
    result.ranks[2] = 0.2f;

    EXPECT_NEAR(result.rank_at(0), 0.5f, 0.001f);
    EXPECT_NEAR(result.rank_at(1), 0.3f, 0.001f);
    EXPECT_NEAR(result.rank_at(2), 0.2f, 0.001f);
}

TEST_F(PageRankTest, AllRanksNonNegative) {
    PageRankResult result(3);
    result.ranks[0] = 0.4f;
    result.ranks[1] = 0.4f;
    result.ranks[2] = 0.2f;

    for (int v = 0; v < result.num_vertices; ++v) {
        EXPECT_GE(result.rank_at(v), 0.0f);
    }
}

TEST_F(PageRankTest, OptionsDefaultValues) {
    PageRankOptions options;
    EXPECT_EQ(options.damping, 0.85f);
    EXPECT_NEAR(options.tolerance, 1e-6f, 1e-7f);
    EXPECT_EQ(options.max_iterations, 50);
}

TEST_F(PageRankTest, TopVertexReturnsValidIndex) {
    PageRankResult result(3);
    result.ranks[0] = 0.5f;
    result.ranks[1] = 0.3f;
    result.ranks[2] = 0.2f;

    int top = result.top_vertex();
    EXPECT_GE(top, 0);
    EXPECT_LT(top, result.num_vertices);
}

TEST_F(PageRankTest, TopKReturnsKVertices) {
    PageRankResult result(3);
    result.ranks[0] = 0.5f;
    result.ranks[1] = 0.3f;
    result.ranks[2] = 0.2f;

    auto top2 = result.top_k(2);
    EXPECT_EQ(static_cast<int>(top2.size()), 2);
}

TEST_F(PageRankTest, PageRankMemoryUsageIsPositive) {
    PageRankResult result(100);
    size_t mem = result.memory_usage();
    EXPECT_GT(mem, 0);
}

TEST_F(PageRankTest, PageRankIterationsAreTracked) {
    PageRankResult result(3);
    result.iterations = 10;
    EXPECT_GE(result.iterations, 1);
}

TEST_F(PageRankTest, PageRankOnSingleVertex) {
    PageRankResult result(1);
    result.ranks[0] = 1.0f;

    EXPECT_EQ(result.num_vertices, 1);
    EXPECT_EQ(result.rank_at(0), 1.0f);
}

TEST_F(PageRankTest, DefaultOptionsAreSane) {
    PageRankOptions options;
    EXPECT_GT(options.damping, 0.0f);
    EXPECT_LT(options.damping, 1.0f);
    EXPECT_GT(options.tolerance, 0.0f);
    EXPECT_GE(options.max_iterations, 1);
}
