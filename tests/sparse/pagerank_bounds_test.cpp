/**
 * @file pagerank_bounds_test.cpp
 * @brief PageRankResult::rank_at and top_k must not read past their arrays
 *
 * Two independent gaps in the same class:
 *
 *   rank_at(v)   returned ranks[v] with no check. A default-constructed or cleared result
 *                dereferenced null; an out-of-range v read past the array. Same shape as
 *                CSRGraph::degree, fixed the same way.
 *
 *   top_k(k)     passed scored.begin() + k to std::partial_sort as the middle iterator while
 *                the result loop below it already clamped with std::min(k, num_vertices). Any
 *                k past the vertex count pushed that iterator past end() — undefined behaviour
 *                — and a negative k did the same from the other side.
 *
 * Neither had a test: nothing in the suite called top_k, and the existing rank_at calls all
 * used legal vertices.
 */

#include <cuda/graph/pagerank.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace pagerank_bounds_test {

TEST(PagerankBoundsTest, ALegalVertexStillReportsItsRank) {
    cuda::graph::PageRankResult result(3);
    result.ranks[0] = 0.5F;
    result.ranks[1] = 0.3F;
    result.ranks[2] = 0.2F;

    EXPECT_FLOAT_EQ(result.rank_at(0), 0.5F);
    EXPECT_FLOAT_EQ(result.rank_at(2), 0.2F);
}

TEST(PagerankBoundsTest, AVertexPastTheEndIsRejected) {
    cuda::graph::PageRankResult result(3);
    EXPECT_THROW(result.rank_at(3), std::out_of_range);
    EXPECT_THROW(result.rank_at(99), std::out_of_range);
}

TEST(PagerankBoundsTest, ANegativeVertexIsRejected) {
    cuda::graph::PageRankResult result(3);
    EXPECT_THROW(result.rank_at(-1), std::out_of_range);
}

TEST(PagerankBoundsTest, ADefaultConstructedResultIsRejectedNotFatal) {
    cuda::graph::PageRankResult result;  // ranks is null
    EXPECT_THROW(result.rank_at(0), std::out_of_range);
}

TEST(PagerankBoundsTest, TopKLargerThanTheVertexCountIsClamped) {
    cuda::graph::PageRankResult result(3);
    result.ranks[0] = 0.5F;
    result.ranks[1] = 0.3F;
    result.ranks[2] = 0.2F;

    // Before the clamp this pushed scored.begin() + 10 past the end of a 3-element vector.
    std::vector<int> top;
    ASSERT_NO_THROW(top = result.top_k(10));
    EXPECT_EQ(top.size(), 3u) << "there are only three vertices to return";
    ASSERT_FALSE(top.empty());
    EXPECT_EQ(top[0], 0) << "the highest rank comes first";
}

TEST(PagerankBoundsTest, ANegativeTopKReturnsNothing) {
    cuda::graph::PageRankResult result(3);
    result.ranks[0] = 0.5F;
    result.ranks[1] = 0.3F;
    result.ranks[2] = 0.2F;

    // A negative k made the same iterator arithmetic run backwards.
    std::vector<int> top;
    ASSERT_NO_THROW(top = result.top_k(-1));
    EXPECT_TRUE(top.empty());
}

TEST(PagerankBoundsTest, AZeroTopKReturnsNothing) {
    cuda::graph::PageRankResult result(3);
    std::vector<int> top;
    ASSERT_NO_THROW(top = result.top_k(0));
    EXPECT_TRUE(top.empty());
}

}  // namespace pagerank_bounds_test
