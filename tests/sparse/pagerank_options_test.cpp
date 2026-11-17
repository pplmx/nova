/**
 * @file pagerank_options_test.cpp
 * @brief PageRank's options were never validated
 *
 * Three fields drove the iteration with no checks:
 *
 *   damping         teleport = (1 - damping) / n. At exactly 1 the random walk never jumps, so
 *                   the iteration has no fixed point and can cycle until max_iterations; above
 *                   1 the "probabilities" it accumulates are negative.
 *   tolerance       a negative value makes delta < tolerance false for every iteration, so the
 *                   loop always runs to max_iterations.
 *   max_iterations  zero or negative makes the loop body never run, returning the uniform
 *                   initial ranks as though that were a converged answer.
 *
 * All three are rejected now with a message naming the field and the value.
 */

#include <cuda/graph/pagerank.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace pagerank_options_test {

namespace {

std::unique_ptr<cuda::graph::CSRGraph> cycle() {
    std::vector<std::vector<int>> adjacency(3);
    adjacency[0] = {1};
    adjacency[1] = {2};
    adjacency[2] = {0};
    return cuda::graph::create_csr_from_adjacency(adjacency);
}

}  // namespace

TEST(PagerankOptionsTest, ADampingOfExactlyOneIsRejected) {
    auto graph = cycle();
    cuda::graph::PageRankOptions options;
    options.damping = 1.0F;

    auto call = [&] { return cuda::graph::pagerank(*graph, options); };
    EXPECT_THROW(call(), std::invalid_argument)
        << "damping = 1 leaves no teleport and the walk has no fixed point";
}

TEST(PagerankOptionsTest, ADampingAboveOneIsRejected) {
    auto graph = cycle();
    cuda::graph::PageRankOptions options;
    options.damping = 1.5F;

    auto call = [&] { return cuda::graph::pagerank(*graph, options); };
    EXPECT_THROW(call(), std::invalid_argument)
        << "damping > 1 makes the accumulated weights negative";
}

TEST(PagerankOptionsTest, ANegativeDampingIsRejected) {
    auto graph = cycle();
    cuda::graph::PageRankOptions options;
    options.damping = -0.1F;

    auto call = [&] { return cuda::graph::pagerank(*graph, options); };
    EXPECT_THROW(call(), std::invalid_argument);
}

TEST(PagerankOptionsTest, AZeroOrNegativeToleranceIsRejected) {
    auto graph = cycle();
    cuda::graph::PageRankOptions options;
    options.tolerance = 0.0F;

    auto zero = [&] { return cuda::graph::pagerank(*graph, options); };
    EXPECT_THROW(zero(), std::invalid_argument);

    options.tolerance = -1e-6F;
    auto negative = [&] { return cuda::graph::pagerank(*graph, options); };
    EXPECT_THROW(negative(), std::invalid_argument);
}

TEST(PagerankOptionsTest, ANonPositiveIterationCountIsRejected) {
    auto graph = cycle();
    cuda::graph::PageRankOptions options;
    options.max_iterations = 0;

    auto zero = [&] { return cuda::graph::pagerank(*graph, options); };
    EXPECT_THROW(zero(), std::invalid_argument);

    options.max_iterations = -5;
    auto negative = [&] { return cuda::graph::pagerank(*graph, options); };
    EXPECT_THROW(negative(), std::invalid_argument);
}

TEST(PagerankOptionsTest, TheDefaultsStillProduceRanksSummingToOne) {
    auto graph = cycle();
    cuda::graph::PageRankOptions options;  // 0.85, 1e-6, 50

    cuda::graph::PageRankResult result(0);
    ASSERT_NO_THROW(result = cuda::graph::pagerank(*graph, options));
    ASSERT_EQ(result.num_vertices, 3);

    float total = 0.0F;
    for (int v = 0; v < 3; ++v) {
        total += result.rank_at(v);
    }
    EXPECT_NEAR(total, 1.0F, 1e-3F) << "the guards must not disturb the ordinary path";
}

}  // namespace pagerank_options_test
