/**
 * @file pagerank_empty_graph_test.cpp
 * @brief pagerank and pagerank_iteration must not run on a zero-vertex graph
 *
 * Two independent failures on the same input:
 *
 *   pagerank()            begins with std::fill(ranks, ranks + n, 1.0f / n), which divides by
 *                         zero when n == 0.
 *   pagerank_iteration()  computes vertex_grid = (n + 255) / 256 — 0 blocks, which the driver
 *                         rejects — and teleport = (1 - damping) / n. Only its edge grid was
 *                         clamped with std::max(1, ...).
 *
 * A graph with no vertices has no ranks and no iteration to perform; returning an empty result
 * is the meaningful answer.
 */

#include <cuda/graph/pagerank.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>

namespace pagerank_empty_graph_test {

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

class PagerankEmptyGraphTest : public ::testing::Test {
protected:
    void TearDown() override { (void)cudaGetLastError(); }
};

TEST_F(PagerankEmptyGraphTest, PagerankOnAZeroVertexGraphReturnsAnEmptyResult) {
    cuda::graph::CSRGraph graph;  // no vertices
    ASSERT_EQ(graph.vertices(), 0);

    cuda::graph::PageRankOptions options;
    options.max_iterations = 3;

    cuda::graph::PageRankResult result(0);
    ASSERT_NO_THROW(result = cuda::graph::pagerank(graph, options));
    EXPECT_EQ(result.num_vertices, 0) << "there are no ranks to report";
}

TEST_F(PagerankEmptyGraphTest, TheIterationOnAZeroVertexGraphDoesNothing) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::graph::CSRGraph graph;  // no vertices
    // The guard returns before any launch, so null device pointers are never dereferenced.
    EXPECT_NO_THROW(
        cuda::graph::pagerank_iteration(graph, nullptr, nullptr, 0.85F, 0.0F, nullptr));
}

TEST_F(PagerankEmptyGraphTest, AnOrdinaryGraphStillComputesRanks) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The guards must not disable the real path: three isolated vertices and a couple of
    // iterations still produce a result with three ranks summing to about 1.
    std::vector<std::vector<int>> adjacency(3);
    adjacency[0] = {1};
    adjacency[1] = {2};
    adjacency[2] = {0};
    auto graph = cuda::graph::create_csr_from_adjacency(adjacency);

    cuda::graph::PageRankOptions options;
    options.max_iterations = 5;
    auto result = cuda::graph::pagerank(*graph, options);

    ASSERT_EQ(result.num_vertices, 3);
    float total = 0.0F;
    for (int v = 0; v < 3; ++v) {
        const float rank = result.rank_at(v);
        EXPECT_TRUE(rank > 0.0F) << "vertex " << v << " has rank " << rank;
        total += rank;
    }
    EXPECT_NEAR(total, 1.0F, 1e-3F) << "a cycle distributes all the rank";
}

}  // namespace pagerank_empty_graph_test
