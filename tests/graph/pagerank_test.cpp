#include <gtest/gtest.h>
#include <utility>
#include <vector>
#include <cmath>
#include "cuda/graph/csr_graph.h"
#include "cuda/graph/pagerank.h"

using namespace cuda::graph;

class PageRankTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess && err != cudaErrorNoDevice) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }

    void TearDown() override {
        cudaDeviceSynchronize();
        cudaGetLastError();
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

TEST_F(PageRankTest, PageRankResultIsMovable) {
    PageRankResult a(4);
    a.ranks[0] = 0.5f;

    PageRankResult b = std::move(a);
    EXPECT_EQ(b.num_vertices, 4);
    EXPECT_NEAR(b.rank_at(0), 0.5f, 1e-6f);
    EXPECT_EQ(a.num_vertices, 0);
    EXPECT_EQ(a.ranks, nullptr);
    EXPECT_EQ(a.d_ranks, nullptr);

    PageRankResult c(4);
    c = std::move(b);
    EXPECT_EQ(c.num_vertices, 4);
    EXPECT_NEAR(c.rank_at(0), 0.5f, 1e-6f);
    EXPECT_EQ(b.ranks, nullptr);

    c.clear();  // must not double-free
}

TEST_F(PageRankTest, DefaultOptionsAreSane) {
    PageRankOptions options;
    EXPECT_GT(options.damping, 0.0f);
    EXPECT_LT(options.damping, 1.0f);
    EXPECT_GT(options.tolerance, 0.0f);
    EXPECT_GE(options.max_iterations, 1);
}

// Correct PageRank on a directed path 0 -> 1 -> 2 (2 dangling, out-degree 0).
// With damping d=0.85, teleport t = 0.15/3 = 0.05, the fixed point is
//     r0 = t + d*0        = 0.05        (no incoming edges)
//     r1 = t + d*r0       = 0.0925      (in from 0)
//     r2 = min_rank = 0                  (dangling: rank collapsed per options)
// Uniform initial ranks force several genuine iterations before convergence,
// so a correct implementation must report iterations > 1 and delta -> 0.
TEST_F(PageRankTest, PageRankConvergesOnDirectedPath) {
    std::vector<std::vector<int>> adj(3);
    adj[0] = {1};
    adj[1] = {2};
    auto graph = create_csr_from_adjacency(adj);

    PageRankOptions options;
    options.max_iterations = 200;
    options.tolerance = 1e-7f;
    auto result = pagerank(*graph, options);

    EXPECT_GT(result.iterations, 1);
    EXPECT_LT(result.final_delta, 1e-6f);
    EXPECT_NEAR(result.rank_at(0), 0.05f, 1e-4f);
    EXPECT_NEAR(result.rank_at(1), 0.0925f, 1e-4f);
    EXPECT_NEAR(result.rank_at(2), 0.0f, 1e-4f);

    // Every rank must remain non-negative (the direction bug produced
    // sums over the wrong neighbors, so this also guards against regressions).
    for (int v = 0; v < result.num_vertices; ++v) {
        EXPECT_GE(result.rank_at(v), 0.0f);
    }
}

// A fully isolated graph (no edges at all) must not throw: every vertex has
// out-degree 0, so all ranks collapse to min_rank (the documented dangling
// semantics). This guards the edge-sized kernel launch grid — a 0-block
// launch used to throw a cryptic CUDA error.
TEST_F(PageRankTest, PageRankOnZeroEdgeGraph) {
    std::vector<std::vector<int>> adj(4);  // 4 isolated vertices
    auto graph = create_csr_from_adjacency(adj);

    PageRankOptions options;
    options.max_iterations = 5;
    auto result = pagerank(*graph, options);

    EXPECT_EQ(result.num_vertices, 4);
    for (int v = 0; v < result.num_vertices; ++v) {
        EXPECT_NEAR(result.rank_at(v), options.min_rank, 1e-6f);
    }
}

// A symmetric ring 0 -> 1 -> 2 -> 0 is a fixed point of uniform ranks, so a
// single iteration's delta must already be ~0. This guards the convergence
// machinery against both "converged on iteration 1" false-positives and
// never-converging implementations.
TEST_F(PageRankTest, PageRankIsUniformOnSymmetricRing) {
    std::vector<std::vector<int>> adj(3);
    adj[0] = {1};
    adj[1] = {2};
    adj[2] = {0};
    auto graph = create_csr_from_adjacency(adj);

    PageRankOptions options;
    options.max_iterations = 200;
    options.tolerance = 1e-7f;
    auto result = pagerank(*graph, options);

    EXPECT_EQ(result.iterations, 1);
    EXPECT_LT(result.final_delta, 1e-6f);
    EXPECT_NEAR(result.rank_at(0), 1.0f / 3.0f, 1e-4f);
    EXPECT_NEAR(result.rank_at(1), 1.0f / 3.0f, 1e-4f);
    EXPECT_NEAR(result.rank_at(2), 1.0f / 3.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// The sink-collapsing dangling semantics, pinned by value.
//
// pagerank.h used to say nothing about dangling vertices while the test file
// referred to "the documented dangling semantics" -- so the behaviour was real,
// covered, and undocumented. The header now states it, and these cases measure
// the consequence: ranks do not sum to 1 when the graph has a sink.
// ---------------------------------------------------------------------------

// 0 -> 1 -> 2 with vertex 2 dangling. The rank that reaches 2 is discarded rather than
// redistributed, so the total is 0.1425 rather than 1.0.
TEST_F(PageRankTest, DanglingVertexDiscardsItsRankAndBreaksMassConservation) {
    std::vector<std::vector<int>> adj(3);
    adj[0] = {1};
    adj[1] = {2};
    auto graph = create_csr_from_adjacency(adj);

    PageRankOptions options;
    options.max_iterations = 200;
    options.tolerance = 1e-7f;
    auto result = pagerank(*graph, options);

    double total = 0.0;
    for (int v = 0; v < result.num_vertices; ++v) {
        total += result.rank_at(v);
    }

    // The numbers the header documents, measured rather than derived.
    EXPECT_NEAR(result.rank_at(0), 0.050000f, 1e-5f);
    EXPECT_NEAR(result.rank_at(1), 0.092500f, 1e-5f);
    EXPECT_NEAR(result.rank_at(2), 0.000000f, 1e-6f);
    EXPECT_NEAR(total, 0.142500, 1e-5) << "sink-collapsing semantics discard the dangling vertex's rank; textbook PageRank would conserve it";

    // The point of the case: the total is far from 1. If a future change implements
    // redistribution this assertion is what fails, which is the intended signal.
    EXPECT_LT(total, 0.5) << "a mass-conserving implementation would give total ~1.0";
}

// A vertex with out-degree 0 ends at min_rank regardless of what flowed into it.
TEST_F(PageRankTest, DanglingVertexEndsAtMinRankWhateverReachesIt) {
    std::vector<std::vector<int>> adj(3);
    adj[0] = {2};
    adj[1] = {2};
    auto graph = create_csr_from_adjacency(adj);

    PageRankOptions options;
    options.min_rank = 0.25f;
    options.max_iterations = 100;
    options.tolerance = 1e-7f;
    auto result = pagerank(*graph, options);

    EXPECT_NEAR(result.rank_at(2), options.min_rank, 1e-6f) << "the sink is reset to min_rank, not accumulated into";
}

// min_rank also fills an entirely isolated graph, which is the degenerate case of the
// same rule. The existing PageRankOnZeroEdgeGraph case covers min_rank == 0.
TEST_F(PageRankTest, IsolatedGraphFillsEveryRankWithMinRank) {
    std::vector<std::vector<int>> adj(4);
    auto graph = create_csr_from_adjacency(adj);

    PageRankOptions options;
    options.min_rank = 0.5f;
    options.max_iterations = 5;
    auto result = pagerank(*graph, options);

    double total = 0.0;
    for (int v = 0; v < result.num_vertices; ++v) {
        EXPECT_NEAR(result.rank_at(v), options.min_rank, 1e-6f);
        total += result.rank_at(v);
    }
    EXPECT_NEAR(total, 4.0 * options.min_rank, 1e-5) << "with every vertex dangling, the total is n * min_rank";
}

// A graph with a sink still ranks the reachable vertices monotonically: 0 feeds 1, so 1
// outranks 0. This pins that the collapsing sink does not disturb the ordering upstream
// of it, which is the property a caller actually uses.
TEST_F(PageRankTest, ASinkDoesNotDisturbTheOrderingUpstreamOfIt) {
    std::vector<std::vector<int>> adj(3);
    adj[0] = {1};
    adj[1] = {2};
    auto graph = create_csr_from_adjacency(adj);

    PageRankOptions options;
    options.max_iterations = 200;
    options.tolerance = 1e-7f;
    auto result = pagerank(*graph, options);

    EXPECT_GT(result.rank_at(1), result.rank_at(0)) << "1 receives from 0 and must outrank it";
    EXPECT_EQ(result.top_vertex(), 1) << "the highest-ranked vertex is the one upstream of the sink, not the sink";
}
