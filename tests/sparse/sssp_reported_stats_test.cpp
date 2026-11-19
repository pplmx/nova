/**
 * @file sssp_reported_stats_test.cpp
 * @brief Contracts for what a shortest-path run reports about itself
 *
 * include/cuda/algo/sssp.h returned a result whose converged and iterations fields
 * were hard-coded:
 *
 *   result.converged = true;
 *   result.iterations = 0;
 *
 * They therefore carried no information at all. Worse, Bellman-Ford had no
 * negative-cycle detection, so a graph whose distances are unbounded below was
 * reported as a converged run. Measured on the cycle 1 -> 2 -> 1 with weights
 * -3 and +1 (total -2):
 *
 *   converged=1 iterations=0
 *   distances: 0.00 -3.00 -4.00
 *
 * Those distances are not shortest paths — no shortest path exists — but nothing in
 * the result said so. The solver now counts its relaxation rounds and runs the
 * definitive extra pass: an edge that still relaxes after n-1 rounds proves a
 * negative cycle reachable from the source.
 */

#include <cuda/algo/sssp.h>

#include <gtest/gtest.h>

#include <cuda/graph/csr_graph.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace cuda::algo::sssp::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Builds a graph from edge lists.
        std::unique_ptr<cuda::graph::CSRGraph> make_graph(const std::vector<int>& sources, const std::vector<int>& targets, const std::vector<float>& weights, int num_vertices) {
            return cuda::graph::create_csr_from_edges(sources.data(), targets.data(), weights.data(), num_vertices, static_cast<int>(sources.size()));
        }

        std::vector<float> read_distances(const SSSPResult& result) {
            std::vector<float> host(static_cast<size_t>(result.num_vertices), 0.0f);
            result.distances.copy_to(host.data(), host.size());
            return host;
        }

    }  // namespace

    // Regression: converged was hard-coded true and iterations hard-coded 0.
    TEST(SsspReportedStatsTest, AHealthyRunReportsRealProgress) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // The chain 0 -> 1 -> 2 needs two relaxation rounds to settle.
        auto graph = make_graph({0, 1}, {1, 2}, {1.0f, 2.0f}, 3);
        ASSERT_NE(graph, nullptr);

        const auto result = run(*graph, 0);

        EXPECT_TRUE(result.converged);
        EXPECT_FALSE(result.negative_cycle);
        EXPECT_GT(result.iterations, 0) << "a run that relaxed edges must report the rounds it used";
    }

    TEST(SsspReportedStatsTest, DistanceValuesAreShortestPaths) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 0 -> 1 (1), 0 -> 2 (10), 1 -> 2 (2): the cheap route to 2 is via 1.
        auto graph = make_graph({0, 0, 1}, {1, 2, 2}, {1.0f, 10.0f, 2.0f}, 3);
        ASSERT_NE(graph, nullptr);

        const auto result = run(*graph, 0);
        const auto distances = read_distances(result);

        ASSERT_EQ(distances.size(), 3u);
        EXPECT_FLOAT_EQ(distances[0], 0.0f);
        EXPECT_FLOAT_EQ(distances[1], 1.0f);
        EXPECT_FLOAT_EQ(distances[2], 3.0f) << "1 + 2 beats the direct edge of 10";
    }

    TEST(SsspReportedStatsTest, UnreachableVerticesStayAtInfinity) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Vertex 3 has no incoming edge, so it is unreachable from 0.
        auto graph = make_graph({0, 1}, {1, 2}, {1.0f, 1.0f}, 4);
        ASSERT_NE(graph, nullptr);

        const auto result = run(*graph, 0);
        const auto distances = read_distances(result);

        ASSERT_EQ(distances.size(), 4u);
        EXPECT_TRUE(std::isinf(distances[3])) << "an unreachable vertex has no finite distance, got " << distances[3];
    }

    // Regression: this reported converged=true and meaningless distances.
    TEST(SsspReportedStatsTest, NegativeCycleIsDetectedRatherThanReportedAsSuccess) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 1 -> 2 (weight -3) and 2 -> 1 (weight 1) form a cycle of total -2.
        auto graph = make_graph({0, 1, 2}, {1, 2, 1}, {1.0f, -3.0f, 1.0f}, 3);
        ASSERT_NE(graph, nullptr);

        const auto result = run(*graph, 0);

        EXPECT_TRUE(result.negative_cycle) << "relaxing a negative cycle forever must be reported";
        EXPECT_FALSE(result.converged) << "a run whose distances are unbounded below did not converge";
    }

    TEST(SsspReportedStatsTest, NegativeEdgeWithoutACycleStillConverges) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A single negative edge 1 -> 2 of -5, but no cycle: the shortest paths exist.
        auto graph = make_graph({0, 1}, {1, 2}, {1.0f, -5.0f}, 3);
        ASSERT_NE(graph, nullptr);

        const auto result = run(*graph, 0);
        const auto distances = read_distances(result);

        EXPECT_TRUE(result.converged) << "a negative edge alone is legitimate; only a cycle is not";
        EXPECT_FALSE(result.negative_cycle);
        EXPECT_FLOAT_EQ(distances[1], 1.0f);
        EXPECT_FLOAT_EQ(distances[2], -4.0f) << "1 + (-5)";
    }

    TEST(SsspReportedStatsTest, SingleVertexGraphConvergesImmediately) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // No edges: the source is already settled.
        auto graph = make_graph({}, {}, {}, 1);
        ASSERT_NE(graph, nullptr);

        const auto result = run(*graph, 0);
        const auto distances = read_distances(result);

        ASSERT_EQ(distances.size(), 1u);
        EXPECT_FLOAT_EQ(distances[0], 0.0f);
        EXPECT_TRUE(result.converged);
        EXPECT_FALSE(result.negative_cycle);
    }

    TEST(SsspReportedStatsTest, TheCompatibleOverloadStillReturnsDistances) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // The older entry point returns the buffer directly for callers that do not
        // need the statistics; it must produce the same values.
        auto graph = make_graph({0, 1}, {1, 2}, {1.0f, 2.0f}, 3);
        ASSERT_NE(graph, nullptr);

        auto buffer = compute_distances<float>(*graph, 0, 1.0f);
        std::vector<float> distances(3);
        buffer.copy_to(distances.data(), distances.size());

        EXPECT_FLOAT_EQ(distances[0], 0.0f);
        EXPECT_FLOAT_EQ(distances[1], 1.0f);
        EXPECT_FLOAT_EQ(distances[2], 3.0f);
    }

    TEST(SsspReportedStatsTest, IterationCountIsBoundedByTheVertexCount) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Bellman-Ford needs at most n-1 rounds; a larger count would mean the loop
        // is not converging as designed.
        auto graph = make_graph({0, 1, 2}, {1, 2, 3}, {1.0f, 1.0f, 1.0f}, 4);
        ASSERT_NE(graph, nullptr);

        const auto result = run(*graph, 0);

        EXPECT_GT(result.iterations, 0);
        EXPECT_LE(result.iterations, result.num_vertices) << "a settled run cannot use more rounds than there are vertices";
    }

}  // namespace cuda::algo::sssp::test
