/**
 * @file graph_container_state_test.cpp
 * @brief Contracts for the graph result containers' lifetime states
 *
 * include/cuda/graph/bfs.h and include/cuda/graph/csr_graph.h own raw host and
 * device buffers tracked by separate count fields. Round 50 fixed exactly this
 * mismatch in PageRankResult; the same defect existed in both of these.
 *
 * - BFSResult::clear() released distances and visited and nulled both, but left
 *   num_vertices and visited_count describing the buffers that no longer existed.
 *   Measured: after clear() on an 8-vertex result, num_vertices was still 8 while
 *   distances was (nil) — a later upload()/download() would have copied 8 elements
 *   through null pointers.
 *
 * - BFSResult::init_source wrote distances[source] and visited[source] with no
 *   state or range check, and measured "dumped core" on a moved-from result.
 *
 * - CSRGraph::clear() likewise left num_vertices and num_edges set, so
 *   memory_usage() reported 64 bytes for a 3-vertex graph whose buffers were
 *   already gone. memory_usage() also reported 2*sizeof(int) for a graph that never
 *   had any buffers at all, because the (num_vertices + 1) term is non-zero at zero
 *   vertices.
 */

#include <cuda/graph/bfs.h>
#include <cuda/graph/csr_graph.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cuda::graph::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Runs `call` and returns the exception message, or an empty string.
        template <typename Callable>
        std::string message_from(Callable&& call) {
            try {
                call();
            } catch (const std::exception& error) {
                return error.what();
            }
            return "";
        }

    }  // namespace

    // ------------------------------------------------------------ BFSResult ----

    // Regression: num_vertices survived clear() while the buffers did not.
    TEST(BFSResultStateTest, ClearResetsTheCountsAlongWithTheBuffers) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        BFSResult result(8);
        result.visited_count = 5;
        result.clear();

        EXPECT_EQ(result.distances, nullptr);
        EXPECT_EQ(result.visited, nullptr);
        EXPECT_EQ(result.num_vertices, 0) << "the count must not outlive the buffer it describes";
        EXPECT_EQ(result.visited_count, 0);
    }

    // Regression: this wrote through null and aborted the process.
    TEST(BFSResultStateTest, InitSourceRejectsAReleasedResult) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        BFSResult original(4);
        BFSResult owner(std::move(original));

        const std::string message = message_from([&] { original.init_source(0); });

        EXPECT_FALSE(message.empty()) << "a moved-from result owns no buffers to initialise";
        EXPECT_NE(message.find("no vertices"), std::string::npos) << message;
    }

    // Regression: source indexed both arrays without a range check.
    TEST(BFSResultStateTest, InitSourceRejectsAnOutOfRangeVertex) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        BFSResult result(4);

        const std::string above = message_from([&] { result.init_source(4); });
        EXPECT_FALSE(above.empty()) << "index 4 is past a 4-vertex array";
        EXPECT_NE(above.find("4"), std::string::npos) << above;

        const std::string below = message_from([&] { result.init_source(-1); });
        EXPECT_FALSE(below.empty()) << "a negative index writes before the array";
    }

    TEST(BFSResultStateTest, InitSourceMarksOnlyTheSource) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        BFSResult result(5);
        result.init_source(2);

        for (int v = 0; v < 5; ++v) {
            if (v == 2) {
                EXPECT_EQ(result.distances[v], 0) << "the source is at distance zero";
                EXPECT_TRUE(result.visited[v]);
            } else {
                EXPECT_EQ(result.distances[v], -1) << "vertex " << v << " is unreached";
                EXPECT_FALSE(result.visited[v]);
            }
        }
    }

    TEST(BFSResultStateTest, InitSourceAcceptsTheLastVertex) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        BFSResult result(3);
        // num_vertices - 1 is the largest legal index; an off-by-one would make the
        // last vertex unusable as a source.
        EXPECT_NO_THROW(result.init_source(2));
        EXPECT_EQ(result.distances[2], 0);
    }

    TEST(BFSResultStateTest, EmptyResultRejectsInitSource) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        BFSResult empty(0);
        EXPECT_THROW(empty.init_source(0), std::invalid_argument);
    }

    // ------------------------------------------------------------ CSRGraph ----

    // Regression: the counts survived clear() and memory_usage() reported freed memory.
    TEST(CSRGraphStateTest, ClearResetsCountsAndMemoryUsage) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<int> sources = {0, 1};
        const std::vector<int> targets = {1, 2};
        const std::vector<float> weights = {1.0f, 2.0f};

        auto graph = create_csr_from_edges(sources.data(), targets.data(), weights.data(), 3, 2);
        ASSERT_NE(graph, nullptr);
        EXPECT_GT(graph->memory_usage(), 0u);

        graph->clear();

        EXPECT_EQ(graph->row_offsets, nullptr);
        EXPECT_EQ(graph->num_vertices, 0);
        EXPECT_EQ(graph->num_edges, 0);
        EXPECT_EQ(graph->memory_usage(), 0u) << "a cleared graph holds no memory, so it must not claim any";
    }

    // Regression: this reported 2 * sizeof(int) for a graph with no buffers.
    TEST(CSRGraphStateTest, NeverAllocatedGraphReportsNoMemory) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        CSRGraph graph;
        EXPECT_EQ(graph.row_offsets, nullptr);
        EXPECT_EQ(graph.memory_usage(), 0u) << "the (num_vertices + 1) term is non-zero even at zero vertices";
    }

    TEST(CSRGraphStateTest, BuiltGraphHasTheRequestedShape) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<int> sources = {0, 0, 1};
        const std::vector<int> targets = {1, 2, 2};
        const std::vector<float> weights = {1.5f, 2.5f, 3.5f};

        auto graph = create_csr_from_edges(sources.data(), targets.data(), weights.data(), 3, 3);
        ASSERT_NE(graph, nullptr);

        EXPECT_EQ(graph->num_vertices, 3);
        EXPECT_EQ(graph->num_edges, 3);
        ASSERT_NE(graph->row_offsets, nullptr);
        // row_offsets has num_vertices + 1 entries and its last value is the edge count.
        EXPECT_EQ(graph->row_offsets[0], 0);
        EXPECT_EQ(graph->row_offsets[3], 3);
    }

    TEST(CSRGraphStateTest, MemoryUsageCountsHostAndDeviceCopies) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<int> sources = {0};
        const std::vector<int> targets = {1};
        const std::vector<float> weights = {1.0f};

        auto graph = create_csr_from_edges(sources.data(), targets.data(), weights.data(), 2, 1);
        ASSERT_NE(graph, nullptr);

        // (2 + 1) * 4 bytes of offsets + 1 * (4 + 4) bytes of edges, doubled for the
        // device copy.
        const size_t expected = 2 * ((2u + 1u) * sizeof(int) + 1u * (sizeof(int) + sizeof(float)));
        EXPECT_EQ(graph->memory_usage(), expected);
    }

}  // namespace cuda::graph::test
