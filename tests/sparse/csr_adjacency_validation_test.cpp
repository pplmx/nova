/**
 * @file csr_adjacency_validation_test.cpp
 * @brief create_csr_from_adjacency must reject a neighbour index outside the vertex range
 *
 * The builder validated the weights shape and the row lengths, then copied neighbours straight
 * into graph->columns with no bound:
 *
 *     graph->columns[offset + i] = adjacency[v][i];
 *
 * A neighbour outside [0, num_vertices) therefore produced a graph whose column indices are
 * invalid. validate_csr reports that after the fact, but the constructor had already handed
 * back a malformed result. create_csr_from_edges validates the same thing for its edge list;
 * this matches it now.
 */

#include <cuda/graph/csr_graph.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace csr_adjacency_validation_test {

    TEST(CsrAdjacencyValidationTest, ANeighbourPastTheVertexCountIsRejected) {
        // Three vertices, but vertex 0 points at 5.
        std::vector<std::vector<int>> adjacency(3);
        adjacency[0] = {5};
        adjacency[1] = {};
        adjacency[2] = {};

        auto call = [&] { return cuda::graph::create_csr_from_adjacency(adjacency); };
        EXPECT_THROW(call(), std::out_of_range) << "a column index outside the vertex range makes an invalid CSR structure";
    }

    TEST(CsrAdjacencyValidationTest, ANegativeNeighbourIsRejected) {
        std::vector<std::vector<int>> adjacency(2);
        adjacency[0] = {-1};
        adjacency[1] = {};

        auto call = [&] { return cuda::graph::create_csr_from_adjacency(adjacency); };
        EXPECT_THROW(call(), std::out_of_range);
    }

    TEST(CsrAdjacencyValidationTest, AWellFormedAdjacencyListStillBuildsAValidGraph) {
        // A three-vertex cycle. The guard must not reject legal input.
        std::vector<std::vector<int>> adjacency(3);
        adjacency[0] = {1};
        adjacency[1] = {2};
        adjacency[2] = {0};

        auto graph = cuda::graph::create_csr_from_adjacency(adjacency);
        ASSERT_NE(graph, nullptr);
        EXPECT_EQ(graph->vertices(), 3);
        EXPECT_EQ(graph->edges(), 3);
        EXPECT_TRUE(cuda::graph::validate_csr(*graph));
        EXPECT_EQ(graph->degree(0), 1);
        EXPECT_EQ(graph->degree(1), 1);
        EXPECT_EQ(graph->degree(2), 1);
    }

    TEST(CsrAdjacencyValidationTest, TheMismatchedWeightsCheckStillFires) {
        // The check that was already there, kept next to the new one so a change to either is
        // visible alongside the other.
        std::vector<std::vector<int>> adjacency(2);
        adjacency[0] = {1};
        adjacency[1] = {0};
        std::vector<std::vector<float>> wrong_weights(3);  // three rows for two vertices
        wrong_weights[0] = {1.0F};
        wrong_weights[1] = {1.0F};
        wrong_weights[2] = {1.0F};

        auto call = [&] { return cuda::graph::create_csr_from_adjacency(adjacency, wrong_weights); };
        EXPECT_THROW(call(), std::invalid_argument);
    }

}  // namespace csr_adjacency_validation_test
