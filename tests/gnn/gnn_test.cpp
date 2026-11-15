#include "message_passing.hpp"
#include "attention.hpp"
#include "sampling.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace nova::gnn;

class GNNTest : public ::testing::Test {
protected:
    nova::sparse::SparseMatrixCSR<float> create_test_graph() {
        std::vector<float> values = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<int> row_offsets = {0, 2, 4, 6};
        std::vector<int> col_indices = {1, 2, 0, 2, 0, 1};

        return nova::sparse::SparseMatrixCSR<float>(std::move(values), std::move(row_offsets),
                                               std::move(col_indices), 3, 3);
    }
};

TEST_F(GNNTest, MessagePassingForward) {
    auto graph = create_test_graph();
    std::vector<float> features = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3, 0.0f);

    MessagePassing mp;
    mp.forward(graph, features.data(), output.data());

    EXPECT_GT(output[0], 0.0f);
    EXPECT_GT(output[1], 0.0f);
    EXPECT_GT(output[2], 0.0f);
}

TEST_F(GNNTest, GCNAggregation) {
    auto graph = create_test_graph();
    std::vector<float> features = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3, 0.0f);

    MessagePassing mp;
    mp.gcn_aggregate(graph, features.data(), output.data());

    EXPECT_GT(output[0], 0.0f);
    EXPECT_GT(output[1], 0.0f);
    EXPECT_GT(output[2], 0.0f);
}

TEST_F(GNNTest, GCNPropagationMultipleHops) {
    auto graph = create_test_graph();
    std::vector<float> features = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3, 0.0f);

    MessagePassing mp;
    mp.gcn_propagate(graph, features.data(), output.data(), 2);

    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(output[i], 0.0f);
    }
}

TEST_F(GNNTest, GraphAttention) {
    auto graph = create_test_graph();
    std::vector<float> features = {1.0f, 2.0f, 1.5f, 2.5f, 1.0f, 1.0f, 2.0f, 1.5f, 2.5f};
    std::vector<float> output(6, 0.0f);

    GraphAttention gat(3, 2, 1);
    gat.forward(graph, features.data(), output.data());

    for (int i = 0; i < 6; ++i) {
        EXPECT_GE(output[i], -1000.0f);
        EXPECT_LE(output[i], 1000.0f);
    }
}

TEST_F(GNNTest, NeighborSampling) {
    auto graph = create_test_graph();
    GraphSampler sampler(42);

    std::vector<int> seed_nodes = {0};
    auto result = sampler.sample_neighbors(graph, seed_nodes, 2);

    EXPECT_EQ(result.node_ids.size(), 1);
    EXPECT_EQ(result.node_ids[0], 0);
    EXPECT_LE(result.neighbor_counts[0], 2);
}

TEST_F(GNNTest, KHopAggregation) {
    auto graph = create_test_graph();
    GraphSampler sampler(42);

    std::vector<int> seed_nodes = {0};
    auto result = sampler.k_hop_aggregation(graph, seed_nodes, 1);

    EXPECT_GE(result.size(), 1);
    EXPECT_NE(std::find(result.begin(), result.end(), 0), result.end());
    EXPECT_NE(std::find(result.begin(), result.end(), 1), result.end());
    EXPECT_NE(std::find(result.begin(), result.end(), 2), result.end());
}
