#include "message_passing.hpp"
#include "attention.hpp"
#include "sampling.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace nova::gnn;

class GNNTest : public ::testing::Test {
protected:
    nova::sparse::SparseMatrixCSR<float> create_test_graph() {
        std::vector<float> values = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<int> row_offsets = {0, 2, 4, 6};
        std::vector<int> col_indices = {1, 2, 0, 2, 0, 1};

        return nova::sparse::SparseMatrixCSR<float>(std::move(values), std::move(row_offsets), std::move(col_indices), 3, 3);
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

// ---------------------------------------------------------------------------
// Regressions for the bounds and argument defects in attention.hpp,
// sampling.hpp and message_passing.hpp. Every case below was verified against
// the unfixed header first, where it crashed, threw the wrong exception, or
// produced the wrong numbers.
// ---------------------------------------------------------------------------

namespace {

    // A 2x3 adjacency whose entries point at column 2, which does not exist as a
    // node when the layer sizes everything by num_rows().
    nova::sparse::SparseMatrixCSR<float> wide_graph() {
        return nova::sparse::SparseMatrixCSR<float>({1.0f, 1.0f}, {0, 1, 2}, {2, 2}, 2, 3);
    }

}  // namespace

// Regression: attention_scores_ is sized num_nodes * num_nodes, but the write indexed
// it with the graph's column index. ASan on a 2x3 graph: heap-buffer-overflow WRITE in
// GraphAttention::forward.
TEST_F(GNNTest, AttentionRejectsAColumnIndexOutsideTheNodes) {
    auto graph = wide_graph();
    std::vector<float> features(3, 1.0f);
    std::vector<float> output(2, 0.0f);

    GraphAttention gat(1, 1, 1);
    EXPECT_THROW(gat.forward(graph, features.data(), output.data()), std::out_of_range);
}

// A negative column index is the same defect from the other end, and it must not be
// silently treated as an unsigned offset either.
TEST_F(GNNTest, AttentionRejectsANegativeColumnIndex) {
    nova::sparse::SparseMatrixCSR<float> graph({1.0f}, {0, 1, 1}, {-1}, 2, 2);
    std::vector<float> features(4, 1.0f);
    std::vector<float> output(2, 0.0f);

    GraphAttention gat(1, 1, 1);
    EXPECT_THROW(gat.forward(graph, features.data(), output.data()), std::out_of_range);
}

// The column check must not reject a graph that is merely rectangular: columns may be
// fewer than rows, which is a perfectly ordinary CSR matrix.
TEST_F(GNNTest, AttentionAcceptsAFewerColumnsThanNodes) {
    nova::sparse::SparseMatrixCSR<float> graph({1.0f, 1.0f}, {0, 1, 2}, {0, 1}, 2, 2);
    std::vector<float> features(2, 1.0f);
    std::vector<float> output(2, 0.0f);

    GraphAttention gat(1, 1, 1);
    EXPECT_NO_THROW(gat.forward(graph, features.data(), output.data()));
}

// Regression: num_heads was stored and read by nothing. Measured: 1, 2 and 4 produced
// bit-identical output, so a caller could believe it had multi-head attention.
TEST_F(GNNTest, AttentionRejectsMultiHeadRatherThanIgnoringIt) {
    EXPECT_THROW(GraphAttention(3, 2, 2), std::invalid_argument);
    EXPECT_THROW(GraphAttention(3, 2, 4), std::invalid_argument);
    EXPECT_NO_THROW(GraphAttention(3, 2, 1));
}

// Regression: set_attention_weights stored a vector that forward() never read. Measured:
// calling it changed nothing about the output.
TEST_F(GNNTest, AttentionRejectsAttentionWeightsRatherThanIgnoringThem) {
    GraphAttention gat(3, 2, 1);
    EXPECT_THROW(gat.set_attention_weights({5.0f, 5.0f, 5.0f}), std::logic_error);
}

// The existing GraphAttention case only asserted each output was within +/-1000, which
// any finite garbage satisfies. These are the values the implementation actually
// produces, derived independently from the definition:
//   score(src, dst) = tanh(dot(feat[src], feat[dst]) / sqrt(F))
//   out[dst][o]     = sum over neighbours src of normalised(score) * feat[src][o % F]
TEST_F(GNNTest, AttentionProducesTheDefinedWeighting) {
    auto graph = create_test_graph();
    std::vector<float> features = {1.0f, 2.0f, 1.5f, 2.5f, 1.0f, 1.0f, 2.0f, 1.5f, 2.5f};
    std::vector<float> output(6, 0.0f);

    GraphAttention gat(3, 2, 1);
    gat.forward(graph, features.data(), output.data());

    EXPECT_NEAR(output[0], 2.2497654f, 1e-5f);
    EXPECT_NEAR(output[1], 1.2502347f, 1e-5f);
    EXPECT_NEAR(output[2], 1.5004746f, 1e-5f);
    EXPECT_NEAR(output[3], 1.7497628f, 1e-5f);
    EXPECT_NEAR(output[4], 1.7500078f, 1e-5f);
    EXPECT_NEAR(output[5], 1.4999949f, 1e-5f);
}

// Regression: a seed outside the graph indexed row_offsets() directly. ASan on a
// 3-node graph with seed 7: heap-buffer-overflow in sample_neighbors.
TEST_F(GNNTest, NeighborSamplingRejectsASeedOutsideTheGraph) {
    auto graph = create_test_graph();
    GraphSampler sampler(42);

    EXPECT_THROW(sampler.sample_neighbors(graph, {7}, 2), std::out_of_range);
    EXPECT_THROW(sampler.sample_neighbors(graph, {0, -1}, 2), std::out_of_range);
}

// Regression: a negative budget reached all_neighbors.begin() + num_samples, which
// threw std::length_error from inside vector's constructor.
TEST_F(GNNTest, NeighborSamplingRejectsANegativeBudget) {
    auto graph = create_test_graph();
    GraphSampler sampler(42);

    EXPECT_THROW(sampler.sample_neighbors(graph, {0}, -1), std::invalid_argument);
}

// A budget larger than every degree is satisfied by returning all neighbours, and it
// already did; this pins that the new guard did not narrow it.
TEST_F(GNNTest, NeighborSamplingAcceptsABudgetLargerThanEveryDegree) {
    auto graph = create_test_graph();
    GraphSampler sampler(42);

    auto result = sampler.sample_neighbors(graph, {0}, 100);
    ASSERT_EQ(result.node_ids.size(), 1u);
    EXPECT_EQ(result.neighbor_counts[0], 2);
}

// Regression: the same out-of-range read, in k_hop_aggregation.
TEST_F(GNNTest, KHopAggregationRejectsASeedOutsideTheGraph) {
    auto graph = create_test_graph();
    GraphSampler sampler(42);

    EXPECT_THROW(sampler.k_hop_aggregation(graph, {9}, 1), std::out_of_range);
    EXPECT_THROW(sampler.k_hop_aggregation(graph, {-2}, 1), std::out_of_range);
}

// Regression: k < 0 returned the seeds unchanged, which reads as "nothing reachable".
TEST_F(GNNTest, KHopAggregationRejectsANegativeHopCount) {
    auto graph = create_test_graph();
    GraphSampler sampler(42);

    EXPECT_THROW(sampler.k_hop_aggregation(graph, {0}, -1), std::invalid_argument);
}

// k=0 is meaningful: the neighbourhood of a node at zero hops is the node itself. It
// must keep returning exactly the seeds now that 0 is no longer lumped in with negative.
TEST_F(GNNTest, KHopAggregationAtZeroHopsReturnsTheSeedsUnchanged) {
    auto graph = create_test_graph();
    GraphSampler sampler(42);

    auto result = sampler.k_hop_aggregation(graph, {0, 2}, 0);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[1], 2);
}

// Regression: gcn_propagate's `hops <= 1` branch sent 0 and every negative value
// through one aggregation. Measured: hops 0, 1 and -5 produced identical output.
TEST_F(GNNTest, PropagationAtZeroHopsCopiesTheInputRatherThanAggregating) {
    auto graph = create_test_graph();
    std::vector<float> features = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3, -99.0f);

    MessagePassing mp;
    mp.gcn_propagate(graph, features.data(), output.data(), 0);

    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(output[i], features[i]) << "hops=0 must not aggregate anything, at node " << i;
    }
}

TEST_F(GNNTest, PropagationRejectsANegativeHopCount) {
    auto graph = create_test_graph();
    std::vector<float> features = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3, 0.0f);

    MessagePassing mp;
    EXPECT_THROW(mp.gcn_propagate(graph, features.data(), output.data(), -5), std::invalid_argument);
}

// One hop must still aggregate, so the hops==0 branch above cannot be satisfied by
// copying unconditionally.
TEST_F(GNNTest, PropagationAtOneHopActuallyAggregates) {
    auto graph = create_test_graph();
    std::vector<float> features = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3, 0.0f);

    MessagePassing mp;
    mp.gcn_propagate(graph, features.data(), output.data(), 1);

    // Node 0's neighbours are 1 and 2: (2 + 3) / sqrt(2 + 1).
    const float expected = 5.0f / std::sqrt(3.0f);
    EXPECT_FLOAT_EQ(output[0], expected);
    // And it differs from the input, which is what hops=0 must not do.
    EXPECT_NE(output[0], features[0]);
}
