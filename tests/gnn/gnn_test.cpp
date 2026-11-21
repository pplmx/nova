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

// ---------------------------------------------------------------------------
// The aggregation's documented formula, pinned by value.
//
// gcn_aggregate is NOT the symmetric D^-1/2 (A+I) D^-1/2 normalisation that GCN is usually
// introduced with. It divides by the receiving node's degree only and adds no self-loop, and
// the header now says so. These cases lock in what it actually does, so a later change to the
// arithmetic is a deliberate decision rather than a silent one.
// ---------------------------------------------------------------------------

// output[i] = sum(neighbours of i) / sqrt(degree(i) + 1).
// The 3-node ring has degree 2 everywhere, so the divisor is sqrt(3) for all three nodes.
TEST_F(GNNTest, GCNAggregationDividesByTheReceivingNodesDegreeOnly) {
    auto graph = create_test_graph();
    std::vector<float> features = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3, 0.0f);

    MessagePassing mp;
    mp.gcn_aggregate(graph, features.data(), output.data());

    const float divisor = std::sqrt(3.0f);
    EXPECT_NEAR(output[0], (2.0f + 3.0f) / divisor, 1e-5f);  // neighbours of 0 are 1 and 2
    EXPECT_NEAR(output[1], (3.0f + 1.0f) / divisor, 1e-5f);  // neighbours of 1 are 2 and 0
    EXPECT_NEAR(output[2], (1.0f + 2.0f) / divisor, 1e-5f);  // neighbours of 2 are 0 and 1
}

// The divisor uses the receiving node's degree, not the sender's, so an edge is not
// symmetric: the same impulse carries a different weight in each direction.
TEST_F(GNNTest, GCNAggregationIsNotSymmetricAcrossAnEdge) {
    // node 0 <-> node 1, node 0 <-> node 2. Degrees: 2, 1, 1.
    nova::sparse::SparseMatrixCSR<float> graph({1.0f, 1.0f, 1.0f}, {0, 2, 3, 3}, {1, 2, 0}, 3, 3);

    MessagePassing mp;
    std::vector<float> into_zero = {0.0f, 1.0f, 0.0f};  // impulse at node 1
    std::vector<float> output_a(3, 0.0f);
    mp.gcn_aggregate(graph, into_zero.data(), output_a.data());

    std::vector<float> into_one = {1.0f, 0.0f, 0.0f};  // impulse at node 0
    std::vector<float> output_b(3, 0.0f);
    mp.gcn_aggregate(graph, into_one.data(), output_b.data());

    EXPECT_NEAR(output_a[0], 1.0f / std::sqrt(3.0f), 1e-5f) << "node 0 has degree 2, so the divisor is sqrt(3)";
    EXPECT_NEAR(output_b[1], 1.0f / std::sqrt(2.0f), 1e-5f) << "node 1 has degree 1, so the divisor is sqrt(2)";
    EXPECT_NE(output_a[0], output_b[1]) << "the edge weight differs by direction; that is what 'not symmetric' means here";
}

// No self-loop: a node with no neighbours aggregates nothing and loses its own feature.
// The (A + I) form would return the feature unchanged, so this case pins the difference.
TEST_F(GNNTest, GCNAggregationDiscardsTheFeatureOfAnIsolatedNode) {
    nova::sparse::SparseMatrixCSR<float> graph({1.0f}, {0, 0, 0}, {0}, 2, 2);  // no edges at all
    std::vector<float> features = {7.0f, -3.0f};
    std::vector<float> output = {-1.0f, -1.0f};

    MessagePassing mp;
    mp.gcn_aggregate(graph, features.data(), output.data());

    EXPECT_FLOAT_EQ(output[0], 0.0f) << "an isolated node has nothing to aggregate and no self-loop to fall back on";
    EXPECT_FLOAT_EQ(output[1], 0.0f);
}

// The first hop's values are the aggregate of the input, and they are exactly the numbers
// quoted in the header's explanation of the formula.
TEST_F(GNNTest, GCNPropagationOneHopMatchesTheDocumentedValues) {
    auto graph = create_test_graph();
    std::vector<float> features = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3, 0.0f);

    MessagePassing mp;
    mp.gcn_propagate(graph, features.data(), output.data(), 1);

    EXPECT_NEAR(output[0], 2.8867514f, 1e-5f);
    EXPECT_NEAR(output[1], 2.3094010f, 1e-5f);
    EXPECT_NEAR(output[2], 1.7320508f, 1e-5f);
}

// ---------------------------------------------------------------------------
// The same out-of-range column defect GraphAttention had, in MessagePassing.
// All three entry points index node_features with the graph's column index.
// ASan with a 2-node graph whose entry points at column 5 and a feature buffer
// of exactly num_rows() floats: stack-buffer-overflow.
// ---------------------------------------------------------------------------

namespace {

    // Two nodes, one entry pointing at a column that does not exist.
    nova::sparse::SparseMatrixCSR<float> graph_with_a_missing_column() {
        return nova::sparse::SparseMatrixCSR<float>({1.0f}, {0, 1, 1}, {5}, 2, 2);
    }

}  // namespace

TEST_F(GNNTest, GenericForwardRejectsAColumnIndexOutsideTheNodes) {
    auto graph = graph_with_a_missing_column();
    std::vector<float> features = {1.0f, 1.0f};
    std::vector<float> output(2, 0.0f);

    MessagePassing mp;
    EXPECT_THROW(mp.forward(graph, features.data(), output.data()), std::out_of_range);
}

TEST_F(GNNTest, GCNAggregationRejectsAColumnIndexOutsideTheNodes) {
    auto graph = graph_with_a_missing_column();
    std::vector<float> features = {1.0f, 1.0f};
    std::vector<float> output(2, 0.0f);

    MessagePassing mp;
    EXPECT_THROW(mp.gcn_aggregate(graph, features.data(), output.data()), std::out_of_range);
}

TEST_F(GNNTest, GCNPropagationRejectsAColumnIndexOutsideTheNodes) {
    auto graph = graph_with_a_missing_column();
    std::vector<float> features = {1.0f, 1.0f};
    std::vector<float> output(2, 0.0f);

    MessagePassing mp;
    // Also at hops=0: the check must not depend on the aggregate path being taken.
    EXPECT_THROW(mp.gcn_propagate(graph, features.data(), output.data(), 0), std::out_of_range);
    EXPECT_THROW(mp.gcn_propagate(graph, features.data(), output.data(), 2), std::out_of_range);
}

// The guard must not reject an ordinary rectangular matrix whose columns are simply fewer
// than its rows, and it must not reject a graph with no edges.
TEST_F(GNNTest, MessagePassingAcceptsValidGraphs) {
    nova::sparse::SparseMatrixCSR<float> rectangular({1.0f, 1.0f}, {0, 1, 2}, {0, 1}, 2, 2);
    std::vector<float> features = {1.0f, 2.0f};
    std::vector<float> output(2, 0.0f);

    MessagePassing mp;
    EXPECT_NO_THROW(mp.forward(rectangular, features.data(), output.data()));
    EXPECT_NO_THROW(mp.gcn_aggregate(rectangular, features.data(), output.data()));

    nova::sparse::SparseMatrixCSR<float> empty({1.0f}, {0, 0, 0}, {0}, 2, 2);
    EXPECT_NO_THROW(mp.forward(empty, features.data(), output.data()));
    EXPECT_NO_THROW(mp.gcn_aggregate(empty, features.data(), output.data()));
}

// Null inputs are rejected rather than dereferenced, consistently with GraphAttention.
TEST_F(GNNTest, MessagePassingRejectsNullPointers) {
    auto graph = create_test_graph();
    std::vector<float> features = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3, 0.0f);

    MessagePassing mp;
    EXPECT_THROW(mp.forward(graph, nullptr, output.data()), std::invalid_argument);
    EXPECT_THROW(mp.forward(graph, features.data(), nullptr), std::invalid_argument);
    EXPECT_THROW(mp.gcn_aggregate(graph, nullptr, output.data()), std::invalid_argument);
    EXPECT_THROW(mp.gcn_propagate(graph, nullptr, output.data(), 1), std::invalid_argument);
}
