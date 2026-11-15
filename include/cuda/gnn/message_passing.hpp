/**
 * @file message_passing.hpp
 * @brief GNN message passing implementations
 * @defgroup message_passing Message Passing
 * @ingroup gnn
 *
 * Provides message passing primitives for Graph Neural Networks:
 * - Generic message/aggregate functions
 * - GCN aggregation and propagation
 * - GraphSAGE-style operations
 *
 * Example usage:
 * @code
 * MessagePassing mp;
 * mp.gcn_aggregate(graph, node_features, output);
 * @endcode
 *
 * @see sampling.hpp For graph sampling
 * @see attention.hpp For attention mechanisms
 */

#ifndef NOVA_CUDA_GNN_MESSAGE_PASSING_HPP
#define NOVA_CUDA_GNN_MESSAGE_PASSING_HPP

#include <cuda/sparse/sparse_matrix.hpp>
#include <cmath>
#include <functional>
#include <vector>

namespace nova {
namespace gnn {

/** @brief Message function: src_feat -> message */
using MessageFunction = std::function<float(float src_feat, float dst_feat)>;

/** @brief Aggregation function: messages -> aggregated value */
using AggregateFunction = std::function<float(const std::vector<float>& messages)>;

/**
 * @brief Message passing GNN layer
 * @class MessagePassing
 * @ingroup message_passing
 */
class MessagePassing {
public:
    /** @brief Default constructor */
    MessagePassing() = default;

    /** @brief Set custom message function */
    void set_message_function(MessageFunction fn) { message_fn_ = std::move(fn); }

    /** @brief Set custom aggregate function */
    void set_aggregate_function(AggregateFunction fn) { aggregate_fn_ = std::move(fn); }

    /**
     * @brief Generic message passing forward
     * @param graph Adjacency matrix
     * @param node_features Input node features
     * @param[out] output Aggregated features
     */
    void forward(const sparse::SparseMatrixCSR<float>& graph,
                 const float* node_features,
                 float* output);

    /**
     * @brief GCN-style aggregation
     * @param graph Adjacency matrix
     * @param node_features Input node features
     * @param[out] output Aggregated features
     */
    void gcn_aggregate(const sparse::SparseMatrixCSR<float>& graph,
                       const float* node_features,
                       float* output);

    /**
     * @brief GCN-style propagation
     * @param graph Adjacency matrix
     * @param node_features Input node features
     * @param[out] output Output features
     */
    void gcn_propagate(const sparse::SparseMatrixCSR<float>& graph,
                       const float* node_features,
                       float* output,
                       int hops = 1);

private:
    MessageFunction message_fn_;
    AggregateFunction aggregate_fn_;
};

inline void MessagePassing::forward(const sparse::SparseMatrixCSR<float>& graph,
                              const float* node_features,
                              float* output) {
    int num_nodes = graph.num_rows();

    for (int i = 0; i < num_nodes; ++i) {
        std::vector<float> messages;

        for (int idx = graph.row_offsets()[i]; idx < graph.row_offsets()[i + 1]; ++idx) {
            int neighbor = graph.col_indices()[idx];
            float msg;

            if (message_fn_) {
                msg = message_fn_(node_features[neighbor], node_features[i]);
            } else {
                msg = node_features[neighbor];
            }

            messages.push_back(msg);
        }

        if (aggregate_fn_ && !messages.empty()) {
            output[i] = aggregate_fn_(messages);
        } else if (!messages.empty()) {
            float sum = 0.0f;
            for (float m : messages) sum += m;
            output[i] = sum;
        } else {
            output[i] = 0.0f;
        }
    }
}

inline void MessagePassing::gcn_aggregate(const sparse::SparseMatrixCSR<float>& graph,
                                    const float* node_features,
                                    float* output) {
    int num_nodes = graph.num_rows();

    for (int i = 0; i < num_nodes; ++i) {
        float sum = 0.0f;
        int degree = graph.row_offsets()[i + 1] - graph.row_offsets()[i];

        for (int idx = graph.row_offsets()[i]; idx < graph.row_offsets()[i + 1]; ++idx) {
            int neighbor = graph.col_indices()[idx];
            sum += node_features[neighbor];
        }

        output[i] = (degree > 0) ? (sum / std::sqrt(static_cast<float>(degree + 1))) : 0.0f;
    }
}

inline void MessagePassing::gcn_propagate(const sparse::SparseMatrixCSR<float>& graph,
                                    const float* node_features,
                                    float* output,
                                    int hops) {
    if (hops <= 1) {
        gcn_aggregate(graph, node_features, output);
        return;
    }

    std::vector<float> temp(node_features, node_features + graph.num_rows());
    std::vector<float> next(graph.num_rows());

    gcn_aggregate(graph, temp.data(), next.data());

    for (int h = 2; h <= hops; ++h) {
        gcn_aggregate(graph, next.data(), temp.data());
        std::copy(temp.begin(), temp.end(), next.begin());
    }

    std::copy(next.begin(), next.end(), output);
}

} // namespace gnn
} // namespace nova

#endif // NOVA_CUDA_GNN_MESSAGE_PASSING_HPP
