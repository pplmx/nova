/**
 * @file attention.hpp
 * @brief Graph Attention Networks
 * @defgroup gnn_attention Graph Attention
 * @ingroup gnn
 *
 * Provides Graph Attention Network (GAT) layer implementation.
 * Uses multi-head attention for node feature aggregation.
 *
 * @note Time complexity: O(V * k^2 * F) where k=neighbors, F=features
 * @see message_passing.hpp For base message passing
 */

#ifndef NOVA_CUDA_GNN_ATTENTION_HPP
#define NOVA_CUDA_GNN_ATTENTION_HPP

#include <cuda/gnn/message_passing.hpp>
#include <cmath>
#include <vector>

namespace nova {
namespace gnn {

/**
 * @brief Graph Attention Network layer
 * @class GraphAttention
 * @ingroup gnn_attention
 */
class GraphAttention {
public:
    /**
     * @brief Construct GAT layer
     * @param in_features Input feature dimension
     * @param out_features Output feature dimension
     * @param num_heads Number of attention heads
     */
    GraphAttention(int in_features, int out_features, int num_heads = 1)
        : in_features_(in_features)
        , out_features_(out_features)
        , num_heads_(num_heads) {}

    /**
     * @brief Forward pass
     * @param graph Adjacency matrix
     * @param node_features Input node features
     * @param[out] output Output node features
     */
    void forward(const sparse::SparseMatrixCSR<float>& graph,
                 const float* node_features,
                 float* output);

    /**
     * @brief Set attention weight parameters
     * @param weights Attention weight vector
     */
    void set_attention_weights(const std::vector<float>& weights);

private:
    int in_features_;
    int out_features_;
    int num_heads_;
    std::vector<float> weights_;
    std::vector<float> attention_scores_;

    float compute_attention(int src, int dst, const float* features);
};

inline float GraphAttention::compute_attention(int src, int dst, const float* features) {
    float dot_product = 0.0f;

    for (int i = 0; i < in_features_; ++i) {
        dot_product += features[src * in_features_ + i] *
                       features[dst * in_features_ + i];
    }

    return std::tanh(dot_product / std::sqrt(static_cast<float>(in_features_)));
}

inline void GraphAttention::forward(const sparse::SparseMatrixCSR<float>& graph,
                              const float* node_features,
                              float* output) {
    int num_nodes = graph.num_rows();

    attention_scores_.resize(num_nodes * num_nodes, 0.0f);

    for (int i = 0; i < num_nodes; ++i) {
        for (int idx = graph.row_offsets()[i]; idx < graph.row_offsets()[i + 1]; ++idx) {
            int neighbor = graph.col_indices()[idx];
            attention_scores_[i * num_nodes + neighbor] = compute_attention(neighbor, i, node_features);
        }
    }

    for (int i = 0; i < num_nodes; ++i) {
        float sum = 0.0f;
        int degree = graph.row_offsets()[i + 1] - graph.row_offsets()[i];

        for (int idx = graph.row_offsets()[i]; idx < graph.row_offsets()[i + 1]; ++idx) {
            int neighbor = graph.col_indices()[idx];
            sum += attention_scores_[i * num_nodes + neighbor];
        }

        float normalizer = (sum > 0.0f) ? sum : 1.0f;

        for (int o = 0; o < out_features_; ++o) {
            float weighted_sum = 0.0f;
            for (int idx = graph.row_offsets()[i]; idx < graph.row_offsets()[i + 1]; ++idx) {
                int neighbor = graph.col_indices()[idx];
                float attn_weight = attention_scores_[i * num_nodes + neighbor] / normalizer;
                weighted_sum += attn_weight * node_features[neighbor * in_features_ + o % in_features_];
            }
            output[i * out_features_ + o] = weighted_sum;
        }
    }
}

inline void GraphAttention::set_attention_weights(const std::vector<float>& weights) {
    weights_ = weights;
}

} // namespace gnn
} // namespace nova

#endif // NOVA_CUDA_GNN_ATTENTION_HPP
