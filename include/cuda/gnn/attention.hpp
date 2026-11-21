/**
 * @file attention.hpp
 * @brief Graph Attention Networks
 * @defgroup gnn_attention Graph Attention
 * @ingroup gnn
 *
 * Provides a Graph Attention Network (GAT) layer.
 * Aggregates neighbours with additive attention over their feature vectors.
 *
 * @note Single-head only. The layer reads no learned attention weights; see the class
 *       documentation for what that means for num_heads and set_attention_weights().
 * @note Time complexity: O(V * k * F) where k=neighbors, F=features
 * @see message_passing.hpp For base message passing
 */

#ifndef NOVA_CUDA_GNN_ATTENTION_HPP
#define NOVA_CUDA_GNN_ATTENTION_HPP

#include <cuda/gnn/message_passing.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace nova {
    namespace gnn {

        /**
         * @brief Graph Attention Network layer
         * @class GraphAttention
         * @ingroup gnn_attention
         *
         * @warning This class is a single-head, weight-free attention layer. The
         * constructor's @p num_heads argument and set_attention_weights() are accepted and
         * stored but read by nothing: measured, num_heads 1, 2 and 4 produce bit-identical
         * output, and calling set_attention_weights({5, 5, 5}) changes nothing. Both are
         * rejected now rather than silently ignored, so a caller cannot believe it configured
         * multi-head attention when it did not.
         *
         * The scoring here is a dot product between neighbouring features, so there is no
         * learned weight matrix to set. A real multi-head GAT needs per-head projection
         * weights and a concatenated (or averaged) output; when that is implemented the
         * restrictions below come off.
         */
        class GraphAttention {
        public:
            /**
             * @brief Construct GAT layer
             * @param in_features Input feature dimension
             * @param out_features Output feature dimension
             * @param num_heads Number of attention heads; must be 1, see the class warning
             */
            GraphAttention(int in_features, int out_features, int num_heads = 1)
                : in_features_(in_features),
                  out_features_(out_features),
                  num_heads_(num_heads) {
                if (in_features <= 0 || out_features <= 0) {
                    throw std::invalid_argument("GraphAttention: in_features and out_features must be positive, got " + std::to_string(in_features) + " and "
                                                + std::to_string(out_features));
                }
                if (num_heads != 1) {
                    throw std::invalid_argument("GraphAttention: num_heads=" + std::to_string(num_heads) + " is not supported. This layer computes single-head attention and ignored the argument; "
                                                                                                                                     "multi-head attention would need per-head projection weights, which are not implemented. Pass 1.");
                }
            }

            /**
             * @brief Forward pass
             *
             * @pre `node_features` holds `graph.num_rows() * in_features` floats and
             *      `output` holds `graph.num_rows() * out_features` floats. There is no
             *      length parameter to check against, so this cannot be validated here: a
             *      buffer shorter than the layer's `in_features` is read out of bounds
             *      (measured under ASan with in_features=2 and a three-float buffer, at
             *      compute_attention). The column check below covers the other half of the
             *      same problem, where the graph itself names a node that does not exist.
             *
             * @param graph Adjacency matrix; column indices must lie in [0, num_rows()).
             *              A negative index, or one at or past num_rows(), throws
             *              std::out_of_range before anything is written.
             * @param node_features Input node features, num_rows() * in_features elements
             * @param[out] output Output node features, num_rows() * out_features elements
             * @throws std::invalid_argument if a pointer is null
             * @throws std::out_of_range if the graph references a column outside the nodes
             */
            void forward(const sparse::SparseMatrixCSR<float>& graph, const float* node_features, float* output);

            /**
             * @brief Set attention weight parameters
             *
             * @warning Not implemented. The layer's attention score is a normalised dot
             * product between neighbouring feature vectors and consumes no weight vector, so
             * the stored value was never read. Throws rather than pretending the
             * configuration took effect.
             *
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
                dot_product += features[src * in_features_ + i] * features[dst * in_features_ + i];
            }

            return std::tanh(dot_product / std::sqrt(static_cast<float>(in_features_)));
        }

        inline void GraphAttention::forward(const sparse::SparseMatrixCSR<float>& graph, const float* node_features, float* output) {
            if (node_features == nullptr || output == nullptr) {
                throw std::invalid_argument("GraphAttention::forward: node_features and output must not be null");
            }

            const int num_nodes = graph.num_rows();

            // Column indices address node_features and the score matrix, both of which are
            // sized by num_rows() here. A graph whose column indices reach past num_rows --
            // which a square adjacency matrix would never contain, but any other CSR matrix
            // can -- indexed outside both.
            //
            // Measured under ASan with a 2x3 matrix whose two entries point at column 2: a
            // heap-buffer-overflow WRITE into attention_scores_ (the score of the write below
            // is num_nodes * num_nodes for a 2-node graph, and the index was 0*2+2). Checking
            // up front means the caller gets a diagnosable exception instead of corruption,
            // and the loops below are bounds-safe for the rest of the function.
            for (int node = 0; node < num_nodes; ++node) {
                const int begin = graph.row_offsets()[node];
                const int end = graph.row_offsets()[node + 1];
                for (int idx = begin; idx < end; ++idx) {
                    const int neighbor = graph.col_indices()[idx];
                    if (neighbor < 0 || neighbor >= num_nodes) {
                        throw std::out_of_range("GraphAttention::forward: row " + std::to_string(node) + " references column " + std::to_string(neighbor) + ", outside [0, "
                                                + std::to_string(num_nodes) + ")");
                    }
                }
            }

            attention_scores_.resize(num_nodes * num_nodes, 0.0f);

            for (int i = 0; i < num_nodes; ++i) {
                for (int idx = graph.row_offsets()[i]; idx < graph.row_offsets()[i + 1]; ++idx) {
                    int neighbor = graph.col_indices()[idx];
                    attention_scores_[i * num_nodes + neighbor] = compute_attention(neighbor, i, node_features);
                }
            }

            for (int i = 0; i < num_nodes; ++i) {
                float sum = 0.0f;

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
            (void)weights;
            throw std::logic_error(
                "GraphAttention::set_attention_weights: attention weights are not implemented. This layer scores neighbours with a dot product over "
                "their features and reads no weight vector, so storing one had no effect on forward(). Multi-head attention with "
                "per-head projections is not available in this class.");
        }

    }  // namespace gnn
}  // namespace nova

#endif  // NOVA_CUDA_GNN_ATTENTION_HPP
