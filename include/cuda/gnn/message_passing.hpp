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
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
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
            void forward(const sparse::SparseMatrixCSR<float>& graph, const float* node_features, float* output);

            /**
             * @brief GCN-style aggregation
             *
             * Computes, for each node i:
             *
             *     output[i] = sum over neighbours j of node_features[j]
             *                 / sqrt(degree(i) + 1)
             *
             * Two things distinguish this from the normalised adjacency GCN is usually
             * introduced with (Kipf & Welling, D^-1/2 (A + I) D^-1/2):
             *
             *  - The divisor is the RECEIVING node's degree only. The symmetric form also
             *    divides by the sender's, so an edge carries the same weight in both
             *    directions. Here it does not: measured on a 3-node graph where node 0 has
             *    degree 2 and node 1 has degree 1, an impulse at node 1 reaches node 0 with
             *    weight 0.577350, while the same impulse reversed reaches node 1 with
             *    0.707107.
             *  - There is no self-loop. A node with degree 0 aggregates nothing and returns
             *    0, discarding its own feature; the (A + I) form would return the feature
             *    unchanged. Measured with two isolated nodes holding 7 and -3: both outputs
             *    are 0.
             *
             * The `+ 1` in the divisor is a damping term, not a self-loop: it shrinks the
             * contribution of high-degree nodes without adding the node's own feature.
             *
             * This is stated rather than changed because the behaviour is defined, tested and
             * consistent; a caller that needs true GCN normalisation should normalise the
             * CSR matrix itself (scale each entry by 1/sqrt(deg_i deg_j)) and add the
             * diagonal before calling.
             *
             * @param graph Adjacency matrix
             * @param node_features Input node features, one per node
             * @param[out] output Aggregated features, one per node
             */
            void gcn_aggregate(const sparse::SparseMatrixCSR<float>& graph, const float* node_features, float* output);

            /**
             * @brief GCN-style propagation
             *
             * Applies gcn_aggregate() @p hops times. hops=0 copies the input through
             * unchanged, and a negative count throws. See gcn_aggregate() for the
             * normalisation this uses at each step.
             *
             * @param graph Adjacency matrix
             * @param node_features Input node features, one per node
             * @param[out] output Output features, one per node
             * @param hops Number of aggregation rounds; must not be negative
             */
            void gcn_propagate(const sparse::SparseMatrixCSR<float>& graph, const float* node_features, float* output, int hops = 1);

        private:
            MessageFunction message_fn_;
            AggregateFunction aggregate_fn_;

            /// Rejects a graph whose column indices do not name an existing node.
            ///
            /// Every entry point indexes node_features with the graph's column index, so a
            /// column at or past num_rows() reads outside the caller's buffer. Measured under
            /// ASan with a 2-node graph whose single entry points at column 5 and a feature
            /// buffer of exactly num_rows() floats, as documented: stack-buffer-overflow in
            /// gcn_aggregate. forward() has the same read.
            ///
            /// This is the same defect GraphAttention::forward had, fixed there by checking
            /// up front; these three call sites were missed at the time.
            static void require_columns_within_nodes(const sparse::SparseMatrixCSR<float>& graph);
        };

        inline void MessagePassing::require_columns_within_nodes(const sparse::SparseMatrixCSR<float>& graph) {
            const int num_nodes = graph.num_rows();
            for (int node = 0; node < num_nodes; ++node) {
                const int begin = graph.row_offsets()[node];
                const int end = graph.row_offsets()[node + 1];
                for (int idx = begin; idx < end; ++idx) {
                    const int neighbor = graph.col_indices()[idx];
                    if (neighbor < 0 || neighbor >= num_nodes) {
                        throw std::out_of_range("MessagePassing: row " + std::to_string(node) + " references column " + std::to_string(neighbor) + ", outside [0, "
                                                + std::to_string(num_nodes) + ")");
                    }
                }
            }
        }

        inline void MessagePassing::forward(const sparse::SparseMatrixCSR<float>& graph, const float* node_features, float* output) {
            if (node_features == nullptr || output == nullptr) {
                throw std::invalid_argument("MessagePassing::forward: node_features and output must not be null");
            }
            require_columns_within_nodes(graph);

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
                    for (float m : messages) {
                        sum += m;
                    }
                    output[i] = sum;
                } else {
                    output[i] = 0.0f;
                }
            }
        }

        inline void MessagePassing::gcn_aggregate(const sparse::SparseMatrixCSR<float>& graph, const float* node_features, float* output) {
            if (node_features == nullptr || output == nullptr) {
                throw std::invalid_argument("MessagePassing::gcn_aggregate: node_features and output must not be null");
            }
            require_columns_within_nodes(graph);

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

        inline void MessagePassing::gcn_propagate(const sparse::SparseMatrixCSR<float>& graph, const float* node_features, float* output, int hops) {
            // hops is the number of aggregation rounds. The old `hops <= 1` branch sent 0 and
            // every negative value through one aggregation, so hops=0 did a full hop and
            // hops=-5 did too: measured, 0, 1 and -5 produced identical output
            // (2.886751 2.309401 1.732051 on the 3-node ring). A caller asking for no
            // propagation got one, and a caller whose hop count went negative from a counted
            // loop got one as well instead of an error.
            if (node_features == nullptr || output == nullptr) {
                throw std::invalid_argument("MessagePassing::gcn_propagate: node_features and output must not be null");
            }
            if (hops < 0) {
                throw std::invalid_argument("MessagePassing::gcn_propagate: hops must not be negative, got " + std::to_string(hops));
            }
            // Checked once here rather than once per hop: gcn_aggregate re-checks, but the
            // graph does not change between iterations.
            require_columns_within_nodes(graph);
            if (hops == 0) {
                std::copy(node_features, node_features + graph.num_rows(), output);
                return;
            }
            if (hops == 1) {
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

    }  // namespace gnn
}  // namespace nova

#endif  // NOVA_CUDA_GNN_MESSAGE_PASSING_HPP
