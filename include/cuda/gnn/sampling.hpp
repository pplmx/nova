/**
 * @file sampling.hpp
 * @brief Graph sampling for GNN training
 * @defgroup gnn_sampling Graph Sampling
 * @ingroup gnn
 *
 * Provides graph sampling strategies for efficient GNN mini-batch training.
 * Supports neighbor sampling and k-hop subgraph extraction.
 *
 * @note Sampling reduces memory requirements for large graphs
 * @see message_passing.hpp For GNN operations
 */

#ifndef NOVA_CUDA_GNN_SAMPLING_HPP
#define NOVA_CUDA_GNN_SAMPLING_HPP

#include <cuda/sparse/sparse_matrix.hpp>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace nova {
    namespace gnn {

        /**
         * @brief Graph neighbor sampler
         * @class GraphSampler
         * @ingroup gnn_sampling
         */
        class GraphSampler {
        public:
            /**
             * @brief Construct sampler with random seed
             * @param seed Random seed for reproducibility
             */
            explicit GraphSampler(int seed = 42)
                : rng_(seed) {}

            /**
             * @brief Result of neighbor sampling
             * @struct SampledNeighbors
             */
            struct SampledNeighbors {
                /** @brief Sampled node IDs */
                std::vector<int> node_ids;
                /** @brief Number of neighbors per node */
                std::vector<int> neighbor_counts;
                /** @brief Neighbor lists per node */
                std::vector<std::vector<int>> neighbors;
            };

            /**
             * @brief Sample neighbors from graph
             *
             * @param graph Adjacency matrix
             * @param seed_nodes Starting nodes; each must lie in [0, graph.num_rows())
             * @param num_samples Maximum neighbors per node; must not be negative
             * @return Sampled neighbor information
             * @throws std::out_of_range if a seed node is outside the graph. This was a
             *         heap-buffer-overflow: the seed indexed row_offsets() directly, measured
             *         under ASan with a seed of 7 on a 3-node graph.
             * @throws std::invalid_argument if num_samples is negative. It reached
             *         `all_neighbors.begin() + num_samples`, which walks the iterator before
             *         the range begin and threw std::length_error from vector's constructor.
             */
            SampledNeighbors sample_neighbors(const sparse::SparseMatrixCSR<float>& graph, const std::vector<int>& seed_nodes, int num_samples);

            /**
             * @brief Extract k-hop neighborhood
             *
             * @param graph Adjacency matrix
             * @param seed_nodes Starting nodes; each must lie in [0, graph.num_rows())
             * @param k Number of hops; 0 returns the seeds, and a negative value is rejected
             * @return Aggregated node IDs (the seeds plus every node reached within k hops)
             * @throws std::out_of_range if a seed node is outside the graph, for the same
             *         reason as sample_neighbors() -- measured under ASan with a seed of 9 on
             *         a 3-node graph.
             * @throws std::invalid_argument if k is negative. It used to return the seeds
             *         unchanged, which reads as "nothing was reachable" rather than "the hop
             *         count is invalid".
             */
            std::vector<int> k_hop_aggregation(const sparse::SparseMatrixCSR<float>& graph, const std::vector<int>& seed_nodes, int k);

        private:
            std::mt19937 rng_;
        };

        inline GraphSampler::SampledNeighbors GraphSampler::sample_neighbors(const sparse::SparseMatrixCSR<float>& graph, const std::vector<int>& seed_nodes, int num_samples) {
            // A negative budget reached `all_neighbors.begin() + num_samples`, which walks the
            // iterator before the range: measured under ASan,
            //   terminate called after throwing an instance of 'std::length_error'
            //   what(): cannot create std::vector larger than max_size()
            if (num_samples < 0) {
                throw std::invalid_argument("GraphSampler::sample_neighbors: num_samples must not be negative, got " + std::to_string(num_samples));
            }

            const int num_nodes = graph.num_rows();

            // row_offsets()[node] is indexed with the caller's seed, so a seed outside the
            // graph reads past the array: measured under ASan, a seed of 7 on a 3-node graph
            // was a heap-buffer-overflow inside this function.
            SampledNeighbors result;
            result.node_ids = seed_nodes;
            result.neighbor_counts.resize(seed_nodes.size());
            result.neighbors.reserve(seed_nodes.size());

            for (size_t i = 0; i < seed_nodes.size(); ++i) {
                int node = seed_nodes[i];
                if (node < 0 || node >= num_nodes) {
                    throw std::out_of_range("GraphSampler::sample_neighbors: seed node " + std::to_string(node) + " at position " + std::to_string(i) + " is outside [0, "
                                            + std::to_string(num_nodes) + ")");
                }
                int degree = graph.row_offsets()[node + 1] - graph.row_offsets()[node];

                if (degree <= num_samples) {
                    std::vector<int> sampled;
                    for (int idx = graph.row_offsets()[node]; idx < graph.row_offsets()[node + 1]; ++idx) {
                        sampled.push_back(graph.col_indices()[idx]);
                    }
                    result.neighbors.push_back(sampled);
                    result.neighbor_counts[i] = sampled.size();
                } else {
                    std::vector<int> all_neighbors;
                    for (int idx = graph.row_offsets()[node]; idx < graph.row_offsets()[node + 1]; ++idx) {
                        all_neighbors.push_back(graph.col_indices()[idx]);
                    }

                    std::shuffle(all_neighbors.begin(), all_neighbors.end(), rng_);

                    std::vector<int> sampled(all_neighbors.begin(), all_neighbors.begin() + num_samples);
                    result.neighbors.push_back(sampled);
                    result.neighbor_counts[i] = num_samples;
                }
            }

            return result;
        }

        inline std::vector<int> GraphSampler::k_hop_aggregation(const sparse::SparseMatrixCSR<float>& graph, const std::vector<int>& seed_nodes, int k) {
            // A negative hop count ran the loop zero times and returned the seeds, which reads
            // as "the k-hop neighbourhood of these nodes" when nothing was traversed.
            if (k < 0) {
                throw std::invalid_argument("GraphSampler::k_hop_aggregation: k must not be negative, got " + std::to_string(k));
            }

            const int num_nodes = graph.num_rows();

            // Same out-of-range read as sample_neighbors(): the seed indexes row_offsets()
            // directly. Measured under ASan, a seed of 9 on a 3-node graph was a
            // heap-buffer-overflow inside this function.
            for (size_t i = 0; i < seed_nodes.size(); ++i) {
                if (seed_nodes[i] < 0 || seed_nodes[i] >= num_nodes) {
                    throw std::out_of_range("GraphSampler::k_hop_aggregation: seed node " + std::to_string(seed_nodes[i]) + " at position " + std::to_string(i) + " is outside [0, "
                                            + std::to_string(num_nodes) + ")");
                }
            }

            std::vector<int> frontier = seed_nodes;
            std::vector<int> visited(frontier.begin(), frontier.end());

            for (int hop = 0; hop < k; ++hop) {
                std::vector<int> next_frontier;

                for (int node : frontier) {
                    for (int idx = graph.row_offsets()[node]; idx < graph.row_offsets()[node + 1]; ++idx) {
                        int neighbor = graph.col_indices()[idx];
                        if (std::find(visited.begin(), visited.end(), neighbor) == visited.end()) {
                            next_frontier.push_back(neighbor);
                            visited.push_back(neighbor);
                        }
                    }
                }

                frontier = std::move(next_frontier);
            }

            return visited;
        }

    }  // namespace gnn
}  // namespace nova

#endif  // NOVA_CUDA_GNN_SAMPLING_HPP
