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
#include <random>
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
             * @param graph Adjacency matrix
             * @param seed_nodes Starting nodes
             * @param num_samples Maximum neighbors per node
             * @return Sampled neighbor information
             */
            SampledNeighbors sample_neighbors(const sparse::SparseMatrixCSR<float>& graph, const std::vector<int>& seed_nodes, int num_samples);

            /**
             * @brief Extract k-hop neighborhood
             * @param graph Adjacency matrix
             * @param seed_nodes Starting nodes
             * @param k Number of hops
             * @return Aggregated node IDs
             */
            std::vector<int> k_hop_aggregation(const sparse::SparseMatrixCSR<float>& graph, const std::vector<int>& seed_nodes, int k);

        private:
            std::mt19937 rng_;
        };

        inline GraphSampler::SampledNeighbors GraphSampler::sample_neighbors(const sparse::SparseMatrixCSR<float>& graph, const std::vector<int>& seed_nodes, int num_samples) {
            SampledNeighbors result;
            result.node_ids = seed_nodes;
            result.neighbor_counts.resize(seed_nodes.size());

            for (size_t i = 0; i < seed_nodes.size(); ++i) {
                int node = seed_nodes[i];
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
