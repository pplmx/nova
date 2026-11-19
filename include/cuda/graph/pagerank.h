#pragma once

#include "cuda/graph/csr_graph.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::graph {

    struct PageRankResult {
        int num_vertices;
        float* ranks;
        float* d_ranks;
        int iterations;
        float final_delta;

        PageRankResult();
        explicit PageRankResult(int num_vertices);
        ~PageRankResult();

        // Owns raw host+device pointers: move-only (a shallow copy would leave
        // two objects freeing the same memory).
        PageRankResult(const PageRankResult&) = delete;
        PageRankResult& operator=(const PageRankResult&) = delete;
        PageRankResult(PageRankResult&& other) noexcept;
        PageRankResult& operator=(PageRankResult&& other) noexcept;

        void upload();
        void download();
        void clear();

        /**
         * @brief PageRank score of vertex v
         *
         * @throws std::out_of_range if v is outside [0, num_vertices), or if the result holds no
         *         ranks. This is the only accessor here that indexes an array, and it checked
         *         nothing — the same gap CSRGraph::degree had.
         */
        float rank_at(int v) const {
            if (ranks == nullptr) {
                throw ::std::out_of_range("PageRankResult::rank_at: the result holds no ranks");
            }
            if (v < 0 || v >= num_vertices) {
                throw ::std::out_of_range("PageRankResult::rank_at: vertex " + ::std::to_string(v) + " is outside [0, " + ::std::to_string(num_vertices) + ")");
            }
            return ranks[v];
        }

        /**
         * @brief Index of the highest-ranked vertex
         * @return The vertex index, or -1 when the result holds no vertices
         *         (default-constructed, cleared, or moved-from)
         */
        int top_vertex() const;
        std::vector<int> top_k(int k) const;

        size_t memory_usage() const;
    };

    struct PageRankOptions {
        float damping = 0.85f;
        float tolerance = 1e-6f;
        int max_iterations = 50;
        float min_rank = 0.0f;
    };

    PageRankResult pagerank(const CSRGraph& graph, const PageRankOptions& options = PageRankOptions{}, cudaStream_t stream = nullptr);

    void pagerank_iteration(const CSRGraph& graph, const float* prev_ranks, float* next_ranks, float damping, float min_rank, cudaStream_t stream = nullptr);

    float compute_pagerank_delta(const float* prev_ranks, const float* next_ranks, int num_vertices);

}  // namespace cuda::graph
