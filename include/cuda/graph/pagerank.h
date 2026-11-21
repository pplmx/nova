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

    /**
     * @brief PageRank options
     *
     * @warning This implementation uses **sink-collapsing** dangling semantics, not the
     *          mass-conserving redistribution used by the textbook algorithm. A vertex with
     *          out-degree 0 keeps no rank and gives it to nobody: its rank becomes
     *          @ref min_rank and the rest is discarded, so the ranks no longer sum to 1.
     *
     *          Measured on the path 0 -> 1 -> 2 with the default options over 200 iterations:
     *
     *              r0 = 0.050000   r1 = 0.092500   r2 = 0.000000   total = 0.142500
     *
     *          and on a graph where every vertex is isolated, all ranks are 0. Standard
     *          PageRank redistributes a dangling vertex's rank uniformly over the graph,
     *          which keeps the total at 1.0 and changes the relative ordering in any graph
     *          that has a sink.
     *
     *          This is deliberate and covered by tests (the path test asserts exactly the
     *          values above), so it is documented rather than changed: switching to
     *          redistribution would silently change every existing caller's scores. A caller
     *          that needs mass conservation should add a self-loop to each dangling vertex
     *          before calling, or normalise the result itself.
     */
    struct PageRankOptions {
        float damping = 0.85f;
        float tolerance = 1e-6f;
        int max_iterations = 50;
        /**
         * Rank assigned to a vertex with out-degree 0, and the floor a dangling vertex is
         * reset to instead of accumulating. See the note on this struct: a dangling vertex's
         * rank is discarded rather than redistributed, so this is the value it ends at.
         */
        float min_rank = 0.0f;
    };

    PageRankResult pagerank(const CSRGraph& graph, const PageRankOptions& options = PageRankOptions {}, cudaStream_t stream = nullptr);

    /**
     * @brief One PageRank update step
     *
     * Push-based: each edge u -> v adds prev_ranks[u] / out_degree(u) to next_ranks[v], then
     * the teleport term is folded in and the damping applied. A vertex with out-degree 0 is
     * assigned @p min_rank, which is the sink-collapsing behaviour described on
     * PageRankOptions.
     *
     * @warning Reads the graph's device arrays, so upload() must have been called. A
     *          zero-vertex graph returns without doing anything.
     */
    void pagerank_iteration(const CSRGraph& graph, const float* prev_ranks, float* next_ranks, float damping, float min_rank, cudaStream_t stream = nullptr);

    /**
     * @brief Sum of |next - prev| over every vertex
     *
     * The L1 difference the iteration loop compares against options.tolerance. Both arrays
     * are host arrays of @p num_vertices elements.
     */
    float compute_pagerank_delta(const float* prev_ranks, const float* next_ranks, int num_vertices);

}  // namespace cuda::graph
