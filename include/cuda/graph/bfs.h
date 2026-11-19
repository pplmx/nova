#pragma once

#include "cuda/graph/csr_graph.h"

namespace cuda::graph {

    struct BFSResult {
        int num_vertices;
        int* distances;
        int* d_distances;
        bool* visited;
        bool* d_visited;
        int visited_count;
        int max_distance;

        BFSResult();
        explicit BFSResult(int num_vertices);
        ~BFSResult();

        // Owns raw host+device pointers: move-only (a shallow copy would leave
        // two objects freeing the same memory).
        BFSResult(const BFSResult&) = delete;
        BFSResult& operator=(const BFSResult&) = delete;
        BFSResult(BFSResult&& other) noexcept;
        BFSResult& operator=(BFSResult&& other) noexcept;

        void init_source(int source);
        void upload();
        void download();
        void clear();

        int distance_to(int v) const { return distances[v]; }
        bool is_reachable(int v) const { return distances[v] >= 0; }

        size_t memory_usage() const;
    };

    BFSResult bfs(const CSRGraph& graph, int source, cudaStream_t stream = nullptr);

    void bfs_async(const CSRGraph& graph, BFSResult& result, int source, cudaStream_t stream = nullptr);

    /**
     * @brief Number of vertices at distance zero in the BFS result
     *
     * @warning The name says "components", but this does not count connected components and
     * cannot: a BFSResult holds distances and a visited flag, no edge information, so nothing here
     * can tell whether two unreached vertices belong to the same component. A single-source BFS
     * leaves exactly one vertex at distance zero — the source — so for a completed traversal this
     * returns 1 regardless of the graph or how many components it has.
     *
     * The implementation counts `distances[v] == 0`, which is the source marker; `-1` marks
     * unreached. Nothing calls this function today, so it is marked deprecated rather than
     * renamed; a caller that genuinely wants components needs a traversal per unvisited vertex,
     * which this signature cannot express.
     */
    [[deprecated("counts distance-zero vertices, not connected components; see the header")]]
    int count_reachable_components(const BFSResult& result);

}  // namespace cuda::graph
