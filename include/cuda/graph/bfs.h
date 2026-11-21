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

    /**
     * @brief Traverse the graph from @p source
     *
     * Returns when the result is complete and its HOST arrays are populated: the final
     * cudaMemcpy is followed by download(). Passing a @p stream moves the device work onto
     * that stream, but the function still synchronises before returning, so the returned
     * object is always ready to read.
     *
     * @param graph Graph to traverse; it is uploaded if it has not been
     * @param source Vertex to start from; must be in [0, num_vertices)
     * @param stream Stream for the device work; work is completed before returning
     */
    BFSResult bfs(const CSRGraph& graph, int source, cudaStream_t stream = nullptr);

    /**
     * @brief Traverse the graph and store the result in @p result
     *
     * @warning Despite the name this is **not** asynchronous. It runs the whole traversal,
     * including the download that fills the host arrays, before returning:
     *
     *     bfs_async(graph, result, 0, stream);
     *     // no synchronisation here:
     *     result.distance_to(1)   // already the final distance
     *
     * Measured with an explicit stream and no synchronisation afterwards, the distances were
     * 0 1 2 1 -- the completed traversal -- so the name describes nothing and a caller who
     * omits a synchronisation it thinks it needs loses nothing, while one who relies on the
     * call returning before the work starts would be wrong.
     *
     * The name is kept because renaming a public symbol is a breaking change and this
     * declaration has been in the header since the module was added; `bfs()` is what a caller
     * should use, and this is a thin wrapper that assigns its return value.
     *
     * @param graph Graph to traverse
     * @param result Overwritten with the traversal; any previous contents are released
     * @param source Vertex to start from; must be in [0, num_vertices)
     * @param stream Stream for the device work; work is completed before returning
     */
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
