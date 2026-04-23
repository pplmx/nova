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

    void init_source(int source);
    void upload();
    void download();
    void clear();

    int distance_to(int v) const { return distances[v]; }
    bool is_reachable(int v) const { return distances[v] >= 0; }

    size_t memory_usage() const;
};

BFSResult bfs(
    const CSRGraph& graph,
    int source,
    cudaStream_t stream = nullptr
);

void bfs_async(
    const CSRGraph& graph,
    BFSResult& result,
    int source,
    cudaStream_t stream = nullptr
);

int count_reachable_components(const BFSResult& result);

}  // namespace cuda::graph
