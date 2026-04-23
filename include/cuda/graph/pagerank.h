#pragma once

#include "cuda/graph/csr_graph.h"

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

    void upload();
    void download();
    void clear();

    float rank_at(int v) const { return ranks[v]; }

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

PageRankResult pagerank(
    const CSRGraph& graph,
    const PageRankOptions& options = PageRankOptions{},
    cudaStream_t stream = nullptr
);

void pagerank_iteration(
    const CSRGraph& graph,
    const float* prev_ranks,
    float* next_ranks,
    float damping,
    float min_rank,
    cudaStream_t stream = nullptr
);

float compute_pagerank_delta(
    const float* prev_ranks,
    const float* next_ranks,
    int num_vertices
);

}  // namespace cuda::graph
