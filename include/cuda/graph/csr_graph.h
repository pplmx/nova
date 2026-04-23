#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace cuda::graph {

struct CSRGraph {
    int num_vertices;
    int num_edges;

    int* row_offsets;
    int* columns;
    float* weights;

    int* d_row_offsets;
    int* d_columns;
    float* d_weights;

    CSRGraph();
    CSRGraph(int num_vertices, int num_edges);
    ~CSRGraph();

    void allocate_device();
    void upload();
    void free_device();
    void clear();

    int vertices() const { return num_vertices; }
    int edges() const { return num_edges; }

    int degree(int v) const {
        return row_offsets[v + 1] - row_offsets[v];
    }

    size_t memory_usage() const;
};

std::unique_ptr<CSRGraph> create_csr_from_edges(
    const int* src_vertices,
    const int* dst_vertices,
    const float* weights,
    int num_vertices,
    int num_edges
);

std::unique_ptr<CSRGraph> create_csr_from_adjacency(
    const std::vector<std::vector<int>>& adjacency,
    bool weighted = false
);

void csr_mv(
    const CSRGraph& graph,
    const float* x,
    float* y,
    cudaStream_t stream = nullptr
);

void csr_mv_transpose(
    const CSRGraph& graph,
    const float* x,
    float* y,
    cudaStream_t stream = nullptr
);

void compute_degrees(
    const CSRGraph& graph,
    int* out_degrees,
    int* in_degrees = nullptr,
    cudaStream_t stream = nullptr
);

bool validate_csr(const CSRGraph& graph);

}  // namespace cuda::graph
