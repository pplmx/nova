#include "cuda/graph/csr_graph.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <numeric>

namespace cuda::graph {

CSRGraph::CSRGraph()
    : num_vertices(0),
      num_edges(0),
      row_offsets(nullptr),
      columns(nullptr),
      weights(nullptr),
      d_row_offsets(nullptr),
      d_columns(nullptr),
      d_weights(nullptr) {}

CSRGraph::CSRGraph(int num_vertices, int num_edges)
    : num_vertices(num_vertices),
      num_edges(num_edges),
      row_offsets(nullptr),
      columns(nullptr),
      weights(nullptr),
      d_row_offsets(nullptr),
      d_columns(nullptr),
      d_weights(nullptr) {

    row_offsets = new int[num_vertices + 1];
    columns = new int[num_edges];
    weights = new float[num_edges];
}

CSRGraph::~CSRGraph() {
    clear();
}

void CSRGraph::allocate_device() {
    CUDA_CHECK(cudaMalloc(&d_row_offsets, (num_vertices + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_columns, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, num_edges * sizeof(float)));
}

void CSRGraph::upload() {
    if (!d_row_offsets) allocate_device();

    CUDA_CHECK(cudaMemcpy(d_row_offsets, row_offsets, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns, columns, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, num_edges * sizeof(float), cudaMemcpyHostToDevice));
}

void CSRGraph::free_device() {
    if (d_row_offsets) {
        cudaFree(d_row_offsets);
        d_row_offsets = nullptr;
    }
    if (d_columns) {
        cudaFree(d_columns);
        d_columns = nullptr;
    }
    if (d_weights) {
        cudaFree(d_weights);
        d_weights = nullptr;
    }
}

void CSRGraph::clear() {
    delete[] row_offsets;
    delete[] columns;
    delete[] weights;

    row_offsets = nullptr;
    columns = nullptr;
    weights = nullptr;

    free_device();
}

size_t CSRGraph::memory_usage() const {
    size_t host_mem = (num_vertices + 1) * sizeof(int) + num_edges * (sizeof(int) + sizeof(float));
    size_t device_mem = host_mem;
    return host_mem + device_mem;
}

namespace {

void count_sort_edges(
    const int* src_vertices,
    const int* dst_vertices,
    const float* in_weights,
    int* out_dst,
    float* out_weights,
    const int* row_offsets,
    int num_vertices,
    int num_edges
) {
    for (int v = 0; v < num_vertices; ++v) {
        int start = row_offsets[v];
        int end = row_offsets[v + 1];
        for (int i = start; i < end; ++i) {
            out_dst[i] = dst_vertices[i];
            out_weights[i] = in_weights ? in_weights[i] : 1.0f;
        }
    }
}

}  // anonymous namespace

std::unique_ptr<CSRGraph> create_csr_from_edges(
    const int* src_vertices,
    const int* dst_vertices,
    const float* weights,
    int num_vertices,
    int num_edges
) {
    auto graph = std::make_unique<CSRGraph>(num_vertices, num_edges);

    int* counts = new int[num_vertices]();
    for (int i = 0; i < num_edges; ++i) {
        counts[src_vertices[i]]++;
    }

    graph->row_offsets[0] = 0;
    for (int v = 0; v < num_vertices; ++v) {
        graph->row_offsets[v + 1] = graph->row_offsets[v] + counts[v];
    }

    int* current_offset = new int[num_vertices]();
    for (int i = 0; i < num_edges; ++i) {
        int src = src_vertices[i];
        int pos = graph->row_offsets[src] + current_offset[src]++;
        graph->columns[pos] = dst_vertices[i];
        graph->weights[pos] = weights ? weights[i] : 1.0f;
    }

    delete[] counts;
    delete[] current_offset;

    return graph;
}

std::unique_ptr<CSRGraph> create_csr_from_adjacency(
    const std::vector<std::vector<int>>& adjacency,
    bool weighted
) {
    int num_vertices = static_cast<int>(adjacency.size());
    int num_edges = 0;
    for (const auto& neighbors : adjacency) {
        num_edges += static_cast<int>(neighbors.size());
    }

    auto graph = std::make_unique<CSRGraph>(num_vertices, num_edges);

    graph->row_offsets[0] = 0;
    for (int v = 0; v < num_vertices; ++v) {
        graph->row_offsets[v + 1] = graph->row_offsets[v] + static_cast<int>(adjacency[v].size());
    }

    for (int v = 0; v < num_vertices; ++v) {
        int offset = graph->row_offsets[v];
        for (size_t i = 0; i < adjacency[v].size(); ++i) {
            graph->columns[offset + i] = adjacency[v][i];
            graph->weights[offset + i] = weighted ? 1.0f : 1.0f;
        }
    }

    return graph;
}

__global__ void csr_mv_kernel(
    const int* row_offsets,
    const int* columns,
    const float* weights,
    const float* x,
    float* y,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    float sum = 0.0f;
    int start = row_offsets[v];
    int end = row_offsets[v + 1];

    for (int i = start; i < end; ++i) {
        sum += weights[i] * x[columns[i]];
    }

    y[v] = sum;
}

void csr_mv(
    const CSRGraph& graph,
    const float* x,
    float* y,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (graph.num_vertices + block_size - 1) / block_size;

    csr_mv_kernel<<<grid_size, block_size, 0, stream>>>(
        graph.d_row_offsets,
        graph.d_columns,
        graph.d_weights,
        x,
        y,
        graph.num_vertices
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void csr_mv_transpose_kernel(
    const int* row_offsets,
    const int* columns,
    const float* weights,
    const float* x,
    float* y,
    int num_vertices,
    int num_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges) return;

    int dst = columns[tid];
    atomicAdd(&y[dst], weights[tid] * x[tid]);
}

void csr_mv_transpose(
    const CSRGraph& graph,
    const float* x,
    float* y,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemsetAsync(y, 0, graph.num_vertices * sizeof(float), stream));

    int block_size = 256;
    int grid_size = (graph.num_edges + block_size - 1) / block_size;

    csr_mv_transpose_kernel<<<grid_size, block_size, 0, stream>>>(
        graph.d_row_offsets,
        graph.d_columns,
        graph.d_weights,
        x,
        y,
        graph.num_vertices,
        graph.num_edges
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void compute_degrees_kernel(
    const int* row_offsets,
    int* out_degrees,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    out_degrees[v] = row_offsets[v + 1] - row_offsets[v];
}

void compute_degrees(
    const CSRGraph& graph,
    int* out_degrees,
    int* in_degrees,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (graph.num_vertices + block_size - 1) / block_size;

    compute_degrees_kernel<<<grid_size, block_size, 0, stream>>>(
        graph.d_row_offsets,
        out_degrees,
        graph.num_vertices
    );
    CUDA_CHECK(cudaGetLastError());

    (void)in_degrees;
}

bool validate_csr(const CSRGraph& graph) {
    if (graph.num_vertices < 0 || graph.num_edges < 0) return false;
    if (graph.row_offsets[0] != 0) return false;
    if (graph.row_offsets[graph.num_vertices] != graph.num_edges) return false;

    for (int v = 0; v < graph.num_vertices; ++v) {
        if (graph.row_offsets[v] > graph.row_offsets[v + 1]) return false;
        for (int i = graph.row_offsets[v]; i < graph.row_offsets[v + 1]; ++i) {
            if (graph.columns[i] < 0 || graph.columns[i] >= graph.num_vertices) return false;
        }
    }

    return true;
}

}  // namespace cuda::graph
