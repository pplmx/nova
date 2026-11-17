#include "cuda/graph/csr_graph.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

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
    // The driver happens to reject a null host source for the non-empty row_offsets copy, but
    // that is its behaviour rather than this function's contract, and the two trailing copies
    // have length zero for an empty graph — which a driver is free to accept without touching
    // the pointer. State what the requirement is instead of relying on that.
    if (row_offsets == nullptr) {
        throw ::std::invalid_argument(
            "CSRGraph::upload: row_offsets is null; construct with dimensions or fill the "
            "graph before uploading");
    }

    if (!d_row_offsets) allocate_device();

    CUDA_CHECK(cudaMemcpy(d_row_offsets, row_offsets, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns, columns, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, num_edges * sizeof(float), cudaMemcpyHostToDevice));
}

void CSRGraph::free_device() {
    // Using unchecked cudaFree: free_device() is called from clear() which
    // is called from the destructor. Throwing here would be undefined
    // behavior in a noexcept destruction context. cudaFree on invalid
    // pointers returns a non-fatal error, so silent failure is acceptable
    // during cleanup. Users needing error handling should check
    // cudaGetLastError() after calling free_device() directly.
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

    // The counts describe the buffers, which no longer exist. Leaving them set
    // made memory_usage() report memory the object had already released (measured:
    // 64 bytes for a 3-vertex graph after clear()) and let any traversal compute
    // bounds from vertices it cannot read.
    num_vertices = 0;
    num_edges = 0;
}

size_t CSRGraph::memory_usage() const {
    // Report what is actually held. A default-constructed or cleared graph has no
    // buffers, but the (num_vertices + 1) term still produced sizeof(int) and the
    // function claimed 2 * sizeof(int) bytes — memory that does not exist.
    if (row_offsets == nullptr) {
        return 0;
    }
    const size_t host_mem = static_cast<size_t>(num_vertices + 1) * sizeof(int) +
                            static_cast<size_t>(num_edges) *
                                (sizeof(int) + sizeof(float));
    return host_mem * 2;  // host copy plus the device copy
}

std::unique_ptr<CSRGraph> create_csr_from_edges(
    const int* src_vertices,
    const int* dst_vertices,
    const float* weights,
    int num_vertices,
    int num_edges
) {
    // counts[src_vertices[i]]++ indexes a host array with caller-supplied data and no bound, so
    // a vertex id outside [0, num_vertices) corrupts the heap rather than being reported. The
    // adjacency overload below validates its inputs; this one validated nothing.
    if (num_vertices <= 0) {
        throw ::std::invalid_argument(
            "create_csr_from_edges: num_vertices must be positive, got " +
            ::std::to_string(num_vertices));
    }
    if (num_edges < 0) {
        throw ::std::invalid_argument(
            "create_csr_from_edges: num_edges must not be negative, got " +
            ::std::to_string(num_edges));
    }
    if (num_edges > 0 && (src_vertices == nullptr || dst_vertices == nullptr)) {
        throw ::std::invalid_argument(
            "create_csr_from_edges: num_edges is positive but a vertex array is null");
    }

    auto graph = std::make_unique<CSRGraph>(num_vertices, num_edges);

    int* counts = new int[num_vertices]();
    for (int i = 0; i < num_edges; ++i) {
        const int src = src_vertices[i];
        if (src < 0 || src >= num_vertices) {
            delete[] counts;
            throw ::std::out_of_range(
                "create_csr_from_edges: src_vertices[" + ::std::to_string(i) + "] = " +
                ::std::to_string(src) + " is outside [0, " +
                ::std::to_string(num_vertices) + ")");
        }
        counts[src]++;
    }

    graph->row_offsets[0] = 0;
    for (int v = 0; v < num_vertices; ++v) {
        graph->row_offsets[v + 1] = graph->row_offsets[v] + counts[v];
    }

    int* current_offset = new int[num_vertices]();
    for (int i = 0; i < num_edges; ++i) {
        int src = src_vertices[i];
        const int dst = dst_vertices[i];
        // A destination outside the vertex range produces a graph whose column indices are
        // invalid. validate_csr reports that afterwards, but storing it silently leaves the
        // constructor's own output malformed and makes the caller responsible for noticing.
        if (dst < 0 || dst >= num_vertices) {
            delete[] counts;
            delete[] current_offset;
            throw ::std::out_of_range(
                "create_csr_from_edges: dst_vertices[" + ::std::to_string(i) + "] = " +
                ::std::to_string(dst) + " is outside [0, " +
                ::std::to_string(num_vertices) + ")");
        }
        int pos = graph->row_offsets[src] + current_offset[src]++;
        graph->columns[pos] = dst;
        graph->weights[pos] = weights ? weights[i] : 1.0f;
    }

    delete[] counts;
    delete[] current_offset;

    return graph;
}

std::unique_ptr<CSRGraph> create_csr_from_adjacency(
    const std::vector<std::vector<int>>& adjacency,
    const std::vector<std::vector<float>>& weights
) {
    const int num_vertices = static_cast<int>(adjacency.size());
    if (!weights.empty() && weights.size() != adjacency.size()) {
        throw std::invalid_argument(
            "create_csr_from_adjacency: weights has " +
            std::to_string(weights.size()) + " rows but adjacency has " +
            std::to_string(adjacency.size()));
    }

    int num_edges = 0;
    for (size_t v = 0; v < adjacency.size(); ++v) {
        if (!weights.empty() && weights[v].size() != adjacency[v].size()) {
            throw std::invalid_argument(
                "create_csr_from_adjacency: row " + std::to_string(v) +
                " has " + std::to_string(weights[v].size()) +
                " weights but " + std::to_string(adjacency[v].size()) +
                " neighbors");
        }
        num_edges += static_cast<int>(adjacency[v].size());
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
            graph->weights[offset + i] = weights.empty() ? 1.0f : weights[v][i];
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
    // A zero-vertex graph yields a 0-block grid, which CUDA rejects with "invalid
    // configuration argument". y = A*x over no rows is an empty result, so declining to launch
    // is the correct answer rather than an error.
    if (graph.num_vertices <= 0) {
        return;
    }

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

    // y = A^T x: edge tid (u -> v) contributes x[u] (the edge's source, the
    // row u whose range contains tid — not tid itself; only vertices with a
    // single edge would coincide). Recover u by binary search over the prefix
    // row_offsets array.
    int lo = 0;
    int hi = num_vertices;  // row_offsets[hi] == num_edges
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (row_offsets[mid] <= tid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    const int dst = columns[tid];
    atomicAdd(&y[dst], weights[tid] * x[lo]);
}

void csr_mv_transpose(
    const CSRGraph& graph,
    const float* x,
    float* y,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemsetAsync(y, 0, graph.num_vertices * sizeof(float), stream));

    int block_size = 256;
    // Same zero-grid problem as csr_mv, but keyed on edges: no edges means the transpose
    // product is y = 0, which the memset above has already written.
    if (graph.num_edges <= 0) {
        return;
    }

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

__global__ void compute_in_degrees_kernel(
    const int* columns,
    int* in_degrees,
    int num_edges
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    atomicAdd(&in_degrees[columns[e]], 1);
}

void compute_degrees(
    const CSRGraph& graph,
    int* out_degrees,
    int* in_degrees,
    cudaStream_t stream
) {
    // Guarded here as well as in pagerank_iteration: a zero-vertex graph yields a 0-block grid
    // that the driver rejects. This function is public, so it should not depend on every caller
    // remembering that. My first fix put the guard only in the caller and the call still failed.
    if (graph.num_vertices <= 0) {
        return;
    }

    const int block_size = 256;
    const int vertex_grid = (graph.num_vertices + block_size - 1) / block_size;

    compute_degrees_kernel<<<vertex_grid, block_size, 0, stream>>>(
        graph.d_row_offsets,
        out_degrees,
        graph.num_vertices
    );
    CUDA_CHECK(cudaGetLastError());

    if (in_degrees) {
        CUDA_CHECK(cudaMemsetAsync(in_degrees, 0,
                                   graph.num_vertices * sizeof(int), stream));
        const int edge_grid = std::max(1, (graph.num_edges + block_size - 1) / block_size);
        compute_in_degrees_kernel<<<edge_grid, block_size, 0, stream>>>(
            graph.d_columns,
            in_degrees,
            graph.num_edges
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

bool validate_csr(const CSRGraph& graph) {
    if (graph.num_vertices < 0 || graph.num_edges < 0) return false;

    // The host arrays are raw pointers and the default constructor leaves them null, so a
    // caller validating a graph it never filled in would dereference nullptr right here.
    // Reporting an invalid graph is what this function is for; crashing is not.
    if (graph.row_offsets == nullptr) return false;
    if (graph.num_edges > 0 && graph.columns == nullptr) return false;

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
