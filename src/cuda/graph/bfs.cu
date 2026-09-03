#include "cuda/graph/bfs.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <climits>
#include <queue>

namespace cuda::graph {

BFSResult::BFSResult()
    : num_vertices(0),
      distances(nullptr),
      d_distances(nullptr),
      visited(nullptr),
      d_visited(nullptr),
      visited_count(0),
      max_distance(0) {}

BFSResult::BFSResult(int num_vertices)
    : num_vertices(num_vertices),
      distances(nullptr),
      d_distances(nullptr),
      visited(nullptr),
      d_visited(nullptr),
      visited_count(0),
      max_distance(0) {
    distances = new int[num_vertices];
    visited = new bool[num_vertices];
    CUDA_CHECK(cudaMalloc(&d_distances, num_vertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited, num_vertices * sizeof(bool)));
}

BFSResult::~BFSResult() {
    clear();
}

void BFSResult::init_source(int source) {
    std::fill(distances, distances + num_vertices, -1);
    std::fill(visited, visited + num_vertices, false);
    distances[source] = 0;
    visited[source] = true;
}

void BFSResult::upload() {
    CUDA_CHECK(cudaMemcpy(d_distances, distances, num_vertices * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited, visited, num_vertices * sizeof(bool), cudaMemcpyHostToDevice));
}

void BFSResult::download() {
    CUDA_CHECK(cudaMemcpy(distances, d_distances, num_vertices * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(visited, d_visited, num_vertices * sizeof(bool), cudaMemcpyDeviceToHost));
}

void BFSResult::clear() {
    delete[] distances;
    delete[] visited;
    if (d_distances) {
        cudaFree(d_distances);
        d_distances = nullptr;
    }
    if (d_visited) {
        cudaFree(d_visited);
        d_visited = nullptr;
    }
    distances = nullptr;
    visited = nullptr;
}

size_t BFSResult::memory_usage() const {
    size_t host_mem = num_vertices * (sizeof(int) + sizeof(bool));
    size_t device_mem = num_vertices * (sizeof(int) + sizeof(bool));
    return host_mem + device_mem;
}

namespace {

__global__ void bfs_frontier_kernel(
    const int* row_offsets,
    const int* columns,
    const int* distances,
    const bool* visited,
    int* next_distances,
    bool* next_visited,
    int current_level,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    if (distances[v] == current_level) {
        int start = row_offsets[v];
        int end = row_offsets[v + 1];
        for (int i = start; i < end; ++i) {
            int neighbor = columns[i];
            if (!visited[neighbor]) {
                next_visited[neighbor] = true;
                next_distances[neighbor] = current_level + 1;
            }
        }
    }
}

__global__ void bfs_merge_kernel(
    const int* next_distances,
    const bool* next_visited,
    int* distances,
    bool* visited,
    int* changed,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    if (next_visited[v] && !visited[v]) {
        visited[v] = true;
        distances[v] = next_distances[v];
        *changed = 1;
    }
}

}  // anonymous namespace

BFSResult bfs(
    const CSRGraph& graph,
    int source,
    cudaStream_t stream
) {
    BFSResult result(graph.num_vertices);
    result.init_source(source);
    result.upload();

    CSRGraph& non_const_graph = const_cast<CSRGraph&>(graph);
    non_const_graph.upload();

    const int n = graph.num_vertices;

    // Working state: d_dist/d_vis accumulate the explored set across levels,
    // while d_next_dist/d_next_vis receive the freshly discovered frontier.
    // Merging (bfs_merge_kernel) folds the frontier into the accumulated state
    // and reports whether anything new was reached, which drives the loop.
    int* d_dist;
    bool* d_vis;
    int* d_next_dist;
    bool* d_next_vis;
    int* d_changed;
    CUDA_CHECK(cudaMalloc(&d_dist, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vis, n * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_next_dist, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_vis, n * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_dist, result.d_distances,
                               n * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vis, result.d_visited,
                               n * sizeof(bool), cudaMemcpyDeviceToDevice, stream));

    // The frontier kernel only writes entries for this level's newly reached
    // neighbors; slots it never touches must read as false in the merge, so
    // zero both scratch buffers once up front (cudaMalloc does not guarantee
    // zeroed memory).
    CUDA_CHECK(cudaMemsetAsync(d_next_dist, 0, n * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_next_vis, 0, n * sizeof(bool), stream));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    int current_level = 0;
    bool changed = true;
    while (changed && current_level < n) {
        CUDA_CHECK(cudaMemsetAsync(d_changed, 0, sizeof(int), stream));

        bfs_frontier_kernel<<<grid_size, block_size, 0, stream>>>(
            graph.d_row_offsets,
            graph.d_columns,
            d_dist,
            d_vis,
            d_next_dist,
            d_next_vis,
            current_level,
            n
        );
        CUDA_CHECK(cudaGetLastError());

        bfs_merge_kernel<<<grid_size, block_size, 0, stream>>>(
            d_next_dist,
            d_next_vis,
            d_dist,
            d_vis,
            d_changed,
            n
        );
        CUDA_CHECK(cudaGetLastError());

        if (stream) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        int host_changed = 0;
        CUDA_CHECK(cudaMemcpy(&host_changed, d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost));
        changed = host_changed != 0;

        current_level++;
    }

    CUDA_CHECK(cudaMemcpyAsync(result.d_distances, d_dist,
                               n * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.d_visited, d_vis,
                               n * sizeof(bool), cudaMemcpyDeviceToDevice, stream));
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    result.download();

    result.visited_count = 0;
    result.max_distance = 0;
    for (int v = 0; v < n; ++v) {
        if (result.distances[v] >= 0) {
            result.visited_count++;
            result.max_distance = std::max(result.max_distance, result.distances[v]);
        }
    }

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_vis));
    CUDA_CHECK(cudaFree(d_next_dist));
    CUDA_CHECK(cudaFree(d_next_vis));
    CUDA_CHECK(cudaFree(d_changed));
    return result;
}

void bfs_async(
    const CSRGraph& graph,
    BFSResult& result,
    int source,
    cudaStream_t stream
) {
    CSRGraph& non_const_graph = const_cast<CSRGraph&>(graph);
    non_const_graph.upload();
    result = bfs(non_const_graph, source, stream);
}

int count_reachable_components(const BFSResult& result) {
    int components = 0;
    for (int v = 0; v < result.num_vertices; ++v) {
        if (result.distances[v] == 0) {
            components++;
        }
    }
    return components;
}

}  // namespace cuda::graph
