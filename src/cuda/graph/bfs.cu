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

    int* d_distances;
    bool* d_visited;
    CUDA_CHECK(cudaMalloc(&d_distances, graph.num_vertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited, graph.num_vertices * sizeof(bool)));

    CUDA_CHECK(cudaMemcpyAsync(d_distances, result.d_distances, graph.num_vertices * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_visited, result.d_visited, graph.num_vertices * sizeof(bool), cudaMemcpyDeviceToDevice, stream));

    int current_level = 0;
    bool frontier_exists = true;

    while (frontier_exists) {
        frontier_exists = false;
        int block_size = 256;
        int grid_size = (graph.num_vertices + block_size - 1) / block_size;

        bfs_frontier_kernel<<<grid_size, block_size, 0, stream>>>(
            graph.d_row_offsets,
            graph.d_columns,
            d_distances,
            d_visited,
            d_distances,
            d_visited,
            current_level,
            graph.num_vertices
        );
        CUDA_CHECK(cudaGetLastError());

        current_level++;
        if (current_level > graph.num_vertices) break;

        if (stream) {
            cudaStreamSynchronize(stream);
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(result.d_distances, d_distances, graph.num_vertices * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.d_visited, d_visited, graph.num_vertices * sizeof(bool), cudaMemcpyDeviceToDevice, stream));

    result.download();

    result.visited_count = 0;
    result.max_distance = 0;
    for (int v = 0; v < graph.num_vertices; ++v) {
        if (result.distances[v] >= 0) {
            result.visited_count++;
            result.max_distance = std::max(result.max_distance, result.distances[v]);
        }
    }

    cudaFree(d_distances);
    cudaFree(d_visited);

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
