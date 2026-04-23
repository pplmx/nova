#include "cuda/graph/pagerank.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <cmath>

namespace cuda::graph {

PageRankResult::PageRankResult()
    : num_vertices(0),
      ranks(nullptr),
      d_ranks(nullptr),
      iterations(0),
      final_delta(0.0f) {}

PageRankResult::PageRankResult(int num_vertices)
    : num_vertices(num_vertices),
      iterations(0),
      final_delta(0.0f) {

    ranks = new float[num_vertices];
    CUDA_CHECK(cudaMalloc(&d_ranks, num_vertices * sizeof(float)));
}

PageRankResult::~PageRankResult() {
    clear();
}

void PageRankResult::upload() {
    CUDA_CHECK(cudaMemcpy(d_ranks, ranks, num_vertices * sizeof(float), cudaMemcpyHostToDevice));
}

void PageRankResult::download() {
    CUDA_CHECK(cudaMemcpy(ranks, d_ranks, num_vertices * sizeof(float), cudaMemcpyDeviceToHost));
}

void PageRankResult::clear() {
    delete[] ranks;
    if (d_ranks) {
        cudaFree(d_ranks);
        d_ranks = nullptr;
    }
    ranks = nullptr;
}

int PageRankResult::top_vertex() const {
    int best = 0;
    float best_rank = ranks[0];
    for (int v = 1; v < num_vertices; ++v) {
        if (ranks[v] > best_rank) {
            best_rank = ranks[v];
            best = v;
        }
    }
    return best;
}

std::vector<int> PageRankResult::top_k(int k) const {
    std::vector<std::pair<float, int>> scored(num_vertices);
    for (int v = 0; v < num_vertices; ++v) {
        scored[v] = {ranks[v], v};
    }

    std::partial_sort(
        scored.begin(),
        scored.begin() + k,
        scored.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    std::vector<int> result;
    for (int i = 0; i < std::min(k, num_vertices); ++i) {
        result.push_back(scored[i].second);
    }
    return result;
}

size_t PageRankResult::memory_usage() const {
    size_t host_mem = num_vertices * sizeof(float);
    size_t device_mem = num_vertices * sizeof(float);
    return host_mem + device_mem;
}

namespace {

__global__ void pagerank_score_kernel(
    const int* row_offsets,
    const int* columns,
    const float* prev_ranks,
    const int* out_degrees,
    float* next_ranks,
    float damping,
    float min_rank,
    int num_vertices,
    float teleport
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = row_offsets[v];
    int end = row_offsets[v + 1];
    int degree = out_degrees[v];

    if (degree == 0) {
        next_ranks[v] = min_rank;
        return;
    }

    float sum = 0.0f;
    for (int i = start; i < end; ++i) {
        int neighbor = columns[i];
        int neighbor_degree = out_degrees[neighbor];
        if (neighbor_degree > 0) {
            sum += prev_ranks[neighbor] / static_cast<float>(neighbor_degree);
        }
    }

    next_ranks[v] = teleport + damping * sum;
}

__global__ void pagerank_scale_kernel(
    float* ranks,
    float scale,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    ranks[v] *= scale;
}

__global__ void pagerank_delta_kernel(
    const float* prev,
    const float* next,
    float* deltas,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    deltas[v] = fabsf(next[v] - prev[v]);
}

}  // anonymous namespace

void pagerank_iteration(
    const CSRGraph& graph,
    const float* prev_ranks,
    float* next_ranks,
    float damping,
    float min_rank,
    cudaStream_t stream
) {
    int* d_out_degrees;
    CUDA_CHECK(cudaMalloc(&d_out_degrees, graph.num_vertices * sizeof(int)));

    int block_size = 256;
    int grid_size = (graph.num_vertices + block_size - 1) / block_size;

    for (int v = 0; v < graph.num_vertices; ++v) {
        d_out_degrees[v] = graph.degree(v);
    }
    CUDA_CHECK(cudaMemcpy(d_out_degrees, d_out_degrees, graph.num_vertices * sizeof(int), cudaMemcpyHostToDevice));

    float teleport = (1.0f - damping) / static_cast<float>(graph.num_vertices);

    pagerank_score_kernel<<<grid_size, block_size, 0, stream>>>(
        graph.d_row_offsets,
        graph.d_columns,
        prev_ranks,
        d_out_degrees,
        next_ranks,
        damping,
        min_rank,
        graph.num_vertices,
        teleport
    );
    CUDA_CHECK(cudaGetLastError());

    float scale = 1.0f;
    pagerank_scale_kernel<<<grid_size, block_size, 0, stream>>>(
        next_ranks,
        scale,
        graph.num_vertices
    );
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_out_degrees);
}

float compute_pagerank_delta(
    const float* prev_ranks,
    const float* next_ranks,
    int num_vertices
) {
    float delta = 0.0f;
    for (int v = 0; v < num_vertices; ++v) {
        delta += fabsf(next_ranks[v] - prev_ranks[v]);
    }
    return delta;
}

PageRankResult pagerank(
    const CSRGraph& graph,
    const PageRankOptions& options,
    cudaStream_t stream
) {
    PageRankResult result(graph.num_vertices);

    float* d_prev;
    float* d_next;
    CUDA_CHECK(cudaMalloc(&d_prev, graph.num_vertices * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next, graph.num_vertices * sizeof(float)));

    std::fill(result.ranks, result.ranks + graph.num_vertices, 1.0f / graph.num_vertices);
    result.upload();

    CUDA_CHECK(cudaMemcpy(d_prev, result.d_ranks, graph.num_vertices * sizeof(float), cudaMemcpyDeviceToDevice));

    CSRGraph& non_const_graph = const_cast<CSRGraph&>(graph);
    non_const_graph.upload();

    float* d_prev_temp = d_prev;
    float* d_next_temp = d_next;
    const CSRGraph& graph_ref = graph;

    int iter = 0;
    float delta = 0.0f;

    while (iter < options.max_iterations) {
        pagerank_iteration(non_const_graph, d_prev_temp, d_next_temp, options.damping, options.min_rank, stream);

        CUDA_CHECK(cudaMemcpy(result.d_ranks, d_next_temp, graph.num_vertices * sizeof(float), cudaMemcpyDeviceToDevice));
        result.download();

        delta = compute_pagerank_delta(result.ranks, result.ranks, graph.num_vertices);
        result.final_delta = delta;

        if (delta < options.tolerance) {
            break;
        }

        std::swap(d_prev_temp, d_next_temp);
        iter++;
    }

    result.iterations = iter + 1;
    result.download();

    cudaFree(d_prev);
    cudaFree(d_next);

    return result;
}

}  // namespace cuda::graph
