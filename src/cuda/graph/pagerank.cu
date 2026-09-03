#include "cuda/graph/pagerank.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <cmath>
#include <vector>

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

PageRankResult::PageRankResult(PageRankResult&& other) noexcept
    : num_vertices(other.num_vertices),
      ranks(other.ranks),
      d_ranks(other.d_ranks),
      iterations(other.iterations),
      final_delta(other.final_delta) {
    other.num_vertices = 0;
    other.ranks = nullptr;
    other.d_ranks = nullptr;
    other.iterations = 0;
    other.final_delta = 0.0f;
}

PageRankResult& PageRankResult::operator=(PageRankResult&& other) noexcept {
    if (this != &other) {
        clear();
        num_vertices = other.num_vertices;
        ranks = other.ranks;
        d_ranks = other.d_ranks;
        iterations = other.iterations;
        final_delta = other.final_delta;
        other.num_vertices = 0;
        other.ranks = nullptr;
        other.d_ranks = nullptr;
        other.iterations = 0;
        other.final_delta = 0.0f;
    }
    return *this;
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

// Push-based PageRank update: for each edge (u -> v) stored in the CSR, the
// source u distributes prev_ranks[u] / out_deg(u) into next_ranks[v]. The CSR
// rows hold a vertex's out-neighbors, so a single pass over the flat edge
// array covers every contribution. The binary search on row_offsets recovers
// the source u of each edge index e (the unique u with
// row_offsets[u] <= e < row_offsets[u + 1]).
__global__ void pagerank_scatter_kernel(
    const int* row_offsets,
    const int* columns,
    const float* prev_ranks,
    const int* out_degrees,
    float* next_ranks,
    int num_vertices,
    int num_edges
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    int lo = 0;
    int hi = num_vertices;  // row_offsets[hi] == num_edges
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (row_offsets[mid] <= e) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    const int u = lo;
    const int degree = out_degrees[u];
    if (degree == 0) return;  // dangling source distributes nothing

    const int v = columns[e];
    atomicAdd(&next_ranks[v], prev_ranks[u] / static_cast<float>(degree));
}

// After the scatter accumulation, fold in the teleport term and apply the
// damping factor. Dangling vertices (out-degree 0) end at min_rank, matching
// the documented option semantics.
__global__ void pagerank_finalize_kernel(
    const int* out_degrees,
    float* ranks,  // in: accumulated scatter sums; out: final ranks
    float damping,
    float min_rank,
    float teleport,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    if (out_degrees[v] == 0) {
        ranks[v] = min_rank;
        return;
    }
    ranks[v] = teleport + damping * ranks[v];
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
    const int n = graph.num_vertices;
    const int m = graph.num_edges;

    // Out-degrees are computed on-device from the uploaded CSR (the pre-v2
    // path rebuilt a host vector and block-copied it to the device every
    // iteration); graph.d_row_offsets must be uploaded, which pagerank() does.
    int* d_out_degrees;
    CUDA_CHECK(cudaMalloc(&d_out_degrees, n * sizeof(int)));
    compute_degrees(graph, d_out_degrees, nullptr, stream);

    const int block_size = 256;
    const int vertex_grid = (n + block_size - 1) / block_size;
    // Clamp to >= 1 grid: a 0-edge graph yields a 0-block launch otherwise,
    // which the driver rejects. The scatter kernel early-returns e >= m, so
    // one no-op block is harmless.
    const int edge_grid = std::max(1, (m + block_size - 1) / block_size);

    const float teleport = (1.0f - damping) / static_cast<float>(n);

    CUDA_CHECK(cudaMemsetAsync(next_ranks, 0, n * sizeof(float), stream));

    pagerank_scatter_kernel<<<edge_grid, block_size, 0, stream>>>(
        graph.d_row_offsets,
        graph.d_columns,
        prev_ranks,
        d_out_degrees,
        next_ranks,
        n,
        m
    );
    CUDA_CHECK(cudaGetLastError());

    pagerank_finalize_kernel<<<vertex_grid, block_size, 0, stream>>>(
        d_out_degrees,
        next_ranks,
        damping,
        min_rank,
        teleport,
        n
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_out_degrees));
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
    const int n = graph.num_vertices;
    PageRankResult result(n);

    float* d_prev;
    float* d_next;
    CUDA_CHECK(cudaMalloc(&d_prev, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next, n * sizeof(float)));

    std::fill(result.ranks, result.ranks + n, 1.0f / n);

    // Host snapshot of the previous iteration's ranks, so the convergence
    // delta compares the freshly computed ranks against what they were before
    // this update (compare-next-vs-prev, never next-vs-next).
    std::vector<float> prev_host(result.ranks, result.ranks + n);

    result.upload();
    CUDA_CHECK(cudaMemcpy(d_prev, result.d_ranks, n * sizeof(float), cudaMemcpyDeviceToDevice));
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CSRGraph& non_const_graph = const_cast<CSRGraph&>(graph);
    non_const_graph.upload();

    float* d_prev_temp = d_prev;
    float* d_next_temp = d_next;

    int iters_run = 0;
    float delta = 0.0f;

    while (iters_run < options.max_iterations) {
        pagerank_iteration(non_const_graph, d_prev_temp, d_next_temp,
                           options.damping, options.min_rank, stream);

        if (stream) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        CUDA_CHECK(cudaMemcpy(result.d_ranks, d_next_temp, n * sizeof(float), cudaMemcpyDeviceToDevice));
        result.download();

        delta = compute_pagerank_delta(result.ranks, prev_host.data(), n);
        result.final_delta = delta;
        iters_run++;

        if (delta < options.tolerance) {
            break;
        }

        std::copy(result.ranks, result.ranks + n, prev_host.begin());
        std::swap(d_prev_temp, d_next_temp);
    }

    result.iterations = iters_run;

    CUDA_CHECK(cudaFree(d_prev));
    CUDA_CHECK(cudaFree(d_next));

    return result;
}

}  // namespace cuda::graph
