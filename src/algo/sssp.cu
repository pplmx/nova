#include "cuda/algo/sssp.h"

#include "cuda/device/error.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cuda::algo::sssp {

static SSSPConfig g_config;

void set_config(const SSSPConfig& config) {
    g_config = config;
}

SSSPConfig get_config() {
    return g_config;
}

template <typename Weight>
__global__ void sssp_relax_kernel(const int* row_offsets,
                                   const int* col_indices,
                                   const float* weights,
                                   Weight* distances_in,
                                   Weight* distances_out,
                                   const int* frontier,
                                   int frontier_size,
                                   Weight delta) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < frontier_size) {
        const int node = frontier[tid];
        const int row_start = row_offsets[node];
        const int row_end = row_offsets[node + 1];
        const Weight node_dist = distances_in[node];

        for (int i = row_start; i < row_end; ++i) {
            const int neighbor = col_indices[i];
            const Weight edge_weight = static_cast<Weight>(weights[i]);
            const Weight new_dist = node_dist + edge_weight;

            if (new_dist < distances_out[neighbor] && new_dist <= delta * 1000) {
                atomicMin(reinterpret_cast<unsigned int*>(&distances_out[neighbor]),
                          static_cast<unsigned int>(new_dist * 1000));
            }
        }
    }
}

template <typename Weight>
void delta_stepping(const graph::CSRGraph& graph, int source, Weight* distances,
                    Weight delta, cudaStream_t stream) {
    const int num_vertices = static_cast<int>(graph.num_vertices);
    const int* row_offsets = graph.row_offsets;
    const int* col_indices = graph.columns;
    const float* graph_weights = graph.weights;

    thrust::device_ptr<Weight> d_distances(distances);

    thrust::fill(thrust::seq, d_distances, d_distances + num_vertices,
                 static_cast<Weight>(INF));

    distances[source] = Weight{0};

    memory::Buffer<int> frontier_current(num_vertices);
    memory::Buffer<int> frontier_next(num_vertices);
    memory::Buffer<int> bucket(static_cast<size_t>(num_vertices * 1000 / delta));

    int* d_frontier_current = frontier_current.data();
    int* d_frontier_next = frontier_next.data();
    int* d_bucket = bucket.data();

    CUDA_CHECK(cudaMemset(d_frontier_current, 0, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_frontier_current, &source, sizeof(int), cudaMemcpyHostToDevice));

    int current_frontier_size = 1;
    int iterations = 0;

    while (current_frontier_size > 0 && iterations < g_config.max_iterations) {
        int next_frontier_size = 0;

        sssp_relax_kernel<Weight><<<1, current_frontier_size, 0, stream>>>(
            row_offsets, col_indices, graph_weights,
            distances, distances,
            d_frontier_current, current_frontier_size,
            delta);

        cudaStreamSynchronize(stream);

        thrust::device_ptr<Weight> d_dist(distances);
        thrust::copy(thrust::seq, d_dist, d_dist + current_frontier_size, d_frontier_next);

        current_frontier_size = 0;
        iterations++;
    }
}

template <typename Weight>
void bellman_ford(const graph::CSRGraph& graph, int source, Weight* distances,
                  cudaStream_t stream) {
    const int num_vertices = static_cast<int>(graph.num_vertices);
    const int* row_offsets = graph.row_offsets;
    const int* col_indices = graph.columns;
    const float* graph_weights = graph.weights;

    thrust::device_ptr<Weight> d_distances(distances);

    thrust::fill(thrust::seq, d_distances, d_distances + num_vertices,
                 static_cast<Weight>(INF));

    distances[source] = Weight{0};

    for (int iter = 0; iter < num_vertices - 1; ++iter) {
        bool changed = false;

        for (int u = 0; u < num_vertices; ++u) {
            if (distances[u] == INF) continue;

            const int row_start = row_offsets[u];
            const int row_end = row_offsets[u + 1];

            for (int i = row_start; i < row_end; ++i) {
                int v = col_indices[i];
                Weight w = graph_weights ? static_cast<Weight>(graph_weights[i]) : Weight{1};

                if (distances[u] + w < distances[v]) {
                    distances[v] = distances[u] + w;
                    changed = true;
                }
            }
        }

        if (!changed) break;
    }
}

template <typename Weight>
memory::Buffer<Weight> compute_distances(const graph::CSRGraph& graph, int source,
                                         Weight delta, cudaStream_t stream) {
    const int num_vertices = static_cast<int>(graph.num_vertices);
    memory::Buffer<Weight> distances(num_vertices);

    if (g_config.use_delta_stepping) {
        delta_stepping(graph, source, distances.data(), delta, stream);
    } else {
        bellman_ford(graph, source, distances.data(), stream);
    }

    return distances;
}

SSSPResult run(const graph::CSRGraph& graph, int source,
               float delta, cudaStream_t stream) {
    SSSPResult result;
    result.num_vertices = static_cast<int>(graph.num_vertices);
    result.converged = true;
    result.iterations = 0;

    result.distances = compute_distances<float>(graph, source, delta, stream);

    return result;
}

template void delta_stepping<float>(const graph::CSRGraph&, int, float*, float, cudaStream_t);

template void bellman_ford<float>(const graph::CSRGraph&, int, float*, cudaStream_t);

template memory::Buffer<float> compute_distances<float>(const graph::CSRGraph&, int, float, cudaStream_t);

}  // namespace cuda::algo::sssp
