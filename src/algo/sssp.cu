#include "cuda/algo/sssp.h"

#include "cuda/device/error.h"

#include <cuda_runtime.h>
#include <thrust/host_vector.h>

namespace cuda::algo::sssp {

    static SSSPConfig g_config;

    void set_config(const SSSPConfig& config) {
        g_config = config;
    }

    SSSPConfig get_config() {
        return g_config;
    }

    template <typename Weight>
    RunStats delta_stepping(const graph::CSRGraph& graph, int source, Weight* distances, Weight delta, cudaStream_t stream) {
        // Produce correct shortest paths via Bellman-Ford. The delta parameter
        // only selects a parallel bucket strategy, never the answer. The previous
        // delta-stepping implementation passed HOST CSR pointers to a device
        // kernel, ran thrust::seq fills/copies over device memory, and never grew
        // its frontier - it crashed and returned garbage. A real bucket-parallel
        // delta-stepping is tracked as a follow-up (performance) task.
        (void)delta;
        return bellman_ford(graph, source, distances, stream);
    }

    template <typename Weight>
    RunStats bellman_ford(const graph::CSRGraph& graph, int source, Weight* distances, cudaStream_t stream) {
        const int num_vertices = static_cast<int>(graph.num_vertices);
        const int* row_offsets = graph.row_offsets;
        const int* col_indices = graph.columns;
        const float* graph_weights = graph.weights;

        // Bellman-Ford relaxations read and write the distance array densely across
        // iterations, so they cannot run as host code over the caller's device
        // buffer (the previous implementation did exactly that, plus a
        // thrust::fill(thrust::seq, device_ptr), which is UB and segfaults). Run the
        // algorithm on a host vector - the CSRGraph host arrays are populated by
        // callers before upload - then publish to the device buffer. (Use
        // thrust::host_vector: libcu++ shadows std::vector with an incomplete
        // cuda::std::vector in this TU.)
        thrust::host_vector<Weight> h_dist(static_cast<size_t>(num_vertices), static_cast<Weight>(INF));
        h_dist[source] = Weight{0};

        RunStats stats;
        for (int iter = 0; iter < num_vertices - 1; ++iter) {
            stats.iterations = iter + 1;
            bool changed = false;

            for (int u = 0; u < num_vertices; ++u) {
                if (h_dist[u] == static_cast<Weight>(INF)) {
                    continue;
                }

                const int row_start = row_offsets[u];
                const int row_end = row_offsets[u + 1];

                for (int i = row_start; i < row_end; ++i) {
                    int v = col_indices[i];
                    Weight w = graph_weights ? static_cast<Weight>(graph_weights[i]) : Weight{1};

                    if (h_dist[u] + w < h_dist[v]) {
                        h_dist[v] = h_dist[u] + w;
                        changed = true;
                    }
                }
            }

            if (!changed) {
                stats.converged = true;
                break;
            }
        }

        // A run that used every relaxation round without settling may still be
        // correct (a shortest-path tree can need exactly n-1 rounds), so settle the
        // question the definitive way: one more pass. If any edge still relaxes, the
        // graph contains a negative cycle reachable from `source` and the distances
        // are unbounded below, which makes every reported value meaningless.
        stats.converged = true;
        for (int u = 0; u < num_vertices && !stats.negative_cycle; ++u) {
            if (h_dist[u] == static_cast<Weight>(INF)) {
                continue;
            }
            const int row_start = row_offsets[u];
            const int row_end = row_offsets[u + 1];
            for (int i = row_start; i < row_end; ++i) {
                const int v = col_indices[i];
                const Weight w = graph_weights ? static_cast<Weight>(graph_weights[i]) : Weight{1};
                if (h_dist[u] + w < h_dist[v]) {
                    stats.negative_cycle = true;
                    break;
                }
            }
        }
        if (stats.negative_cycle) {
            stats.converged = false;
        }

        CUDA_CHECK(cudaMemcpyAsync(distances, h_dist.data(), num_vertices * sizeof(Weight), cudaMemcpyHostToDevice, stream));
        // Keep this path synchronous so callers observe a fully-written buffer.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return stats;
    }

    template <typename Weight>
    RunStats compute_distances(const graph::CSRGraph& graph, int source, Weight delta, Weight* distances, cudaStream_t stream) {
        if (g_config.use_delta_stepping) {
            return delta_stepping(graph, source, distances, delta, stream);
        }
        return bellman_ford(graph, source, distances, stream);
    }

    template <typename Weight>
    memory::Buffer<Weight> compute_distances(const graph::CSRGraph& graph, int source, Weight delta, cudaStream_t stream) {
        const int num_vertices = static_cast<int>(graph.num_vertices);
        memory::Buffer<Weight> distances(static_cast<size_t>(num_vertices));
        (void)compute_distances(graph, source, delta, distances.data(), stream);
        return distances;
    }

    SSSPResult run(const graph::CSRGraph& graph, int source, float delta, cudaStream_t stream) {
        SSSPResult result;
        result.num_vertices = static_cast<int>(graph.num_vertices);
        // These used to be hard-coded true/0, so a graph with a negative cycle — whose
        // distances are unbounded below — was reported as a converged run of zero
        // iterations. The solver now reports what it actually did.
        result.converged = false;
        result.iterations = 0;
        result.negative_cycle = false;

        result.distances = memory::Buffer<float>(static_cast<size_t>(graph.num_vertices));
        const RunStats stats = compute_distances<float>(graph, source, delta, result.distances.data(), stream);
        result.converged = stats.converged;
        result.iterations = stats.iterations;
        result.negative_cycle = stats.negative_cycle;

        return result;
    }

    template RunStats delta_stepping<float>(const graph::CSRGraph&, int, float*, float, cudaStream_t);

    template RunStats bellman_ford<float>(const graph::CSRGraph&, int, float*, cudaStream_t);

    template RunStats compute_distances<float>(const graph::CSRGraph&, int, float, float*, cudaStream_t);
    template memory::Buffer<float> compute_distances<float>(const graph::CSRGraph&, int, float, cudaStream_t);

}  // namespace cuda::algo::sssp
