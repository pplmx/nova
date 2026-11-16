#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <limits>

#include "cuda/graph/csr_graph.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::algo::sssp {

constexpr float INF = std::numeric_limits<float>::infinity();

/// Outcome of a shortest-path run. Returned by the solver entry points so a caller
/// can distinguish "settled" from "ran out of rounds" from "negative cycle".
struct RunStats {
    int iterations = 0;
    bool converged = false;
    bool negative_cycle = false;
};

template <typename Weight>
RunStats delta_stepping(const graph::CSRGraph& graph, int source, Weight* distances,
                        Weight delta = Weight{1.0}, cudaStream_t stream = nullptr);

template <typename Weight>
RunStats bellman_ford(const graph::CSRGraph& graph, int source, Weight* distances,
                      cudaStream_t stream = nullptr);

template <typename Weight>
RunStats compute_distances(const graph::CSRGraph& graph, int source, Weight delta,
                           Weight* distances, cudaStream_t stream = nullptr);

/// Convenience overload that allocates the output. The caller gets the distances
/// but cannot see the run statistics; use the pointer overload when convergence
/// or negative-cycle detection matters.
template <typename Weight>
memory::Buffer<Weight> compute_distances(const graph::CSRGraph& graph, int source,
                                         Weight delta = Weight{1.0},
                                         cudaStream_t stream = nullptr);

struct SSSPResult {
    memory::Buffer<float> distances;
    int num_vertices;
    /// False when the graph contains a negative cycle reachable from the source.
    bool converged;
    /// Relaxation rounds actually performed (0 when the run short-circuited).
    int iterations;
    /// True when an extra relaxation pass still improved a distance, which makes
    /// every reported distance meaningless: they are unbounded below.
    bool negative_cycle = false;
};

SSSPResult run(const graph::CSRGraph& graph, int source,
               float delta = 1.0f, cudaStream_t stream = nullptr);

struct SSSPConfig {
    float default_delta = 1.0f;
    int max_iterations = 1000;
    float convergence_threshold = 1e-6f;
    bool use_delta_stepping = true;
};

void set_config(const SSSPConfig& config);
SSSPConfig get_config();

}  // namespace cuda::algo::sssp
