#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
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

        /**
         * @brief Out-degree of vertex v
         *
         * @throws std::out_of_range if v is outside [0, num_vertices), or if the graph holds no
         *         host arrays. Every other accessor here reads a count; this one indexes
         *         row_offsets, and an out-of-range v read past the array without saying so. A
         *         default-constructed graph dereferenced null on any call.
         */
        int degree(int v) const {
            if (row_offsets == nullptr) {
                throw ::std::out_of_range(
                    "CSRGraph::degree: the graph holds no row_offsets; construct with dimensions "
                    "or fill it first");
            }
            if (v < 0 || v >= num_vertices) {
                throw ::std::out_of_range("CSRGraph::degree: vertex " + ::std::to_string(v) + " is outside [0, " + ::std::to_string(num_vertices) + ")");
            }
            return row_offsets[v + 1] - row_offsets[v];
        }

        size_t memory_usage() const;
    };

    std::unique_ptr<CSRGraph> create_csr_from_edges(const int* src_vertices, const int* dst_vertices, const float* weights, int num_vertices, int num_edges);

    // Build a CSR graph from an adjacency list. `adjacency[v]` holds the
    // out-neighbors of vertex v. When `weights` is non-empty it must have the same
    // shape as `adjacency` (same number of rows, each row the same length as the
    // corresponding adjacency row) and supplies the per-edge weight in parallel
    // position; otherwise every edge gets the unit weight 1.0f. A shape mismatch
    // throws std::invalid_argument rather than silently misaligning weights.
    std::unique_ptr<CSRGraph> create_csr_from_adjacency(const std::vector<std::vector<int>>& adjacency, const std::vector<std::vector<float>>& weights = {});

    void csr_mv(const CSRGraph& graph, const float* x, float* y, cudaStream_t stream = nullptr);

    void csr_mv_transpose(const CSRGraph& graph, const float* x, float* y, cudaStream_t stream = nullptr);

    /**
     * @brief Compute per-vertex in/out degrees on the device
     *
     * @warning Requires upload() to have been called on the graph. The kernels read d_row_offsets
     * and d_columns, which are null until then, and nothing here checks: a caller that forgets gets
     * a launch reading through null device pointers, which corrupts the CUDA context rather than
     * failing an assertion. Every device-side operation in this header carries that requirement.
     *
     * @param graph       the CSR graph; must have been uploaded
     * @param out_degrees device array of graph.num_vertices elements
     * @param in_degrees  optional device array of graph.num_vertices elements; may be null
     * @param stream      CUDA stream
     */
    void compute_degrees(const CSRGraph& graph, int* out_degrees, int* in_degrees = nullptr, cudaStream_t stream = nullptr);

    bool validate_csr(const CSRGraph& graph);

}  // namespace cuda::graph
