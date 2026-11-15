/**
 * @file matrix.hpp
 * @brief Unified sparse matrix with GPU memory management
 * @defgroup SparseMatrix Unified Sparse Matrix
 * @ingroup sparse
 *
 * Modern sparse matrix type with automatic GPU memory management.
 * Provides multiple construction methods and device-memory buffers.
 *
 * Example usage:
 * @code
 * // Create from dense
 * auto matrix = SparseMatrix<float>::FromDense(dense.data(), rows, cols, 1e-6f);
 *
 * // Perform SpMV
 * spmv(*matrix, d_x, d_y);
 *
 * // Copy back to host
 * std::vector<float> result(rows);
 * matrix.values_buffer().copy_to(result.data(), rows);
 * @endcode
 *
 * @see sparse_matrix.hpp For legacy format classes
 * @see sparse_ops.hpp For supported operations
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/memory/buffer.h"

namespace nova::sparse {

/**
 * @brief Unified sparse matrix with GPU memory management
 * @class SparseMatrix
 * @tparam T Element type (float, double, etc.)
 * @ingroup SparseMatrix
 *
 * Provides GPU-resident sparse matrix storage with CSR format.
 * Data is stored in cuda::memory::Buffer for automatic management.
 *
 * @note Thread-safe buffer operations
 * @see cuda::memory::Buffer For memory management details
 */
template<typename T>
class SparseMatrix {
public:
    /** @brief Default constructor creates empty matrix */
    SparseMatrix() = default;

    /**
     * @brief Construct with pre-allocated storage
     * @param num_rows Number of rows
     * @param num_cols Number of columns
     * @param nnz Number of non-zeros
     */
    SparseMatrix(int num_rows, int num_cols, int nnz)
        : values_(static_cast<size_t>(nnz))
        , row_offsets_(static_cast<size_t>(num_rows + 1))
        , col_indices_(static_cast<size_t>(nnz))
        , num_rows_(num_rows)
        , num_cols_(num_cols) {}

    /**
     * @brief Create from dense matrix
     * @param dense Dense matrix data (row-major)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param sparsity_threshold Treat elements with |value| <= threshold as zero
     * @return Sparse matrix, or nullopt if fully dense
     *
     * @note Automatically copies data to GPU
     * @note Time complexity: O(rows * cols)
     */
    static std::optional<SparseMatrix> FromDense(const T* dense, int rows, int cols,
                                                  float sparsity_threshold = 0.0f);

    /**
     * @brief Create from host data
     * @param values Non-zero values
     * @param row_offsets Row offset array (size = rows + 1)
     * @param col_indices Column indices
     * @param num_rows Number of rows
     * @param num_cols Number of columns
     * @return Sparse matrix with copied data
     */
    static SparseMatrix FromHostData(std::vector<T> values,
                                      std::vector<int> row_offsets,
                                      std::vector<int> col_indices,
                                      int num_rows, int num_cols);

    /**
     * @brief Create a square adjacency matrix from an edge list
     * @param edges Vector of (src, dst) directed vertex pairs
     * @param num_vertices Number of vertices
     * @param weights Optional edge weights, one per edge
     * @return Sparse matrix whose row `src` holds src's out-edges
     * @throws std::invalid_argument if an endpoint is outside
     *         [0, num_vertices) or `weights` does not have one entry per edge
     */
    static SparseMatrix FromEdgeList(const std::vector<std::pair<int, int>>& edges,
                                      int num_vertices,
                                      const std::vector<T>* weights = nullptr);

    /** @brief Get number of rows */
    int rows() const { return num_rows_; }

    /** @brief Get number of columns */
    int cols() const { return num_cols_; }

    /** @brief Get number of non-zeros */
    int nnz() const { return static_cast<int>(values_.size()); }

    /** @brief Get mutable pointer to device values */
    T* values() { return values_.data(); }

    /** @brief Get mutable pointer to device row offsets */
    int* row_offsets() { return row_offsets_.data(); }

    /** @brief Get mutable pointer to device column indices */
    int* col_indices() { return col_indices_.data(); }

    /** @brief Get const pointer to device values */
    const T* values() const { return values_.data(); }

    /** @brief Get const pointer to device row offsets */
    const int* row_offsets() const { return row_offsets_.data(); }

    /** @brief Get const pointer to device column indices */
    const int* col_indices() const { return col_indices_.data(); }

    /** @brief Get reference to values buffer */
    cuda::memory::Buffer<T>& values_buffer() { return values_; }

    /** @brief Get reference to row offsets buffer */
    cuda::memory::Buffer<int>& row_offsets_buffer() { return row_offsets_; }

    /** @brief Get reference to column indices buffer */
    cuda::memory::Buffer<int>& col_indices_buffer() { return col_indices_; }

    /** @brief Get const reference to values buffer */
    const cuda::memory::Buffer<T>& values_buffer() const { return values_; }

    /** @brief Get const reference to row offsets buffer */
    const cuda::memory::Buffer<int>& row_offsets_buffer() const { return row_offsets_; }

    /** @brief Get const reference to column indices buffer */
    const cuda::memory::Buffer<int>& col_indices_buffer() const { return col_indices_; }

    /**
     * @brief Copy matrix data back to host
     * @param[out] out_values Output values vector
     * @param[out] out_row_offsets Output row offsets vector
     * @param[out] out_col_indices Output column indices vector
     */
    void copy_to_host(std::vector<T>& out_values,
                      std::vector<int>& out_row_offsets,
                      std::vector<int>& out_col_indices) const;

private:
    cuda::memory::Buffer<T> values_;
    cuda::memory::Buffer<int> row_offsets_;
    cuda::memory::Buffer<int> col_indices_;
    int num_rows_ = 0;
    int num_cols_ = 0;
};

template<typename T>
std::optional<SparseMatrix<T>> SparseMatrix<T>::FromDense(const T* dense,
                                                          int rows, int cols,
                                                          float threshold) {
    std::vector<T> values;
    std::vector<int> row_offsets(1, 0);
    std::vector<int> col_indices;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            T val = dense[i * cols + j];
            if (val != T{0} && std::abs(val) > threshold) {
                values.push_back(val);
                col_indices.push_back(j);
            }
        }
        row_offsets.push_back(static_cast<int>(values.size()));
    }

    if (values.empty()) {
        return std::nullopt;
    }

    SparseMatrix<T> result(rows, cols, static_cast<int>(values.size()));
    result.values_buffer().copy_from(values.data(), static_cast<size_t>(values.size()));
    result.row_offsets_buffer().copy_from(row_offsets.data(), static_cast<size_t>(row_offsets.size()));
    result.col_indices_buffer().copy_from(col_indices.data(), static_cast<size_t>(col_indices.size()));

    return result;
}

template<typename T>
SparseMatrix<T> SparseMatrix<T>::FromHostData(std::vector<T> values,
                                               std::vector<int> row_offsets,
                                               std::vector<int> col_indices,
                                               int num_rows, int num_cols) {
    SparseMatrix<T> result(num_rows, num_cols, static_cast<int>(values.size()));
    result.values_buffer().copy_from(values.data(), static_cast<size_t>(values.size()));
    result.row_offsets_buffer().copy_from(row_offsets.data(), static_cast<size_t>(row_offsets.size()));
    result.col_indices_buffer().copy_from(col_indices.data(), static_cast<size_t>(col_indices.size()));
    return result;
}

template<typename T>
SparseMatrix<T> SparseMatrix<T>::FromEdgeList(const std::vector<std::pair<int, int>>& edges,
                                              int num_vertices,
                                              const std::vector<T>* weights) {
    if (num_vertices < 0) {
        throw std::invalid_argument(
            "SparseMatrix::FromEdgeList: num_vertices must be non-negative, got " +
            std::to_string(num_vertices));
    }
    if (weights && weights->size() != edges.size()) {
        throw std::invalid_argument(
            "SparseMatrix::FromEdgeList: " + std::to_string(weights->size()) +
            " weights for " + std::to_string(edges.size()) + " edges");
    }
    for (size_t e = 0; e < edges.size(); ++e) {
        const int src = edges[e].first;
        const int dst = edges[e].second;
        // Both endpoints index rows/columns of the result; an out-of-range one
        // would make spmv read past x or write past y.
        if (src < 0 || src >= num_vertices || dst < 0 || dst >= num_vertices) {
            throw std::invalid_argument(
                "SparseMatrix::FromEdgeList: edge " + std::to_string(e) + " (" +
                std::to_string(src) + ", " + std::to_string(dst) +
                ") is outside [0, " + std::to_string(num_vertices) + ")");
        }
    }

    // Bucket the edges by source so each CSR row holds exactly its own
    // out-edges. Counting sort keeps the original edge order within a row.
    std::vector<int> row_counts(static_cast<size_t>(num_vertices), 0);
    for (const auto& edge : edges) {
        ++row_counts[static_cast<size_t>(edge.first)];
    }

    std::vector<int> row_offsets(static_cast<size_t>(num_vertices) + 1, 0);
    for (int v = 0; v < num_vertices; ++v) {
        row_offsets[static_cast<size_t>(v) + 1] =
            row_offsets[static_cast<size_t>(v)] + row_counts[static_cast<size_t>(v)];
    }

    std::vector<int> next(row_offsets.begin(), row_offsets.end() - 1);
    std::vector<int> col_indices(edges.size());
    std::vector<T> values(edges.size());

    for (size_t e = 0; e < edges.size(); ++e) {
        const size_t slot = static_cast<size_t>(next[static_cast<size_t>(edges[e].first)]++);
        col_indices[slot] = edges[e].second;
        values[slot] = weights ? (*weights)[e] : T{1};
    }

    return FromHostData(values, row_offsets, col_indices, num_vertices, num_vertices);
}

template<typename T>
void SparseMatrix<T>::copy_to_host(std::vector<T>& out_values,
                                    std::vector<int>& out_row_offsets,
                                    std::vector<int>& out_col_indices) const {
    out_values.resize(static_cast<size_t>(nnz()));
    out_row_offsets.resize(static_cast<size_t>(num_rows_ + 1));
    out_col_indices.resize(static_cast<size_t>(nnz()));

    values_.copy_to(out_values.data(), static_cast<size_t>(nnz()));
    row_offsets_.copy_to(out_row_offsets.data(), static_cast<size_t>(num_rows_ + 1));
    col_indices_.copy_to(out_col_indices.data(), static_cast<size_t>(nnz()));
}

/**
 * @brief Sparse Matrix-Vector Product (y = A*x)
 * @param A Sparse matrix
 * @param x Input vector (device)
 * @param[out] y Output vector (device)
 * @tparam T Element type
 * @tparam T Element type
 * @note Synchronous operation
 * @note Device memory: O(nnz + rows + cols) for temporaries
 *
 * @def spmv
 * @ingroup SparseMatrix
 */
template<typename T>
void spmv(const SparseMatrix<T>& A, const T* x, T* y);

/**
 * @brief Sparse Matrix-Vector Product with host vectors
 * @param A Sparse matrix
 * @param x Input vector (host)
 * @param[out] y Output vector (host)
 * @tparam T Element type
 *
 * @note Automatically copies data between host and device
 * @note For performance-critical code, use the device pointer version
 *
 * @def spmv_host
 * @ingroup SparseMatrix
 */
template<typename T>
void spmv(const SparseMatrix<T>& A, const std::vector<T>& x, std::vector<T>& y);

/**
 * @brief Async Sparse Matrix-Vector Product
 * @param A Sparse matrix
 * @param x Input vector (device)
 * @param[out] y Output vector (device)
 * @param stream CUDA stream
 * @tparam T Element type
 *
 * @def spmv_async
 * @ingroup SparseMatrix
 */
template<typename T>
void spmv_async(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream);

/**
 * @brief Sparse Matrix-Vector Product with transpose (y = A^T*x)
 * @param A Sparse matrix
 * @param x Input vector (device)
 * @param[out] y Output vector (device)
 * @tparam T Element type
 *
 * @def spmv_transpose
 * @ingroup SparseMatrix
 */
template<typename T>
void spmv_transpose(const SparseMatrix<T>& A, const T* x, T* y);

/**
 * @brief Async Sparse Matrix-Vector Product with transpose
 * @param A Sparse matrix
 * @param x Input vector (device)
 * @param[out] y Output vector (device)
 * @param stream CUDA stream
 * @tparam T Element type
 *
 * @def spmv_transpose_async
 * @ingroup SparseMatrix
 */
template<typename T>
void spmv_transpose_async(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream);

/** @cond */
template<typename T>
class SparseMatrixCSR;
/** @endcond */

/**
 * @brief Convert legacy SparseMatrixCSR to unified SparseMatrix
 * @param csr Legacy CSR matrix
 * @return Unified SparseMatrix with GPU memory
 * @tparam T Element type
 *
 * @def ToSparseMatrix
 * @ingroup SparseMatrix
 * @deprecated Use SparseMatrix directly
 */
template<typename T>
SparseMatrix<T> ToSparseMatrix(const SparseMatrixCSR<T>& csr) {
    if (csr.nnz() == 0) {
        return SparseMatrix<T>(csr.num_rows(), csr.num_cols(), 0);
    }

    std::vector<T> values(csr.nnz());
    std::vector<int> row_offsets(csr.num_rows() + 1);
    std::vector<int> col_indices(csr.nnz());

    std::copy(csr.values(), csr.values() + csr.nnz(), values.begin());
    std::copy(csr.row_offsets(), csr.row_offsets() + csr.num_rows() + 1, row_offsets.begin());
    std::copy(csr.col_indices(), csr.col_indices() + csr.nnz(), col_indices.begin());

    return SparseMatrix<T>::FromHostData(values, row_offsets, col_indices,
                                          csr.num_rows(), csr.num_cols());
}

}  // namespace nova::sparse
