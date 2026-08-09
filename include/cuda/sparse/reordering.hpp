/**
 * @file reordering.hpp
 * @brief Sparse matrix bandwidth reduction reordering
 * @defgroup reordering Matrix Reordering
 * @ingroup sparse
 *
 * Provides bandwidth reduction reordering for sparse matrices,
 * improving cache locality and solver convergence.
 *
 * Supported methods:
 * - RCM (Reverse Cuthill-McKee) for symmetric matrices
 *
 * Example usage:
 * @code
 * RCMReorderer<float> reorderer;
 * auto result = reorderer.reorder(A);
 * std::cout << "Bandwidth reduced by " << result.bandwidth_reduction_ratio << "x\n";
 * @endcode
 *
 * @see krylov.hpp For iterative solvers that benefit from reordering
 */

#pragma once

#include "cuda/sparse/matrix.hpp"
#include <vector>
#include <queue>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace nova::sparse {

/**
 * @brief Error for reordering operations
 * @class ReorderingError
 * @ingroup reordering
 */
class ReorderingError : public std::runtime_error {
public:
    /** @brief Construct with error message */
    explicit ReorderingError(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * @brief Result of matrix reordering
 * @struct ReorderingResult
 * @ingroup reordering
 */
struct ReorderingResult {
    /** @brief Permutation vector P such that P*A*P^T is reordered */
    std::vector<int> permutation;

    /** @brief Inverse permutation P^-1 */
    std::vector<int> inverse_permutation;

    /** @brief Original matrix bandwidth */
    int original_bandwidth = 0;

    /** @brief Reordered matrix bandwidth */
    int reordered_bandwidth = 0;

    /** @brief Ratio of original to reordered bandwidth */
    double bandwidth_reduction_ratio = 0.0;
};

/**
 * @brief Reverse Cuthill-McKee bandwidth reduction
 * @class RCMReorderer
 * @tparam T Element type
 * @ingroup reordering
 *
 * Reduces matrix bandwidth for better numerical stability and cache utilization.
 */
template<typename T>
class RCMReorderer {
public:
    /** @brief Default constructor */
    RCMReorderer() = default;

    /**
     * @brief Reorder sparse matrix
     * @param A Input sparse matrix
     * @param in_place Whether to reorder in-place (not supported, ignored)
     * @return Reordering result with permutation
     */
    ReorderingResult reorder(const SparseMatrix<T>& A, bool in_place = false);

    SparseMatrix<T> apply_reordering(const SparseMatrix<T>& A, const ReorderingResult& result);

    static int compute_bandwidth(const SparseMatrix<T>& A);

private:
    int find_starting_node(const SparseMatrix<T>& A);

    std::vector<int> bfs_level_order(const SparseMatrix<T>& A, int start_node);

    static int compute_matrix_bandwidth(const SparseMatrix<T>& A);

    void apply_permutation(std::vector<T>& values,
                          std::vector<int>& row_offsets,
                          std::vector<int>& col_indices,
                          const std::vector<int>& perm);
};

template<typename T>
ReorderingResult RCMReorderer<T>::reorder(const SparseMatrix<T>& A, bool in_place) {
    ReorderingResult result;
    const int n = A.rows();

    (void)in_place;

    result.original_bandwidth = compute_matrix_bandwidth(A);

    int start = find_starting_node(A);
    std::vector<int> level_order = bfs_level_order(A, start);

    result.permutation.resize(n);
    for (int i = 0; i < n; ++i) {
        result.permutation[i] = level_order[i];
    }

    std::vector<int> inv_perm(n);
    for (int i = 0; i < n; ++i) {
        inv_perm[result.permutation[i]] = i;
    }
    result.inverse_permutation = inv_perm;

    auto reordered_matrix = apply_reordering(A, result);

    result.reordered_bandwidth = compute_matrix_bandwidth(reordered_matrix);

    if (result.original_bandwidth > 0) {
        result.bandwidth_reduction_ratio =
            1.0 - static_cast<double>(result.reordered_bandwidth) / result.original_bandwidth;
    }

    return result;
}

template<typename T>
SparseMatrix<T> RCMReorderer<T>::apply_reordering(const SparseMatrix<T>& A,
                                                  const ReorderingResult& result) {
    const int n = A.rows();
    const int nnz = A.nnz();

    std::vector<T> h_values(nnz);
    std::vector<int> h_row_offsets(n + 1);
    std::vector<int> h_col_indices(nnz);
    A.copy_to_host(h_values, h_row_offsets, h_col_indices);

    apply_permutation(h_values, h_row_offsets, h_col_indices, result.permutation);

    return SparseMatrix<T>::FromHostData(h_values, h_row_offsets, h_col_indices, n, n);
}

template<typename T>
int RCMReorderer<T>::find_starting_node(const SparseMatrix<T>& A) {
    const int n = A.rows();

    // SparseMatrix stores arrays on the device; stage a host copy before the
    // host-side degree scan (A.row_offsets() returns a device pointer).
    std::vector<T> h_values;
    std::vector<int> h_row_offsets;
    std::vector<int> h_col_indices;
    A.copy_to_host(h_values, h_row_offsets, h_col_indices);

    int min_degree = std::numeric_limits<int>::max();
    int start_node = 0;

    for (int i = 0; i < n; ++i) {
        int degree = h_row_offsets[i + 1] - h_row_offsets[i];
        if (degree > 0 && degree < min_degree) {
            min_degree = degree;
            start_node = i;
        }
    }

    return start_node;
}

template<typename T>
std::vector<int> RCMReorderer<T>::bfs_level_order(const SparseMatrix<T>& A, int start_node) {
    const int n = A.rows();

    // Stage a host copy of the CSR arrays (device pointers cannot be read on
    // the CPU).
    std::vector<T> h_values;
    std::vector<int> h_row_offsets;
    std::vector<int> h_col_indices;
    A.copy_to_host(h_values, h_row_offsets, h_col_indices);

    std::vector<int> visited(n, 0);
    std::vector<int> level_order;
    level_order.reserve(n);

    std::queue<int> queue;
    queue.push(start_node);
    visited[start_node] = 1;

    while (!queue.empty()) {
        std::vector<int> current_level;

        while (!queue.empty()) {
            int node = queue.front();
            queue.pop();
            level_order.push_back(node);

            const int row_start = h_row_offsets[node];
            const int row_end = h_row_offsets[node + 1];

            for (int idx = row_start; idx < row_end; ++idx) {
                int neighbor = h_col_indices[idx];
                if (!visited[neighbor]) {
                    visited[neighbor] = 1;
                    current_level.push_back(neighbor);
                }
            }
        }

        std::reverse(current_level.begin(), current_level.end());

        for (int neighbor : current_level) {
            queue.push(neighbor);
        }
    }

    return level_order;
}

template<typename T>
int RCMReorderer<T>::compute_bandwidth(const SparseMatrix<T>& A) {
    return compute_matrix_bandwidth(A);
}

template<typename T>
int RCMReorderer<T>::compute_matrix_bandwidth(const SparseMatrix<T>& A) {
    const int n = A.rows();

    // Stage a host copy of the CSR arrays (device pointers cannot be read on
    // the CPU).
    std::vector<T> h_values;
    std::vector<int> h_row_offsets;
    std::vector<int> h_col_indices;
    A.copy_to_host(h_values, h_row_offsets, h_col_indices);

    int max_bandwidth = 0;

    for (int i = 0; i < n; ++i) {
        const int row_start = h_row_offsets[i];
        const int row_end = h_row_offsets[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int j = h_col_indices[idx];
            int bandwidth = std::abs(i - j);
            max_bandwidth = std::max(max_bandwidth, bandwidth);
        }
    }

    return max_bandwidth;
}

template<typename T>
void RCMReorderer<T>::apply_permutation(std::vector<T>& values,
                                        std::vector<int>& row_offsets,
                                        std::vector<int>& col_indices,
                                        const std::vector<int>& perm) {
    const int n = static_cast<int>(row_offsets.size()) - 1;
    int nnz = static_cast<int>(values.size());

    std::vector<T> new_values(nnz);
    std::vector<int> new_row_offsets(n + 1);
    std::vector<int> new_col_indices(nnz);

    for (int i = 0; i < n; ++i) {
        int new_row = perm[i];
        int old_row_start = row_offsets[i];
        int old_row_end = row_offsets[i + 1];
        int new_row_start = new_row_offsets[new_row];
        int nnz_in_row = old_row_end - old_row_start;

        for (int k = 0; k < nnz_in_row; ++k) {
            new_values[new_row_start + k] = values[old_row_start + k];
            new_col_indices[new_row_start + k] = perm[col_indices[old_row_start + k]];
        }
    }

    for (int i = 0; i < n; ++i) {
        new_row_offsets[i] = row_offsets[perm[i]];
    }
    new_row_offsets[n] = nnz;

    for (int i = 1; i <= n; ++i) {
        if (new_row_offsets[i] < new_row_offsets[i - 1]) {
            new_row_offsets[i] = new_row_offsets[i - 1] +
                (row_offsets[perm[i]] - row_offsets[perm[i - 1]]);
        }
    }

    values = std::move(new_values);
    row_offsets = std::move(new_row_offsets);
    col_indices = std::move(new_col_indices);
}

}
