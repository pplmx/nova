/**
 * @file sparse_ops.hpp
 * @brief Sparse matrix operations (SpMV, SpMM)
 * @defgroup sparse_ops Sparse Operations
 * @ingroup sparse
 *
 * Provides sparse matrix-vector (SpMV) and sparse matrix-matrix (SpMM) products
 * for various sparse matrix formats.
 *
 * Example usage:
 * @code
 * // CSR SpMV
 * sparse_mv(csr_matrix, d_x, d_y);
 *
 * // Multi-column SpMM
 * sparse_mm(csr_matrix, d_B, d_C, num_cols);
 * @endcode
 *
 * @note Time complexity: O(nnz) for SpMV, O(nnz * num_cols) for SpMM
 * @see sparse_matrix.hpp For matrix format definitions
 * @see matrix.hpp For unified SparseMatrix type
 */

#ifndef NOVA_CUDA_SPARSE_OPS_HPP
#define NOVA_CUDA_SPARSE_OPS_HPP

#include <cuda/sparse/sparse_matrix.hpp>
#include <cuda/sparse/matrix.hpp>
#include <memory>

namespace nova {
    namespace sparse {

        /**
         * @brief Sparse Matrix-Vector Product for CSR format
         * @param matrix CSR sparse matrix
         * @param x Input vector (device)
         * @param[out] y Output vector (device)
         * @tparam T Element type
         *
         * Computes: y = A * x
         *
         * @def sparse_mv
         * @ingroup sparse_ops
         */
        template <typename T>
        void sparse_mv(const SparseMatrixCSR<T>& matrix, const T* x, T* y);

        /**
         * @brief Sparse Matrix-Vector Product for unified SparseMatrix
         * @param matrix SparseMatrix
         * @param x Input vector (device)
         * @param[out] y Output vector (device)
         * @tparam T Element type
         *
         * @def sparse_mv
         * @ingroup sparse_ops
         */
        template <typename T>
        void sparse_mv(const SparseMatrix<T>& matrix, const T* x, T* y);

        /**
         * @brief Sparse Matrix-Matrix Product
         * @param matrix CSR sparse matrix (m x n)
         * @param B Dense matrix (n x num_vecs)
         * @param[out] C Result matrix (m x num_vecs)
         * @param num_vecs Number of columns in B and C
         * @tparam T Element type
         *
         * Computes: C = A * B for each column of B
         *
         * @note Time complexity: O(nnz * num_vecs)
         * @note Device memory: O(m * num_vecs) for output
         *
         * @def sparse_mm
         * @ingroup sparse_ops
         */
        template <typename T>
        void sparse_mm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_vecs);

        /**
         * @brief Static operations class for various sparse formats
         * @class SparseOps
         * @tparam T Element type
         * @ingroup sparse_ops
         *
         * Provides format-specific SpMV implementations.
         */
        template <typename T>
        class SparseOps {
        public:
            /**
             * @brief SpMV for CSR format
             * @param matrix CSR matrix
             * @param x Input vector
             * @param[out] y Output vector
             *
             * @note Time complexity: O(nnz)
             */
            static void spmv(const SparseMatrixCSR<T>& matrix, const T* x, T* y);

            /**
             * @brief SpMM for CSR format
             * @param matrix CSR matrix
             * @param B Input dense matrix
             * @param[out] C Output dense matrix
             * @param num_cols Number of columns
             *
             * @note Time complexity: O(nnz * num_cols)
             */
            static void spmm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_cols);

            /**
             * @brief SpMV for ELL format
             * @param matrix ELL matrix
             * @param x Input vector
             * @param[out] y Output vector
             *
             * @note Better GPU utilization for regular sparsity
             */
            static void spmv(const SparseMatrixELL<T>& matrix, const T* x, T* y);

            /**
             * @brief SpMV for SELL format
             * @param matrix SELL matrix
             * @param x Input vector
             * @param[out] y Output vector
             *
             * @note Better load balancing for irregular patterns
             */
            static void spmv(const SparseMatrixSELL<T>& matrix, const T* x, T* y);
        };

        template <typename T>
        void SparseOps<T>::spmv(const SparseMatrixCSR<T>& matrix, const T* x, T* y) {
            int num_rows = matrix.num_rows();

            for (int i = 0; i < num_rows; ++i) {
                T sum = T {0};
                for (int idx = matrix.row_offsets()[i]; idx < matrix.row_offsets()[i + 1]; ++idx) {
                    int col = matrix.col_indices()[idx];
                    sum += matrix.values()[idx] * x[col];
                }
                y[i] = sum;
            }
        }

        template <typename T>
        void SparseOps<T>::spmm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_cols) {
            int num_rows = matrix.num_rows();

            for (int i = 0; i < num_rows; ++i) {
                for (int j = 0; j < num_cols; ++j) {
                    T sum = T {0};
                    for (int idx = matrix.row_offsets()[i]; idx < matrix.row_offsets()[i + 1]; ++idx) {
                        int col = matrix.col_indices()[idx];
                        sum += matrix.values()[idx] * B[col * num_cols + j];
                    }
                    C[i * num_cols + j] = sum;
                }
            }
        }

        template <typename T>
        void sparse_mv(const SparseMatrixCSR<T>& matrix, const T* x, T* y) {
            SparseOps<T>::spmv(matrix, x, y);
        }

        template <typename T>
        void sparse_mv(const SparseMatrix<T>& matrix, const T* x, T* y) {
            spmv(matrix, x, y);
        }

        template <typename T>
        void sparse_mm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_vecs) {
            SparseOps<T>::spmm(matrix, B, C, num_vecs);
        }

        template <typename T>
        void SparseOps<T>::spmv(const SparseMatrixELL<T>& matrix, const T* x, T* y) {
            int num_rows = matrix.num_rows();
            int max_nnz = matrix.max_nnz_per_row();

            for (int i = 0; i < num_rows; ++i) {
                T sum = T {0};
                int base = i * max_nnz;
                for (int j = 0; j < max_nnz; ++j) {
                    int col = matrix.col_indices()[base + j];
                    if (col >= 0) {
                        sum += matrix.values()[base + j] * x[col];
                    }
                }
                y[i] = sum;
            }
        }

        template <typename T>
        void SparseOps<T>::spmv(const SparseMatrixSELL<T>& matrix, const T* x, T* y) {
            int num_rows = matrix.num_rows();
            int slice_height = matrix.slice_height();
            int num_slices = (num_rows + slice_height - 1) / slice_height;

            for (int s = 0; s < num_slices; ++s) {
                int slice_start_row = s * slice_height;
                int slice_end_row = std::min(slice_start_row + slice_height, num_rows);

                int slice_base = matrix.slice_ptr()[s];
                int slice_next_base = matrix.slice_ptr()[s + 1];
                int slice_nnz = (slice_next_base - slice_base) / slice_height;

                for (int local_i = 0; local_i < slice_end_row - slice_start_row; ++local_i) {
                    int global_i = slice_start_row + local_i;
                    T sum = T {0};
                    int base = slice_base + local_i * slice_nnz;

                    for (int j = 0; j < slice_nnz; ++j) {
                        int col = matrix.col_indices()[base + j];
                        if (col >= 0) {
                            sum += matrix.values()[base + j] * x[col];
                        }
                    }
                    y[global_i] = sum;
                }
            }
        }

        /**
         * @brief SpMV for ELL format
         * @param matrix ELL sparse matrix
         * @param x Input vector
         * @param[out] y Output vector
         * @tparam T Element type
         * @def sparse_mv
         * @ingroup sparse_ops
         */
        template <typename T>
        void sparse_mv(const SparseMatrixELL<T>& matrix, const T* x, T* y) {
            SparseOps<T>::spmv(matrix, x, y);
        }

        /**
         * @brief SpMV for SELL format
         * @param matrix SELL sparse matrix
         * @param x Input vector
         * @param[out] y Output vector
         * @tparam T Element type
         * @def sparse_mv
         * @ingroup sparse_ops
         */
        template <typename T>
        void sparse_mv(const SparseMatrixSELL<T>& matrix, const T* x, T* y) {
            SparseOps<T>::spmv(matrix, x, y);
        }

    }  // namespace sparse
}  // namespace nova

#endif  // NOVA_CUDA_SPARSE_OPS_HPP
