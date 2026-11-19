/**
 * @file sparse_matrix.hpp
 * @brief Sparse matrix storage formats for GPU computation
 * @defgroup sparse_matrix Sparse Matrix Formats
 * @ingroup sparse
 *
 * This module provides multiple sparse matrix storage formats optimized
 * for different sparse matrix structures and GPU compute patterns.
 *
 * Supported formats:
 * - CSR (Compressed Sparse Row) - General purpose, good for matrix-vector products
 * - CSC (Compressed Sparse Column) - Good for matrix-vector products with column access
 * - ELL (Ellpack-Itpack) - Fast for regular sparsity patterns
 * - SELL (Sliced ELL) - Better load balancing for irregular patterns
 *
 * Example usage:
 * @code
 * // Convert dense to CSR
 * auto csr = SparseMatrixCSR<float>::FromDense(dense_data, rows, cols);
 *
 * // Convert between formats
 * auto csc = SparseMatrixCSC<float>::FromCSR(*csr);
 * auto ell = SparseMatrixELL<float>::FromCSR(*csr);
 * @endcode
 *
 * @see sparse_ops.hpp For sparse matrix operations
 * @see solver_workspace.hpp For iterative solvers
 */

#ifndef NOVA_CUDA_SPARSE_MATRIX_HPP
#define NOVA_CUDA_SPARSE_MATRIX_HPP

#include <algorithm>
#include <vector>
#include <memory>
#include <optional>

namespace nova {
    namespace sparse {

        /**
         * @brief Supported sparse matrix storage formats
         * @enum SparseFormat
         * @ingroup sparse_matrix
         */
        enum class SparseFormat { CSR, CSC, ELL, SELL, HYB };

        /**
         * @brief Compressed Sparse Row matrix format
         * @class SparseMatrixCSR
         * @ingroup sparse_matrix
         *
         * Stores sparse matrix in CSR format:
         * - values: Non-zero elements in row-major order
         * - row_offsets: Starting index of each row in values (size = rows + 1)
         * - col_indices: Column indices for each value (size = nnz)
         *
         * @note Time complexity: O(rows + cols + nnz) for construction
         * @note Space: O(nnz + rows) for typical sparse matrices
         *
         * @deprecated Use cuda::sparse::SparseMatrix<T> instead via ToSparseMatrix()
         * @see SparseMatrix For the new unified sparse matrix type
         */
        template <typename T>
        class [[deprecated("Use cuda::sparse::SparseMatrix<T> instead")]] SparseMatrixCSR {
        public:
            /** @brief Default constructor creates empty matrix */
            SparseMatrixCSR() = default;

            /**
             * @brief Construct CSR matrix from raw data
             * @param values Non-zero values in row-major order
             * @param row_offsets Row offset array (size = rows + 1)
             * @param col_indices Column indices (size = values.size())
             * @param num_rows Number of rows
             * @param num_cols Number of columns
             */
            SparseMatrixCSR(std::vector<T> values, std::vector<int> row_offsets, std::vector<int> col_indices, int num_rows, int num_cols)
                : values_(std::move(values)),
                  row_offsets_(std::move(row_offsets)),
                  col_indices_(std::move(col_indices)),
                  num_rows_(num_rows),
                  num_cols_(num_cols) {}

            /**
             * @brief Create CSR matrix from dense matrix
             * @param dense Pointer to dense matrix data (row-major)
             * @param rows Number of rows
             * @param cols Number of columns
             * @param sparsity_threshold Elements with |value| <= threshold treated as zero
             * @return CSR matrix, or nullopt if fully dense (nnz == 0)
             * @tparam T Numeric type (float, double, etc.)
             *
             * @note Time complexity: O(rows * cols)
             * @note Allocates O(nnz) temporary storage
             */
            static std::optional<SparseMatrixCSR<T>> FromDense(const T* dense, int rows, int cols, float sparsity_threshold = 0.0f);

            /**
             * @brief Get number of rows
             * @return Row count
             */
            int num_rows() const { return num_rows_; }

            /**
             * @brief Get number of columns
             * @return Column count
             */
            int num_cols() const { return num_cols_; }

            /**
             * @brief Get number of non-zeros
             * @return NNZ count
             */
            int nnz() const { return static_cast<int>(values_.size()); }

            /** @brief Get const pointer to values array */
            const T* values() const { return values_.data(); }

            /** @brief Get const pointer to row offset array */
            const int* row_offsets() const { return row_offsets_.data(); }

            /** @brief Get const pointer to column indices array */
            const int* col_indices() const { return col_indices_.data(); }

            /** @brief Get mutable pointer to values array */
            T* values() { return values_.data(); }

            /** @brief Get mutable pointer to row offset array */
            int* row_offsets() { return row_offsets_.data(); }

            /** @brief Get mutable pointer to column indices array */
            int* col_indices() { return col_indices_.data(); }

        private:
            std::vector<T> values_;
            std::vector<int> row_offsets_;
            std::vector<int> col_indices_;
            int num_rows_ = 0;
            int num_cols_ = 0;
        };

        /**
         * @brief Compressed Sparse Column matrix format
         * @class SparseMatrixCSC
         * @ingroup sparse_matrix
         *
         * Stores sparse matrix in CSC format:
         * - values: Non-zero elements in column-major order
         * - col_offsets: Starting index of each column in values (size = cols + 1)
         * - row_indices: Row indices for each value (size = nnz)
         *
         * @note Good for operations that access columns efficiently
         * @see SparseMatrixCSR For row-oriented access patterns
         */
        template <typename T>
        class SparseMatrixCSC {
        public:
            /** @brief Default constructor creates empty matrix */
            SparseMatrixCSC() = default;

            /**
             * @brief Construct CSC matrix from raw data
             * @param values Non-zero values in column-major order
             * @param col_offsets Column offset array (size = cols + 1)
             * @param row_indices Row indices (size = values.size())
             * @param num_rows Number of rows
             * @param num_cols Number of columns
             */
            SparseMatrixCSC(std::vector<T> values, std::vector<int> col_offsets, std::vector<int> row_indices, int num_rows, int num_cols)
                : values_(std::move(values)),
                  col_offsets_(std::move(col_offsets)),
                  row_indices_(std::move(row_indices)),
                  num_rows_(num_rows),
                  num_cols_(num_cols) {}

            /** @brief Get number of rows */
            int num_rows() const { return num_rows_; }

            /** @brief Get number of columns */
            int num_cols() const { return num_cols_; }

            /** @brief Get number of non-zeros */
            int nnz() const { return static_cast<int>(values_.size()); }

            /** @brief Get const pointer to values array */
            const T* values() const { return values_.data(); }

            /** @brief Get const pointer to column offset array */
            const int* col_offsets() const { return col_offsets_.data(); }

            /** @brief Get const pointer to row indices array */
            const int* row_indices() const { return row_indices_.data(); }

            /**
             * @brief Create CSC matrix from CSR matrix
             * @param csr Source CSR matrix
             * @return CSC matrix representation
             *
             * @note Time complexity: O(rows + cols + nnz)
             */
            static SparseMatrixCSC<T> FromCSR(const SparseMatrixCSR<T>& csr);

        private:
            std::vector<T> values_;
            std::vector<int> col_offsets_;
            std::vector<int> row_indices_;
            int num_rows_ = 0;
            int num_cols_ = 0;
        };

        template <typename T>
        std::optional<SparseMatrixCSR<T>> SparseMatrixCSR<T>::FromDense(const T* dense, int rows, int cols, float threshold) {
            std::vector<T> values;
            std::vector<int> row_offsets(1, 0);
            std::vector<int> col_indices;

            for (int i = 0; i < rows; ++i) {
                int row_nnz = 0;
                for (int j = 0; j < cols; ++j) {
                    T val = dense[i * cols + j];
                    if (val != T {0}) {
                        values.push_back(val);
                        col_indices.push_back(j);
                        ++row_nnz;
                    }
                }
                row_offsets.push_back(static_cast<int>(values.size()));
            }

            if (values.empty()) {
                return std::nullopt;
            }

            return SparseMatrixCSR<T>(std::move(values), std::move(row_offsets), std::move(col_indices), rows, cols);
        }

        template <typename T>
        SparseMatrixCSC<T> SparseMatrixCSC<T>::FromCSR(const SparseMatrixCSR<T>& csr) {
            int nnz = csr.nnz();
            int rows = csr.num_rows();
            int cols = csr.num_cols();

            std::vector<T> values;
            std::vector<int> row_indices;
            std::vector<int> col_offsets(cols + 1, 0);

            std::vector<int> temp_col_count(cols, 0);
            for (int i = 0; i < nnz; ++i) {
                ++temp_col_count[csr.col_indices()[i]];
            }

            for (int j = 1; j <= cols; ++j) {
                col_offsets[j] = col_offsets[j - 1] + temp_col_count[j - 1];
            }

            values.resize(nnz);
            row_indices.resize(nnz);
            std::vector<int> write_pos = col_offsets;

            for (int i = 0; i < rows; ++i) {
                for (int idx = csr.row_offsets()[i]; idx < csr.row_offsets()[i + 1]; ++idx) {
                    int col = csr.col_indices()[idx];
                    int write_idx = write_pos[col]++;
                    values[write_idx] = csr.values()[idx];
                    row_indices[write_idx] = i;
                }
            }

            return SparseMatrixCSC<T>(std::move(values), std::move(col_offsets), std::move(row_indices), rows, cols);
        }

        /**
         * @brief ELL (Ellpack-Itpack) sparse matrix format
         * @class SparseMatrixELL
         * @ingroup sparse_matrix
         *
         * Stores sparse matrix with fixed-width columns per row (padded to max_nnz).
         * - values: Padded values array (rows * max_nnz_per_row)
         * - col_indices: Padded column indices (-1 for padding)
         * - row_offsets: Starting index for each row (computed: i * max_nnz_per_row)
         *
         * @note Fast for GPUs due to regular memory access patterns
         * @note Wastes space when max_nnz_per_row is much larger than average
         * @note Time complexity: O(rows * max_nnz_per_row) for SpMV
         */
        template <typename T>
        class SparseMatrixELL {
        public:
            /** @brief Default constructor creates empty matrix */
            SparseMatrixELL() = default;

            /**
             * @brief Construct ELL matrix from raw data
             * @param values Padded values (rows * max_nnz_per_row)
             * @param col_indices Padded column indices (-1 for padding)
             * @param num_rows Number of rows
             * @param num_cols Number of columns
             * @param max_nnz_per_row Maximum non-zeros per row
             */
            SparseMatrixELL(std::vector<T> values, std::vector<int> col_indices, int num_rows, int num_cols, int max_nnz_per_row)
                : values_(std::move(values)),
                  col_indices_(std::move(col_indices)),
                  row_offsets_(num_rows + 1),
                  num_rows_(num_rows),
                  num_cols_(num_cols),
                  max_nnz_per_row_(max_nnz_per_row) {
                for (int i = 0; i <= num_rows; ++i) {
                    row_offsets_[i] = i * max_nnz_per_row;
                }
            }

            /**
             * @brief Create ELL matrix from CSR matrix
             * @param csr Source CSR matrix
             * @return ELL matrix representation
             *
             * @note Time complexity: O(rows + cols + nnz)
             * @note Space: O(rows * max_nnz_per_row)
             */
            static SparseMatrixELL<T> FromCSR(const SparseMatrixCSR<T>& csr);

            /** @brief Get number of rows */
            int num_rows() const { return num_rows_; }

            /** @brief Get number of columns */
            int num_cols() const { return num_cols_; }

            /** @brief Get actual number of non-zeros (excluding padding) */
            int nnz() const { return static_cast<int>(values_.size()) - (num_rows_ * max_nnz_per_row_ - count_nnz()); }

            /** @brief Get total padded storage (including zeros) */
            int padded_nnz() const { return num_rows_ * max_nnz_per_row_; }

            /** @brief Get maximum non-zeros per row (padding width) */
            int max_nnz_per_row() const { return max_nnz_per_row_; }

            /** @brief Get const pointer to values array */
            const T* values() const { return values_.data(); }

            /** @brief Get const pointer to column indices array */
            const int* col_indices() const { return col_indices_.data(); }

            /** @brief Get const pointer to row offset array */
            const int* row_offsets() const { return row_offsets_.data(); }

            /** @brief Get mutable pointer to values array */
            T* values() { return values_.data(); }

            /** @brief Get mutable pointer to column indices array */
            int* col_indices() { return col_indices_.data(); }

        private:
            int count_nnz() const;

            std::vector<T> values_;
            std::vector<int> col_indices_;
            std::vector<int> row_offsets_;
            int num_rows_ = 0;
            int num_cols_ = 0;
            int max_nnz_per_row_ = 0;
        };

        /**
         * @brief Sliced ELL (SELL) sparse matrix format
         * @class SparseMatrixSELL
         * @ingroup sparse_matrix
         *
         * Variant of ELL that slices rows into groups for better load balancing.
         * Each slice has its own max_nnz, reducing padding waste.
         *
         * @note Better for irregular sparsity patterns than standard ELL
         * @note Slice height controls tradeoff between padding and launch overhead
         * @see SparseMatrixELL For simpler regular-sparsity use cases
         */
        template <typename T>
        class SparseMatrixSELL {
        public:
            /** @brief Default constructor creates empty matrix */
            SparseMatrixSELL() = default;

            /**
             * @brief Construct SELL matrix from raw data
             * @param values Padded values array
             * @param col_indices Padded column indices (-1 for padding)
             * @param slice_ptr Slice boundaries (size = num_slices + 1)
             * @param num_rows Number of rows
             * @param num_cols Number of columns
             * @param slice_height Number of rows per slice
             */
            SparseMatrixSELL(std::vector<T> values, std::vector<int> col_indices, std::vector<int> slice_ptr, int num_rows, int num_cols, int slice_height)
                : values_(std::move(values)),
                  col_indices_(std::move(col_indices)),
                  slice_ptr_(std::move(slice_ptr)),
                  num_rows_(num_rows),
                  num_cols_(num_cols),
                  slice_height_(slice_height) {}

            /**
             * @brief Create SELL matrix from CSR matrix
             * @param csr Source CSR matrix
             * @param slice_height Rows per slice (default: 32)
             * @return SELL matrix representation
             *
             * @note Time complexity: O(rows + cols + nnz)
             * @note slice_height=32 is typically optimal for CUDA
             */
            static SparseMatrixSELL<T> FromCSR(const SparseMatrixCSR<T>& csr, int slice_height = 32);

            /** @brief Get number of rows */
            int num_rows() const { return num_rows_; }

            /** @brief Get number of columns */
            int num_cols() const { return num_cols_; }

            /** @brief Get actual number of non-zeros */
            int nnz() const { return count_nnz(); }

            /** @brief Get total padded storage */
            int padded_nnz() const { return static_cast<int>(values_.size()); }

            /** @brief Get slice height */
            int slice_height() const { return slice_height_; }

            /** @brief Get const pointer to values array */
            const T* values() const { return values_.data(); }

            /** @brief Get const pointer to column indices array */
            const int* col_indices() const { return col_indices_.data(); }

            /** @brief Get const pointer to slice pointer array */
            const int* slice_ptr() const { return slice_ptr_.data(); }

            /** @brief Get mutable pointer to values array */
            T* values() { return values_.data(); }

            /** @brief Get mutable pointer to column indices array */
            int* col_indices() { return col_indices_.data(); }

        private:
            int count_nnz() const;

            std::vector<T> values_;
            std::vector<int> col_indices_;
            std::vector<int> slice_ptr_;
            int num_rows_ = 0;
            int num_cols_ = 0;
            int slice_height_ = 32;
        };

        template <typename T>
        int SparseMatrixELL<T>::count_nnz() const {
            int count = 0;
            for (int i = 0; i < padded_nnz(); ++i) {
                if (col_indices_[i] >= 0 && values_[i] != T {0}) {
                    ++count;
                }
            }
            return count;
        }

        template <typename T>
        int SparseMatrixSELL<T>::count_nnz() const {
            int count = 0;
            for (int i = 0; i < padded_nnz(); ++i) {
                if (col_indices_[i] >= 0 && values_[i] != T {0}) {
                    ++count;
                }
            }
            return count;
        }

        template <typename T>
        SparseMatrixELL<T> SparseMatrixELL<T>::FromCSR(const SparseMatrixCSR<T>& csr) {
            int num_rows = csr.num_rows();
            int num_cols = csr.num_cols();

            int max_nnz = 0;
            for (int i = 0; i < num_rows; ++i) {
                int row_nnz = csr.row_offsets()[i + 1] - csr.row_offsets()[i];
                max_nnz = std::max(max_nnz, row_nnz);
            }

            if (max_nnz == 0) {
                return SparseMatrixELL<T>();
            }

            std::vector<T> values(num_rows * max_nnz, T {0});
            std::vector<int> col_indices(num_rows * max_nnz, -1);

            for (int i = 0; i < num_rows; ++i) {
                int csr_start = csr.row_offsets()[i];
                int csr_end = csr.row_offsets()[i + 1];
                int ell_base = i * max_nnz;

                for (int j = csr_start; j < csr_end; ++j) {
                    values[ell_base + (j - csr_start)] = csr.values()[j];
                    col_indices[ell_base + (j - csr_start)] = csr.col_indices()[j];
                }
            }

            return SparseMatrixELL<T>(std::move(values), std::move(col_indices), num_rows, num_cols, max_nnz);
        }

        template <typename T>
        SparseMatrixSELL<T> SparseMatrixSELL<T>::FromCSR(const SparseMatrixCSR<T>& csr, int slice_height) {
            int num_rows = csr.num_rows();
            int num_cols = csr.num_cols();

            if (num_rows == 0 || slice_height <= 0) {
                return SparseMatrixSELL<T>();
            }

            int num_slices = (num_rows + slice_height - 1) / slice_height;

            std::vector<int> slice_max_nnz(num_slices, 0);
            for (int i = 0; i < num_rows; ++i) {
                int slice_idx = i / slice_height;
                int row_nnz = csr.row_offsets()[i + 1] - csr.row_offsets()[i];
                slice_max_nnz[slice_idx] = std::max(slice_max_nnz[slice_idx], row_nnz);
            }

            std::vector<int> slice_ptr(num_slices + 1, 0);
            for (int s = 0; s < num_slices; ++s) {
                slice_ptr[s + 1] = slice_ptr[s] + slice_max_nnz[s] * slice_height;
            }
            int total_padded_nnz = slice_ptr[num_slices];

            if (total_padded_nnz == 0) {
                return SparseMatrixSELL<T>({}, {}, std::move(slice_ptr), num_rows, num_cols, slice_height);
            }

            std::vector<T> values(total_padded_nnz, T {0});
            std::vector<int> col_indices(total_padded_nnz, -1);

            for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
                int slice_start_row = slice_idx * slice_height;
                int slice_end_row = std::min(slice_start_row + slice_height, num_rows);
                int local_base = slice_ptr[slice_idx];
                int local_max_nnz = slice_max_nnz[slice_idx];

                for (int i = slice_start_row; i < slice_end_row; ++i) {
                    int csr_start = csr.row_offsets()[i];
                    int csr_end = csr.row_offsets()[i + 1];
                    int local_row = i - slice_start_row;

                    for (int j = csr_start; j < csr_end; ++j) {
                        int ell_idx = local_base + local_row * local_max_nnz + (j - csr_start);
                        values[ell_idx] = csr.values()[j];
                        col_indices[ell_idx] = csr.col_indices()[j];
                    }
                }
            }

            return SparseMatrixSELL<T>(std::move(values), std::move(col_indices), std::move(slice_ptr), num_rows, num_cols, slice_height);
        }

        /**
         * @brief Forward declaration of unified sparse matrix type
         * @class SparseMatrix
         * @tparam T Element type
         * @ingroup sparse
         *
         * @see ToSparseMatrix() For conversion from legacy CSR
         */
        template <typename T>
        class SparseMatrix;

        /**
         * @brief Convert legacy SparseMatrixCSR to new unified SparseMatrix
         * @param csr Legacy CSR matrix
         * @return Unified sparse matrix representation
         * @tparam T Element type
         * @ingroup sparse
         *
         * @deprecated Use cuda::sparse::SparseMatrix<T> directly
         */
        template <typename T>
        SparseMatrix<T> ToSparseMatrix(const SparseMatrixCSR<T>& csr);

    }  // namespace sparse
}  // namespace nova

#endif  // NOVA_CUDA_SPARSE_MATRIX_HPP
