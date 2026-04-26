#ifndef NOVA_CUDA_SPARSE_MATRIX_HPP
#define NOVA_CUDA_SPARSE_MATRIX_HPP

#include <nova/memory/buffer.hpp>
#include <vector>
#include <memory>
#include <optional>

namespace nova {
namespace sparse {

enum class SparseFormat { CSR, CSC };

template<typename T>
class SparseMatrixCSR {
public:
    SparseMatrixCSR() = default;

    SparseMatrixCSR(std::vector<T> values, std::vector<int> row_offsets,
                    std::vector<int> col_indices, int num_rows, int num_cols)
        : values_(std::move(values))
        , row_offsets_(std::move(row_offsets))
        , col_indices_(std::move(col_indices))
        , num_rows_(num_rows)
        , num_cols_(num_cols) {}

    static std::optional<SparseMatrixCSR<T>> FromDense(const T* dense, int rows, int cols,
                                                       float sparsity_threshold = 0.0f);

    int num_rows() const { return num_rows_; }
    int num_cols() const { return num_cols_; }
    int nnz() const { return static_cast<int>(values_.size()); }

    const T* values() const { return values_.data(); }
    const int* row_offsets() const { return row_offsets_.data(); }
    const int* col_indices() const { return col_indices_.data(); }

    T* values() { return values_.data(); }
    int* row_offsets() { return row_offsets_.data(); }
    int* col_indices() { return col_indices_.data(); }

private:
    std::vector<T> values_;
    std::vector<int> row_offsets_;
    std::vector<int> col_indices_;
    int num_rows_ = 0;
    int num_cols_ = 0;
};

template<typename T>
class SparseMatrixCSC {
public:
    SparseMatrixCSC() = default;

    SparseMatrixCSC(std::vector<T> values, std::vector<int> col_offsets,
                    std::vector<int> row_indices, int num_rows, int num_cols)
        : values_(std::move(values))
        , col_offsets_(std::move(col_offsets))
        , row_indices_(std::move(row_indices))
        , num_rows_(num_rows)
        , num_cols_(num_cols) {}

    int num_rows() const { return num_rows_; }
    int num_cols() const { return num_cols_; }
    int nnz() const { return static_cast<int>(values_.size()); }

    const T* values() const { return values_.data(); }
    const int* col_offsets() const { return col_offsets_.data(); }
    const int* row_indices() const { return row_indices_.data(); }

    static SparseMatrixCSC<T> FromCSR(const SparseMatrixCSR<T>& csr);

private:
    std::vector<T> values_;
    std::vector<int> col_offsets_;
    std::vector<int> row_indices_;
    int num_rows_ = 0;
    int num_cols_ = 0;
};

template<typename T>
std::optional<SparseMatrixCSR<T>> SparseMatrixCSR<T>::FromDense(const T* dense,
                                                                  int rows, int cols,
                                                                  float threshold) {
    std::vector<T> values;
    std::vector<int> row_offsets(1, 0);
    std::vector<int> col_indices;

    for (int i = 0; i < rows; ++i) {
        int row_nnz = 0;
        for (int j = 0; j < cols; ++j) {
            T val = dense[i * cols + j];
            if (val != T{0}) {
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

    return SparseMatrixCSR<T>(std::move(values), std::move(row_offsets),
                              std::move(col_indices), rows, cols);
}

template<typename T>
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

    return SparseMatrixCSC<T>(std::move(values), std::move(col_offsets),
                              std::move(row_indices), rows, cols);
}

} // namespace sparse
} // namespace nova

#endif // NOVA_CUDA_SPARSE_MATRIX_HPP
