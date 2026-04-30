#ifndef NOVA_CUDA_SPARSE_MATRIX_HPP
#define NOVA_CUDA_SPARSE_MATRIX_HPP

#include <algorithm>
#include <vector>
#include <memory>
#include <optional>

namespace nova {
namespace sparse {

enum class SparseFormat { CSR, CSC, ELL, SELL, HYB };

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

template<typename T>
class SparseMatrixELL {
public:
    SparseMatrixELL() = default;

    SparseMatrixELL(std::vector<T> values, std::vector<int> col_indices,
                    int num_rows, int num_cols, int max_nnz_per_row)
        : values_(std::move(values))
        , col_indices_(std::move(col_indices))
        , row_offsets_(num_rows + 1)
        , num_rows_(num_rows)
        , num_cols_(num_cols)
        , max_nnz_per_row_(max_nnz_per_row) {
        for (int i = 0; i <= num_rows; ++i) {
            row_offsets_[i] = i * max_nnz_per_row;
        }
    }

    static SparseMatrixELL<T> FromCSR(const SparseMatrixCSR<T>& csr);

    int num_rows() const { return num_rows_; }
    int num_cols() const { return num_cols_; }
    int nnz() const { return static_cast<int>(values_.size()) - (num_rows_ * max_nnz_per_row_ - count_nnz()); }
    int padded_nnz() const { return num_rows_ * max_nnz_per_row_; }
    int max_nnz_per_row() const { return max_nnz_per_row_; }

    const T* values() const { return values_.data(); }
    const int* col_indices() const { return col_indices_.data(); }
    const int* row_offsets() const { return row_offsets_.data(); }

    T* values() { return values_.data(); }
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

template<typename T>
class SparseMatrixSELL {
public:
    SparseMatrixSELL() = default;

    SparseMatrixSELL(std::vector<T> values, std::vector<int> col_indices,
                     std::vector<int> slice_ptr, int num_rows, int num_cols, int slice_height)
        : values_(std::move(values))
        , col_indices_(std::move(col_indices))
        , slice_ptr_(std::move(slice_ptr))
        , num_rows_(num_rows)
        , num_cols_(num_cols)
        , slice_height_(slice_height) {}

    static SparseMatrixSELL<T> FromCSR(const SparseMatrixCSR<T>& csr, int slice_height = 32);

    int num_rows() const { return num_rows_; }
    int num_cols() const { return num_cols_; }
    int nnz() const { return count_nnz(); }
    int padded_nnz() const { return static_cast<int>(values_.size()); }
    int slice_height() const { return slice_height_; }

    const T* values() const { return values_.data(); }
    const int* col_indices() const { return col_indices_.data(); }
    const int* slice_ptr() const { return slice_ptr_.data(); }

    T* values() { return values_.data(); }
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

template<typename T>
int SparseMatrixELL<T>::count_nnz() const {
    int count = 0;
    for (int i = 0; i < padded_nnz(); ++i) {
        if (col_indices_[i] >= 0 && values_[i] != T{0}) {
            ++count;
        }
    }
    return count;
}

template<typename T>
int SparseMatrixSELL<T>::count_nnz() const {
    int count = 0;
    for (int i = 0; i < padded_nnz(); ++i) {
        if (col_indices_[i] >= 0 && values_[i] != T{0}) {
            ++count;
        }
    }
    return count;
}

template<typename T>
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

    std::vector<T> values(num_rows * max_nnz, T{0});
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

    return SparseMatrixELL<T>(std::move(values), std::move(col_indices),
                               num_rows, num_cols, max_nnz);
}

template<typename T>
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

    std::vector<T> values(total_padded_nnz, T{0});
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

    return SparseMatrixSELL<T>(std::move(values), std::move(col_indices),
                                std::move(slice_ptr), num_rows, num_cols, slice_height);
}

} // namespace sparse
} // namespace nova

#endif // NOVA_CUDA_SPARSE_MATRIX_HPP
