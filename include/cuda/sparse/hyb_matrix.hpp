#ifndef NOVA_CUDA_SPARSE_HYB_MATRIX_HPP
#define NOVA_CUDA_SPARSE_HYB_MATRIX_HPP

#include "sparse_matrix.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace nova {
namespace sparse {

template<typename T>
class SparseMatrixHYB {
public:
    SparseMatrixHYB() = default;

    static SparseMatrixHYB FromCSR(const SparseMatrixCSR<T>& csr, int threshold_divisor = 2) {
        SparseMatrixHYB result;
        result.num_rows_ = csr.num_rows();
        result.num_cols_ = csr.num_cols();

        if (result.num_rows_ == 0 || csr.nnz() == 0) {
            return result;
        }

        int max_nnz_per_row = 0;
        for (int i = 0; i < result.num_rows_; ++i) {
            int row_nnz = csr.row_offsets()[i + 1] - csr.row_offsets()[i];
            max_nnz_per_row = std::max(max_nnz_per_row, row_nnz);
        }
        result.max_nnz_per_row_ = max_nnz_per_row;
        result.threshold_ = max_nnz_per_row / threshold_divisor;

        result.row_to_format_.resize(result.num_rows_, 1);

        result.ell_row_count_ = 0;
        std::vector<int> ell_row_to_csr(result.num_rows_);

        for (int i = 0; i < result.num_rows_; ++i) {
            int row_nnz = csr.row_offsets()[i + 1] - csr.row_offsets()[i];
            if (row_nnz > result.threshold_) {
                result.row_to_format_[i] = 0;
                ell_row_to_csr[result.ell_row_count_] = i;
                ++result.ell_row_count_;
            }
        }

        result.values_ell_.resize(result.ell_row_count_ * max_nnz_per_row, T{0});
        result.col_indices_ell_.resize(result.ell_row_count_ * max_nnz_per_row, -1);
        result.row_offsets_ell_.resize(result.ell_row_count_ + 1);
        result.row_offsets_ell_[0] = 0;
        for (int i = 0; i < result.ell_row_count_; ++i) {
            result.row_offsets_ell_[i + 1] = result.row_offsets_ell_[i] + max_nnz_per_row;
        }

        for (int ell_row = 0; ell_row < result.ell_row_count_; ++ell_row) {
            int csr_row = ell_row_to_csr[ell_row];
            int csr_start = csr.row_offsets()[csr_row];
            int csr_end = csr.row_offsets()[csr_row + 1];
            int base = ell_row * max_nnz_per_row;

            for (int j = csr_start; j < csr_end; ++j) {
                result.values_ell_[base + (j - csr_start)] = csr.values()[j];
                result.col_indices_ell_[base + (j - csr_start)] = csr.col_indices()[j];
            }
        }

        int coo_nnz = 0;
        for (int i = 0; i < result.num_rows_; ++i) {
            if (result.row_to_format_[i] == 1) {
                coo_nnz += csr.row_offsets()[i + 1] - csr.row_offsets()[i];
            }
        }

        result.values_coo_.reserve(coo_nnz);
        result.row_coo_.reserve(coo_nnz);
        result.col_coo_.reserve(coo_nnz);

        for (int i = 0; i < result.num_rows_; ++i) {
            if (result.row_to_format_[i] == 1) {
                int csr_start = csr.row_offsets()[i];
                int csr_end = csr.row_offsets()[i + 1];

                for (int j = csr_start; j < csr_end; ++j) {
                    result.values_coo_.push_back(csr.values()[j]);
                    result.row_coo_.push_back(i);
                    result.col_coo_.push_back(csr.col_indices()[j]);
                }
            }
        }

        result.ell_nnz_ = 0;
        for (int i = 0; i < result.ell_row_count_; ++i) {
            int base = i * max_nnz_per_row;
            for (int j = 0; j < max_nnz_per_row; ++j) {
                if (result.col_indices_ell_[base + j] >= 0) {
                    ++result.ell_nnz_;
                }
            }
        }

        return result;
    }

    int num_rows() const { return num_rows_; }
    int num_cols() const { return num_cols_; }
    int nnz() const { return ell_nnz_ + static_cast<int>(values_coo_.size()); }
    int ell_row_count() const { return ell_row_count_; }
    int coo_row_count() const { return num_rows_ - ell_row_count_; }
    int max_nnz_per_row() const { return max_nnz_per_row_; }
    int threshold() const { return threshold_; }
    int ell_nnz() const { return ell_nnz_; }
    int coo_nnz() const { return static_cast<int>(values_coo_.size()); }

    const T* ell_values() const { return values_ell_.data(); }
    const int* ell_col_indices() const { return col_indices_ell_.data(); }
    const int* ell_row_offsets() const { return row_offsets_ell_.data(); }

    const T* coo_values() const { return values_coo_.data(); }
    const int* coo_row_indices() const { return row_coo_.data(); }
    const int* coo_col_indices() const { return col_coo_.data(); }

    bool is_ell_row(int row) const {
        return row >= 0 && row < num_rows_ && row_to_format_[row] == 0;
    }

    int get_ell_row_index(int csr_row) const {
        if (!is_ell_row(csr_row)) return -1;
        int ell_idx = 0;
        for (int i = 0; i < csr_row; ++i) {
            if (row_to_format_[i] == 0) ++ell_idx;
        }
        return ell_idx;
    }

private:
    std::vector<T> values_ell_;
    std::vector<int> col_indices_ell_;
    std::vector<int> row_offsets_ell_;
    std::vector<T> values_coo_;
    std::vector<int> row_coo_;
    std::vector<int> col_coo_;
    std::vector<char> row_to_format_;
    int ell_row_count_ = 0;
    int ell_nnz_ = 0;
    int num_rows_ = 0;
    int num_cols_ = 0;
    int max_nnz_per_row_ = 0;
    int threshold_ = 0;
};

template<typename T>
void sparse_mv(const SparseMatrixHYB<T>& A, const T* x, T* y) {
    int num_rows = A.num_rows();

    for (int i = 0; i < num_rows; ++i) {
        y[i] = T{0};
    }

    int max_nnz = A.max_nnz_per_row();
    const T* ell_values = A.ell_values();
    const int* ell_col = A.ell_col_indices();

    for (int csr_row = 0; csr_row < A.ell_row_count(); ++csr_row) {
        T sum = T{0};
        int base = csr_row * max_nnz;

        for (int j = 0; j < max_nnz; ++j) {
            int col = ell_col[base + j];
            if (col >= 0) {
                sum += ell_values[base + j] * x[col];
            }
        }

        int orig_row = -1;
        int count = 0;
        for (int i = 0; i < num_rows; ++i) {
            if (A.is_ell_row(i)) {
                if (count == csr_row) {
                    orig_row = i;
                    break;
                }
                ++count;
            }
        }
        if (orig_row >= 0) {
            y[orig_row] = sum;
        }
    }

    int coo_nnz = A.coo_nnz();
    const T* coo_values = A.coo_values();
    const int* coo_rows = A.coo_row_indices();
    const int* coo_cols = A.coo_col_indices();

    for (int i = 0; i < coo_nnz; ++i) {
        y[coo_rows[i]] += coo_values[i] * x[coo_cols[i]];
    }
}

}
}

#endif
