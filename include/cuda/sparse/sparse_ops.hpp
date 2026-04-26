#ifndef NOVA_CUDA_SPARSE_OPS_HPP
#define NOVA_CUDA_SPARSE_OPS_HPP

#include <nova/sparse/sparse_matrix.hpp>
#include <memory>

namespace nova {
namespace sparse {

template<typename T>
void sparse_mv(const SparseMatrixCSR<T>& matrix, const T* x, T* y);

template<typename T>
void sparse_mm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_vecs);

template<typename T>
class SparseOps {
public:
    static void spmv(const SparseMatrixCSR<T>& matrix, const T* x, T* y);
    static void spmm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_cols);
};

template<typename T>
void SparseOps<T>::spmv(const SparseMatrixCSR<T>& matrix, const T* x, T* y) {
    int num_rows = matrix.num_rows();

    for (int i = 0; i < num_rows; ++i) {
        T sum = T{0};
        for (int idx = matrix.row_offsets()[i]; idx < matrix.row_offsets()[i + 1]; ++idx) {
            int col = matrix.col_indices()[idx];
            sum += matrix.values()[idx] * x[col];
        }
        y[i] = sum;
    }
}

template<typename T>
void SparseOps<T>::spmm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_cols) {
    int num_rows = matrix.num_rows();

    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            T sum = T{0};
            for (int idx = matrix.row_offsets()[i]; idx < matrix.row_offsets()[i + 1]; ++idx) {
                int col = matrix.col_indices()[idx];
                sum += matrix.values()[idx] * B[col * num_cols + j];
            }
            C[i * num_cols + j] = sum;
        }
    }
}

template<typename T>
void sparse_mv(const SparseMatrixCSR<T>& matrix, const T* x, T* y) {
    SparseOps<T>::spmv(matrix, x, y);
}

template<typename T>
void sparse_mm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_vecs) {
    SparseOps<T>::spmm(matrix, B, C, num_vecs);
}

} // namespace sparse
} // namespace nova

#endif // NOVA_CUDA_SPARSE_OPS_HPP
