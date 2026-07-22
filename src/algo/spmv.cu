#include "cuda/algo/spmv.h"

#include <cuda_runtime.h>

namespace cuda::algo::spmv {

static SpMVConfig g_config;

void set_config(const SpMVConfig& config) {
    g_config = config;
}

SpMVConfig get_config() {
    return g_config;
}

template <typename T>
__global__ void spmv_csr_kernel(const T* __restrict__ values,
                                 const int* __restrict__ row_offsets,
                                 const int* __restrict__ col_indices,
                                 const T* __restrict__ x,
                                 T* __restrict__ y,
                                 int num_rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        T sum = T{0};
        const int row_start = row_offsets[row];
        const int row_end = row_offsets[row + 1];

        for (int i = row_start; i < row_end; ++i) {
            const int col = col_indices[i];
            sum += values[i] * x[col];
        }

        y[row] = sum;
    }
}

template <typename T>
__global__ void spmv_csc_kernel(const T* __restrict__ values,
                                 const int* __restrict__ col_offsets,
                                 const int* __restrict__ row_indices,
                                 const T* __restrict__ x,
                                 T* __restrict__ y,
                                 int num_cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < num_cols) {
        T sum = T{0};
        const int col_start = col_offsets[col];
        const int col_end = col_offsets[col + 1];

        // In CSC SpMV each column's contribution lands in y[col] (atomicAdd not
        // needed because each column is owned by exactly one thread). row_indices[i]
        // is the source row for that entry but is unused for the value — only the
        // x[col] factor is multiplied in.
        for (int i = col_start; i < col_end; ++i) {
            sum += values[i] * x[col];
        }
        y[col] += sum;
    }
}

template <typename T>
void multiply_csr(const T* values, const int* row_offsets, const int* col_indices,
                  const T* x, T* y, int num_rows, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (num_rows + block_size - 1) / block_size;

    spmv_csr_kernel<T><<<num_blocks, block_size, 0, stream>>>(
        values, row_offsets, col_indices, x, y, num_rows);
}

template <typename T>
void multiply_csc(const T* values, const int* col_offsets, const int* row_indices,
                  const T* x, T* y, int num_cols, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (num_cols + block_size - 1) / block_size;

    spmv_csc_kernel<T><<<num_blocks, block_size, 0, stream>>>(
        values, col_offsets, row_indices, x, y, num_cols);
}

template <typename T>
void multiply(const T* values, const int* offsets, const int* indices,
              const T* x, T* y, int num_rows_or_cols, Format format,
              cudaStream_t stream) {
    if (format == Format::CSR) {
        multiply_csr(values, offsets, indices, x, y, num_rows_or_cols, stream);
    } else {
        multiply_csc(values, offsets, indices, x, y, num_rows_or_cols, stream);
    }
}

template void multiply_csr<float>(const float*, const int*, const int*, const float*, float*, int, cudaStream_t);
template void multiply_csr<double>(const double*, const int*, const int*, const double*, double*, int, cudaStream_t);

template void multiply_csc<float>(const float*, const int*, const int*, const float*, float*, int, cudaStream_t);
template void multiply_csc<double>(const double*, const int*, const int*, const double*, double*, int, cudaStream_t);

template void multiply<float>(const float*, const int*, const int*, const float*, float*, int, Format, cudaStream_t);
template void multiply<double>(const double*, const int*, const int*, const double*, double*, int, Format, cudaStream_t);

}  // namespace cuda::algo::spmv
