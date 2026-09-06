#include "cuda/algo/spmv.h"

#include <algorithm>
#include <cuda_runtime.h>

#include "cuda/device/error.h"

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
        const int col_start = col_offsets[col];
        const int col_end = col_offsets[col + 1];

        // CSC SpMV: y = A*x. Each nonzero (row_indices[i], values[i]) in column
        // `col` contributes values[i]*x[col] to y[row_indices[i]]. The prior
        // implementation accumulated into y[col] and never read row_indices,
        // producing wrong results (e.g. y[col] = sum(values of column col)*x[col]).
        // atomicAdd is required because multiple columns can write the same row.
        for (int i = col_start; i < col_end; ++i) {
            const int row = row_indices[i];
            atomicAdd(&y[row], values[i] * x[col]);
        }
    }
}

template <typename T>
void multiply_csr(const T* values, const int* row_offsets, const int* col_indices,
                  const T* x, T* y, int num_rows, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (num_rows + block_size - 1) / block_size;

    spmv_csr_kernel<T><<<num_blocks, block_size, 0, stream>>>(
        values, row_offsets, col_indices, x, y, num_rows);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void multiply_csc(const T* values, const int* col_offsets, const int* row_indices,
                  const T* x, T* y, int num_rows, int num_cols, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = std::max(1, (num_cols + block_size - 1) / block_size);

    // The kernel scatters with atomicAdd into y[row] for row in [0, num_rows),
    // so the whole output must start at zero — num_rows entries, not num_cols.
    // With the old num_cols-only memset, a reader with more rows than columns
    // accumulated into uninitialized trailing rows (and cols > rows memset past
    // the buffer).
    CUDA_CHECK(cudaMemsetAsync(y, 0, static_cast<size_t>(num_rows) * sizeof(T), stream));

    spmv_csc_kernel<T><<<num_blocks, block_size, 0, stream>>>(
        values, col_offsets, row_indices, x, y, num_cols);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void multiply(const T* values, const int* offsets, const int* indices,
              const T* x, T* y, int num_rows_or_cols, Format format,
              cudaStream_t stream) {
    if (format == Format::CSR) {
        multiply_csr(values, offsets, indices, x, y, num_rows_or_cols, stream);
    } else {
        // The single-dimension multiply() implies a square matrix, so num_rows
        // == num_cols == num_rows_or_cols; callers with rectangular CSC
        // matrices should use multiply_csc() directly.
        multiply_csc(values, offsets, indices, x, y, num_rows_or_cols, num_rows_or_cols, stream);
    }
}

template void multiply_csr<float>(const float*, const int*, const int*, const float*, float*, int, cudaStream_t);
template void multiply_csr<double>(const double*, const int*, const int*, const double*, double*, int, cudaStream_t);

template void multiply_csc<float>(const float*, const int*, const int*, const float*, float*, int, int, cudaStream_t);
template void multiply_csc<double>(const double*, const int*, const int*, const double*, double*, int, int, cudaStream_t);

template void multiply<float>(const float*, const int*, const int*, const float*, float*, int, Format, cudaStream_t);
template void multiply<double>(const double*, const int*, const int*, const double*, double*, int, Format, cudaStream_t);

}  // namespace cuda::algo::spmv
