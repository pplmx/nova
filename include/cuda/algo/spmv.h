#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::algo::spmv {

enum class Format { CSR, CSC };

template <typename T>
void multiply_csr(const T* values, const int* row_offsets, const int* col_indices,
                  const T* x, T* y, int num_rows, cudaStream_t stream = nullptr);

// CSC SpMV: y = A*x for the num_cols-column matrix whose column offsets and
// row indices are given; the output y has num_rows entries (the kernel scatters
// with atomicAdd and zeros them first). num_rows is required so a rectangular
// matrix with more rows than columns zeroes the whole output — the old
// single-dimension signature memsets only num_cols entries, leaving the
// trailing rows uninitialized before accumulation.
template <typename T>
void multiply_csc(const T* values, const int* col_offsets, const int* row_indices,
                  const T* x, T* y, int num_rows, int num_cols, cudaStream_t stream = nullptr);

template <typename T>
void multiply(const T* values, const int* offsets, const int* indices,
              const T* x, T* y, int num_rows_or_cols, Format format,
              cudaStream_t stream = nullptr);

struct SpMVConfig {
    int vectorization_width = 4;
    int row_chunk_size = 64;
    bool warp_specialization = true;
};

void set_config(const SpMVConfig& config);
SpMVConfig get_config();

}  // namespace cuda::algo::spmv
