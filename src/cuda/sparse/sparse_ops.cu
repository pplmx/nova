#include <cusparse.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include <vector>

#include "cuda/sparse/matrix.hpp"
#include "cuda/sparse/cusparse_context.hpp"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "cuda/device/error.h"

namespace nova::sparse {

namespace detail {

template<typename T>
struct CusparseTraits;

template<>
struct CusparseTraits<float> {
    static constexpr cudaDataType_t type = CUDA_R_32F;
};

template<>
struct CusparseTraits<double> {
    static constexpr cudaDataType_t type = CUDA_R_64F;
};

template<typename T>
void spmv_impl(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream) {
    auto& ctx = CusparseContext::get();
    if (stream) {
        ctx.set_stream(stream);
    }

    cusparseSpMatDescr_t mat_desc;
    cusparseDnVecDescr_t vec_x_desc;
    cusparseDnVecDescr_t vec_y_desc;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_desc,
        A.rows(), A.cols(), A.nnz(),
        const_cast<int*>(A.row_offsets()), const_cast<int*>(A.col_indices()),
        const_cast<T*>(A.values()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CusparseTraits<T>::type
    ));

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vec_x_desc,
        A.cols(), const_cast<T*>(x),
        CusparseTraits<T>::type
    ));

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vec_y_desc,
        A.rows(), y,
        CusparseTraits<T>::type
    ));

    T alpha = T{1};
    T beta = T{0};

    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        ctx.handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_desc, vec_x_desc, &beta, vec_y_desc,
        CusparseTraits<T>::type,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &buffer_size
    ));

    cuda::memory::Buffer<void> buffer(buffer_size);

    CUSPARSE_CHECK(cusparseSpMV(
        ctx.handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_desc, vec_x_desc, &beta, vec_y_desc,
        CusparseTraits<T>::type,
        CUSPARSE_SPMV_ALG_DEFAULT,
        buffer.data()
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUSPARSE_CHECK(cusparseDestroySpMat(mat_desc));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x_desc));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y_desc));
}

template<typename T>
void spmv_transpose_impl(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream) {
    auto& ctx = CusparseContext::get();
    if (stream) {
        ctx.set_stream(stream);
    }

    cusparseSpMatDescr_t mat_desc;
    cusparseDnVecDescr_t vec_x_desc;
    cusparseDnVecDescr_t vec_y_desc;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_desc,
        A.rows(), A.cols(), A.nnz(),
        const_cast<int*>(A.row_offsets()), const_cast<int*>(A.col_indices()),
        const_cast<T*>(A.values()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CusparseTraits<T>::type
    ));

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vec_x_desc,
        A.rows(), const_cast<T*>(x),
        CusparseTraits<T>::type
    ));

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vec_y_desc,
        A.cols(), y,
        CusparseTraits<T>::type
    ));

    T alpha = T{1};
    T beta = T{0};

    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        ctx.handle(),
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, mat_desc, vec_x_desc, &beta, vec_y_desc,
        CusparseTraits<T>::type,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &buffer_size
    ));

    cuda::memory::Buffer<void> buffer(buffer_size);

    CUSPARSE_CHECK(cusparseSpMV(
        ctx.handle(),
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, mat_desc, vec_x_desc, &beta, vec_y_desc,
        CusparseTraits<T>::type,
        CUSPARSE_SPMV_ALG_DEFAULT,
        buffer.data()
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUSPARSE_CHECK(cusparseDestroySpMat(mat_desc));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x_desc));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y_desc));
}

}  // namespace detail

template<typename T>
void spmv(const SparseMatrix<T>& A, const T* x, T* y) {
    detail::spmv_impl(A, x, y, nullptr);
}

template<typename T>
void spmv(const SparseMatrix<T>& A, const std::vector<T>& x, std::vector<T>& y) {
    cuda::memory::Buffer<T> d_x(A.cols());
    cuda::memory::Buffer<T> d_y(A.rows());

    d_x.copy_from(x.data(), static_cast<size_t>(A.cols()));
    detail::spmv_impl(A, d_x.data(), d_y.data(), nullptr);
    d_y.copy_to(y.data(), static_cast<size_t>(A.rows()));
}

template<typename T>
void spmv_async(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream) {
    detail::spmv_impl(A, x, y, stream);
}

template<typename T>
void spmv_transpose(const SparseMatrix<T>& A, const T* x, T* y) {
    detail::spmv_transpose_impl(A, x, y, nullptr);
}

template<typename T>
void spmv_transpose_async(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream) {
    detail::spmv_transpose_impl(A, x, y, stream);
}

template void spmv<float>(const SparseMatrix<float>&, const float*, float*);
template void spmv<double>(const SparseMatrix<double>&, const double*, double*);
template void spmv<float>(const SparseMatrix<float>&, const std::vector<float>&, std::vector<float>&);
template void spmv<double>(const SparseMatrix<double>&, const std::vector<double>&, std::vector<double>&);
template void spmv_async<float>(const SparseMatrix<float>&, const float*, float*, cudaStream_t);
template void spmv_async<double>(const SparseMatrix<double>&, const double*, double*, cudaStream_t);
template void spmv_transpose<float>(const SparseMatrix<float>&, const float*, float*);
template void spmv_transpose<double>(const SparseMatrix<double>&, const double*, double*);
template void spmv_transpose_async<float>(const SparseMatrix<float>&, const float*, float*, cudaStream_t);
template void spmv_transpose_async<double>(const SparseMatrix<double>&, const double*, double*, cudaStream_t);

}  // namespace nova::sparse
