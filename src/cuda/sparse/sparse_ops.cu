#include <cusparse.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include <stdexcept>
#include <vector>

#include "cuda/sparse/matrix.hpp"
#include "cuda/sparse/cusparse_context.hpp"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "cuda/device/error.h"

namespace nova::sparse {

    namespace detail {

        template <typename T>
        struct CusparseTraits;

        template <>
        struct CusparseTraits<float> {
            static constexpr cudaDataType_t type = CUDA_R_32F;
        };

        template <>
        struct CusparseTraits<double> {
            static constexpr cudaDataType_t type = CUDA_R_64F;
        };

        template <typename T>
        void spmv_impl(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream) {
            auto& ctx = CusparseContext::get();
            if (stream) {
                ctx.set_stream(stream);
            }

            cusparseSpMatDescr_t mat_desc;
            cusparseDnVecDescr_t vec_x_desc;
            cusparseDnVecDescr_t vec_y_desc;

            CUSPARSE_CHECK(cusparseCreateCsr(&mat_desc,
                                             A.rows(),
                                             A.cols(),
                                             A.nnz(),
                                             const_cast<int*>(A.row_offsets()),
                                             const_cast<int*>(A.col_indices()),
                                             const_cast<T*>(A.values()),
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO,
                                             CusparseTraits<T>::type));

            CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x_desc, A.cols(), const_cast<T*>(x), CusparseTraits<T>::type));

            CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y_desc, A.rows(), y, CusparseTraits<T>::type));

            T alpha = T{1};
            T beta = T{0};

            // beta == 0 means y is overwritten, so its incoming contents are irrelevant to
            // the result — but cuSPARSE still reads it. Its vector_scalar_multiply fast path,
            // chosen for small matrices, loads y before scaling it by beta. Passing an
            // uninitialised buffer is harmless for the answer and misleading for any tool that
            // checks for uninitialised reads: compute-sanitizer --tool initcheck reported 22
            // such reads across the sparse suites, all of them here. Clearing y keeps the
            // contract honest at the cost of one synchronous memset.
            if (beta == T{0}) {
                CUDA_CHECK(cudaMemsetAsync(y, 0, static_cast<size_t>(A.rows()) * sizeof(T), stream));
            }

            size_t buffer_size = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                ctx.handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_desc, vec_x_desc, &beta, vec_y_desc, CusparseTraits<T>::type, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));

            cuda::memory::Buffer<void> buffer(buffer_size);

            CUSPARSE_CHECK(cusparseSpMV(ctx.handle(),
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        mat_desc,
                                        vec_x_desc,
                                        &beta,
                                        vec_y_desc,
                                        CusparseTraits<T>::type,
                                        CUSPARSE_SPMV_ALG_DEFAULT,
                                        buffer.data()));

            CUDA_CHECK(cudaDeviceSynchronize());

            CUSPARSE_CHECK(cusparseDestroySpMat(mat_desc));
            CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x_desc));
            CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y_desc));
        }

        template <typename T>
        void spmv_transpose_impl(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream) {
            auto& ctx = CusparseContext::get();
            if (stream) {
                ctx.set_stream(stream);
            }

            cusparseSpMatDescr_t mat_desc;
            cusparseDnVecDescr_t vec_x_desc;
            cusparseDnVecDescr_t vec_y_desc;

            CUSPARSE_CHECK(cusparseCreateCsr(&mat_desc,
                                             A.rows(),
                                             A.cols(),
                                             A.nnz(),
                                             const_cast<int*>(A.row_offsets()),
                                             const_cast<int*>(A.col_indices()),
                                             const_cast<T*>(A.values()),
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO,
                                             CusparseTraits<T>::type));

            CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x_desc, A.rows(), const_cast<T*>(x), CusparseTraits<T>::type));

            CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y_desc, A.cols(), y, CusparseTraits<T>::type));

            T alpha = T{1};
            T beta = T{0};

            // Same reasoning as spmv_impl: beta == 0 overwrites y, but cuSPARSE's
            // vector_scalar_multiply fast path still loads it first. The transpose writes
            // A.cols() outputs.
            if (beta == T{0}) {
                CUDA_CHECK(cudaMemsetAsync(y, 0, static_cast<size_t>(A.cols()) * sizeof(T), stream));
            }

            size_t buffer_size = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                ctx.handle(), CUSPARSE_OPERATION_TRANSPOSE, &alpha, mat_desc, vec_x_desc, &beta, vec_y_desc, CusparseTraits<T>::type, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));

            cuda::memory::Buffer<void> buffer(buffer_size);

            CUSPARSE_CHECK(cusparseSpMV(
                ctx.handle(), CUSPARSE_OPERATION_TRANSPOSE, &alpha, mat_desc, vec_x_desc, &beta, vec_y_desc, CusparseTraits<T>::type, CUSPARSE_SPMV_ALG_DEFAULT, buffer.data()));

            CUDA_CHECK(cudaDeviceSynchronize());

            CUSPARSE_CHECK(cusparseDestroySpMat(mat_desc));
            CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x_desc));
            CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y_desc));
        }

    }  // namespace detail

    template <typename T>
    void spmv(const SparseMatrix<T>& A, const T* x, T* y) {
        detail::spmv_impl(A, x, y, nullptr);
    }

    template <typename T>
    void spmv(const SparseMatrix<T>& A, const std::vector<T>& x, std::vector<T>& y) {
        // Reject a mismatched input instead of copying A.cols() elements out of a
        // shorter vector: that read past the host buffer and produced a
        // plausible-looking wrong answer with no error anywhere.
        if (x.size() != static_cast<size_t>(A.cols())) {
            throw std::invalid_argument("spmv: input vector has " + std::to_string(x.size()) + " elements but the matrix has " + std::to_string(A.cols()) + " columns");
        }

        // The output is this function's result, so size it here. Previously the
        // caller had to pre-size y exactly: an empty (or short) one made the
        // cudaMemcpy fail with "invalid argument", and a longer one kept its stale
        // tail past rows().
        y.resize(static_cast<size_t>(A.rows()));

        cuda::memory::Buffer<T> d_x(A.cols());
        cuda::memory::Buffer<T> d_y(A.rows());

        d_x.copy_from(x.data(), static_cast<size_t>(A.cols()));
        detail::spmv_impl(A, d_x.data(), d_y.data(), nullptr);
        d_y.copy_to(y.data(), static_cast<size_t>(A.rows()));
    }

    template <typename T>
    void spmv_async(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream) {
        detail::spmv_impl(A, x, y, stream);
    }

    template <typename T>
    void spmv_transpose(const SparseMatrix<T>& A, const T* x, T* y) {
        detail::spmv_transpose_impl(A, x, y, nullptr);
    }

    template <typename T>
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
