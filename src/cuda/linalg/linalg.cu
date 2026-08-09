#include "cuda/linalg/linalg.h"

#include <cmath>
#include <cstdlib>

#include "cuda/device/error.h"

namespace cuda::linalg {

namespace {

cusolverDnHandle_t get_cusolver_handle() {
    static cusolverDnHandle_t handle = [] {
        cusolverDnHandle_t h;
        cusolverDnCreate(&h);
        return h;
    }();
    return handle;
}

void check_cusolver(cusolverStatus_t status, const char* file, int line) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " - cuSOLVER error: " + std::to_string(static_cast<int>(status)));
    }
}

#define CUSOLVER_CHECK(call) check_cusolver(call, __FILE__, __LINE__)

float compute_condition_number(const float* s, size_t rank) {
    if (rank == 0) return 0.0f;
    float max_s = s[0];
    float min_s = s[rank - 1];
    return (min_s > 0) ? (max_s / min_s) : 0.0f;
}

}  // namespace

void svd(const float* A, size_t m, size_t n, SVDResult& result, SVDMode mode) {
    cusolverDnHandle_t handle = get_cusolver_handle();

    size_t min_dim = (m < n) ? m : n;
    size_t k = (mode == SVDMode::Thin) ? min_dim : min_dim;
    size_t k_actual = (mode == SVDMode::Randomized) ? (min_dim / 2) : k;

    result.U = memory::Buffer<float>(m * k_actual);
    result.S = memory::Buffer<float>(k_actual);
    result.Vt = memory::Buffer<float>(k_actual * n);
    result.actual_rank = k_actual;

    memory::Buffer<float> Acopy(m * n);
    CUDA_CHECK(cudaMemcpy(Acopy.data(), A, m * n * sizeof(float), cudaMemcpyDeviceToDevice));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(handle, static_cast<int>(m), static_cast<int>(n), &lwork));

    memory::Buffer<float> work(lwork);
    memory::Buffer<float> rwork(static_cast<size_t>((m < n) ? m : n));
    memory::Buffer<int> devInfo(1);

    CUSOLVER_CHECK(cusolverDnSgesvd(handle, 'A', 'A', static_cast<int>(m), static_cast<int>(n), Acopy.data(), static_cast<int>(m), result.S.data(), result.U.data(), static_cast<int>(m), result.Vt.data(), static_cast<int>(k_actual), work.data(), lwork, rwork.data(), devInfo.data()));

    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, devInfo.data(), sizeof(int), cudaMemcpyDeviceToHost));

    if (h_info != 0) {
        result.condition_number = 0.0f;
        return;
    }

    std::vector<float> h_s(k_actual);
    result.S.copy_to(h_s.data(), k_actual);
    result.condition_number = compute_condition_number(h_s.data(), k_actual);
}

void eigenvalue_decomposition(const float* A, size_t n, EVDResult& result) {
    cusolverDnHandle_t handle = get_cusolver_handle();

    memory::Buffer<float> Acopy(n * n);
    CUDA_CHECK(cudaMemcpy(Acopy.data(), A, n * n * sizeof(float), cudaMemcpyDeviceToDevice));

    result.eigenvalues = memory::Buffer<float>(n);
    result.eigenvectors = memory::Buffer<float>(n * n);

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, static_cast<int>(n), Acopy.data(), static_cast<int>(n), result.eigenvalues.data(), &lwork));

    memory::Buffer<float> work(lwork);
    memory::Buffer<int> devInfo(1);

    CUSOLVER_CHECK(cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, static_cast<int>(n), Acopy.data(), static_cast<int>(n), result.eigenvalues.data(), work.data(), lwork, devInfo.data()));

    CUDA_CHECK(cudaMemcpy(result.eigenvectors.data(), Acopy.data(), n * n * sizeof(float), cudaMemcpyDeviceToDevice));

    std::vector<float> h_ev(n);
    result.eigenvalues.copy_to(h_ev.data(), n);
    result.condition_number = compute_condition_number(h_ev.data(), n);
}

void qr_decomposition(const float* A, size_t m, size_t n, QRResult& result) {
    cusolverDnHandle_t handle = get_cusolver_handle();

    memory::Buffer<float> Acopy(m * n);
    CUDA_CHECK(cudaMemcpy(Acopy.data(), A, m * n * sizeof(float), cudaMemcpyDeviceToDevice));

    size_t k = (m < n) ? m : n;
    result.Q = memory::Buffer<float>(m * k);
    result.R = memory::Buffer<float>(k * n);

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(handle, static_cast<int>(m), static_cast<int>(n), Acopy.data(), static_cast<int>(m), &lwork));

    memory::Buffer<float> work(lwork);
    memory::Buffer<float> tau(k);
    memory::Buffer<int> devInfo(1);

    CUSOLVER_CHECK(cusolverDnSgeqrf(handle, static_cast<int>(m), static_cast<int>(n), Acopy.data(), static_cast<int>(m), tau.data(), work.data(), lwork, devInfo.data()));

    CUDA_CHECK(cudaMemcpy(result.R.data(), Acopy.data(), k * n * sizeof(float), cudaMemcpyDeviceToDevice));

    CUSOLVER_CHECK(cusolverDnSorgqr(handle, static_cast<int>(m), static_cast<int>(k), static_cast<int>(k), Acopy.data(), static_cast<int>(m), tau.data(), work.data(), lwork, devInfo.data()));

    CUDA_CHECK(cudaMemcpy(result.Q.data(), Acopy.data(), m * k * sizeof(float), cudaMemcpyDeviceToDevice));
}

void cholesky_decomposition(const float* A, size_t n, CholeskyResult& result) {
    cusolverDnHandle_t handle = get_cusolver_handle();

    memory::Buffer<float> Acopy(n * n);
    CUDA_CHECK(cudaMemcpy(Acopy.data(), A, n * n * sizeof(float), cudaMemcpyDeviceToDevice));

    result.L = memory::Buffer<float>(n * n);

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, static_cast<int>(n), Acopy.data(), static_cast<int>(n), &lwork));

    memory::Buffer<float> work(lwork);
    memory::Buffer<int> devInfo(1);

    cusolverStatus_t status = cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, static_cast<int>(n), Acopy.data(), static_cast<int>(n), work.data(), lwork, devInfo.data());

    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, devInfo.data(), sizeof(int), cudaMemcpyDeviceToHost));

    result.is_positive_definite = (h_info == 0);

    if (result.is_positive_definite) {
        // Acopy is device memory; the previous code indexed Acopy.data() from
        // the CPU to zero the strict upper triangle, which writes device memory
        // from the host (SIGSEGV / corruption). Zero it on a host staging copy
        // and upload back before producing the final L factor.
        std::vector<float> h_acopy(n * n);
        CUDA_CHECK(cudaMemcpy(h_acopy.data(), Acopy.data(), n * n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                h_acopy[i * n + j] = 0;
            }
        }
        CUDA_CHECK(cudaMemcpy(Acopy.data(), h_acopy.data(), n * n * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(result.L.data(), Acopy.data(), n * n * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
}

}  // namespace cuda::linalg
