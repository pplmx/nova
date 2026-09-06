#include "cuda/linalg/linalg.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

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

struct SymEigen {
    std::vector<float> w;  // eigenvalues, ascending
    std::vector<float> V;  // row-major, eigenvectors as columns: V[row][col]
};

// Cyclic-Jacobi symmetric eigendecomposition of a row-major symmetric matrix.
// M = V diag(w) V^T with w ascending and orthonormal V. Self-contained
// (no cuSOLVER): cusolverDnSsyevd has been observed non-conformant for
// general symmetric inputs on the target drivers, and a deterministic host
// solver keeps svd()/eigenvalue_decomposition() correct everywhere.
SymEigen jacobi_symmetric_eigen(int n, std::vector<float> M) {
    SymEigen out;
    out.V.assign(static_cast<size_t>(n) * n, 0.0f);
    for (int i = 0; i < n; ++i) out.V[static_cast<size_t>(i) * n + i] = 1.0f;

    constexpr int kMaxSweeps = 60;
    constexpr float kTol = 1e-14f;
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
        float off = 0.0f;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                off += M[static_cast<size_t>(p) * n + q] * M[static_cast<size_t>(p) * n + q];
        if (off < kTol) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const float apq = M[static_cast<size_t>(p) * n + q];
                if (std::fabs(apq) <= kTol) continue;
                const float app = M[static_cast<size_t>(p) * n + p];
                const float aqq = M[static_cast<size_t>(q) * n + q];
                const float theta = (aqq - app) / (2.0f * apq);
                const float t = copysignf(1.0f, theta) /
                                (std::fabs(theta) + std::sqrt(theta * theta + 1.0f));
                const float c = 1.0f / std::sqrt(t * t + 1.0f);
                const float s = t * c;
                for (int k = 0; k < n; ++k) {
                    const size_t pk = static_cast<size_t>(p) * n + k;
                    const size_t qk = static_cast<size_t>(q) * n + k;
                    const float a_pk = M[pk], a_qk = M[qk];
                    M[pk] = c * a_pk - s * a_qk;
                    M[qk] = s * a_pk + c * a_qk;
                }
                for (int k = 0; k < n; ++k) {
                    const size_t kp = static_cast<size_t>(k) * n + p;
                    const size_t kq = static_cast<size_t>(k) * n + q;
                    const float a_kp = M[kp], a_kq = M[kq];
                    M[kp] = c * a_kp - s * a_kq;
                    M[kq] = s * a_kp + c * a_kq;
                }
                for (int k = 0; k < n; ++k) {
                    const size_t kp = static_cast<size_t>(k) * n + p;
                    const size_t kq = static_cast<size_t>(k) * n + q;
                    const float v_kp = out.V[kp], v_kq = out.V[kq];
                    out.V[kp] = c * v_kp - s * v_kq;
                    out.V[kq] = s * v_kp + c * v_kq;
                }
            }
        }
    }

    out.w.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) out.w[static_cast<size_t>(i)] = M[static_cast<size_t>(i) * n + i];

    // Ascending eigenvalues with the matching eigenvector columns.
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (out.w[static_cast<size_t>(j)] < out.w[static_cast<size_t>(i)]) {
                std::swap(out.w[static_cast<size_t>(i)], out.w[static_cast<size_t>(j)]);
                for (int k = 0; k < n; ++k)
                    std::swap(out.V[static_cast<size_t>(k) * n + i],
                              out.V[static_cast<size_t>(k) * n + j]);
            }
        }
    }
    return out;
}

}  // namespace

void svd(const float* A, size_t m, size_t n, SVDResult& result, SVDMode mode) {
    (void)mode;  // every mode returns the thin factorization (see below)
    const size_t min_dim = (m < n) ? m : n;
    result.U = memory::Buffer<float>(0);
    result.S = memory::Buffer<float>(0);
    result.Vt = memory::Buffer<float>(0);
    result.actual_rank = 0;
    result.condition_number = 0.0f;
    if (m == 0 || n == 0) return;

    // SVD via the Gram matrix + host Jacobi rotations, instead of cuSOLVER's
    // cusolverDnSgesvd: on the target drivers that call is non-conformant for
    // non-square inputs (an 'S'/'S' call on m < n returns
    // CUSOLVER_STATUS_INVALID_VALUE, and on m > n it returned an incomplete
    // right factor with info == 0). The old code additionally requested the
    // FULL decomposition ('A'/'A') while sizing U/Vt for a truncation — an
    // out-of-bounds write past the m*min_dim U allocation whenever m != n.
    // A self-contained SVD is correct for every shape and deterministic.
    std::vector<float> hA(m * n);
    CUDA_CHECK(cudaMemcpy(hA.data(), A, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> hU(m * min_dim, 0.0f);
    std::vector<float> hS(min_dim, 0.0f);
    std::vector<float> hVt(min_dim * n, 0.0f);

    if (m >= n) {
        // Right singular vectors from AᵀA (n×n symmetric PSD); σ_k = sqrt(λ_k)
        // and u_k = A·v_k / σ_k.
        std::vector<float> G(n * n, 0.0f);
        for (size_t a = 0; a < n; ++a)
            for (size_t b = 0; b < n; ++b) {
                float acc = 0.0f;
                for (size_t i = 0; i < m; ++i) acc += hA[i * n + a] * hA[i * n + b];
                G[a * n + b] = acc;
            }
        const auto sg = jacobi_symmetric_eigen(static_cast<int>(n), std::move(G));
        std::vector<std::pair<float, size_t>> order;
        for (size_t k = 0; k < n; ++k)
            order.emplace_back(std::sqrt(std::max(0.0f, sg.w[k])), k);
        std::sort(order.begin(), order.end(), std::greater<>());
        const float tol = 1e-9f * std::max(1.0f, order[0].first);
        for (size_t k = 0; k < n; ++k) {
            hS[k] = order[k].first;
            const size_t c = order[k].second;
            if (hS[k] > tol) {
                for (size_t i = 0; i < m; ++i) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < n; ++j) acc += hA[i * n + j] * sg.V[j * n + c];
                    hU[i * min_dim + k] = acc / hS[k];
                }
            }
            for (size_t j = 0; j < n; ++j) hVt[k * n + j] = sg.V[j * n + c];
        }
    } else {
        // Left singular vectors from AAᵀ (m×m); v_kᵀ = u_kᵀ·A / σ_k.
        std::vector<float> G(m * m, 0.0f);
        for (size_t a = 0; a < m; ++a)
            for (size_t b = 0; b < m; ++b) {
                float acc = 0.0f;
                for (size_t j = 0; j < n; ++j) acc += hA[a * n + j] * hA[b * n + j];
                G[a * m + b] = acc;
            }
        const auto sg = jacobi_symmetric_eigen(static_cast<int>(m), std::move(G));
        std::vector<std::pair<float, size_t>> order;
        for (size_t k = 0; k < m; ++k)
            order.emplace_back(std::sqrt(std::max(0.0f, sg.w[k])), k);
        std::sort(order.begin(), order.end(), std::greater<>());
        const float tol = 1e-9f * std::max(1.0f, order[0].first);
        for (size_t k = 0; k < m; ++k) {
            hS[k] = order[k].first;
            const size_t c = order[k].second;
            for (size_t i = 0; i < m; ++i) hU[i * min_dim + k] = sg.V[i * m + c];
            if (hS[k] > tol) {
                for (size_t j = 0; j < n; ++j) {
                    float acc = 0.0f;
                    for (size_t i = 0; i < m; ++i) acc += hA[i * n + j] * sg.V[i * m + c];
                    hVt[k * n + j] = acc / hS[k];
                }
            }
        }
    }

    result.U = memory::Buffer<float>(m * min_dim);
    result.S = memory::Buffer<float>(min_dim);
    result.Vt = memory::Buffer<float>(min_dim * n);
    result.U.copy_from(hU.data(), hU.size());
    result.S.copy_from(hS.data(), hS.size());
    result.Vt.copy_from(hVt.data(), hVt.size());
    result.actual_rank = min_dim;
    result.condition_number = compute_condition_number(hS.data(), min_dim);
}

void eigenvalue_decomposition(const float* A, size_t n, EVDResult& result) {
    if (n == 0) {
        result.eigenvalues = memory::Buffer<float>(0);
        result.eigenvectors = memory::Buffer<float>(0);
        result.condition_number = 0.0f;
        return;
    }

    // Host Jacobi instead of cusolverDnSsyevd (non-conformant for general
    // symmetric inputs on the target drivers — CUSOLVER_STATUS_EXECUTION_FAILED
    // with garbage output on a non-diagonal matrix). Mirrors the old
    // CUBLAS_FILL_MODE_UPPER convention: the strict upper triangle is read and
    // reflected, so the caller need only fill one triangle.
    std::vector<float> hA(n * n);
    CUDA_CHECK(cudaMemcpy(hA.data(), A, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            hA[j * n + i] = hA[i * n + j];

    const auto sg = jacobi_symmetric_eigen(static_cast<int>(n), std::move(hA));

    result.eigenvalues = memory::Buffer<float>(n);
    result.eigenvectors = memory::Buffer<float>(n * n);
    result.eigenvalues.copy_from(sg.w.data(), n);

    // Eigenvectors are returned column-major (V[:, j] at row + j*n), matching
    // the syevd layout the API already exposed.
    std::vector<float> ev(n * n);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            ev[i + j * n] = sg.V[i * n + j];
    result.eigenvectors.copy_from(ev.data(), ev.size());

    // syevd-class results (and this Jacobi solver) are ASCENDING, but the
    // shared condition-number helper assumes the DESCENDING order cuSOLVER
    // uses for SVD singular values — feeding ascending values reported
    // min/max = 1/κ instead of κ. Compute max/min over the array so the
    // result is order-independent.
    float max_ev = std::numeric_limits<float>::lowest();
    float min_ev = std::numeric_limits<float>::max();
    for (size_t i = 0; i < n; ++i) {
        max_ev = std::max(max_ev, sg.w[i]);
        min_ev = std::min(min_ev, sg.w[i]);
    }
    result.condition_number = (min_ev > 0.0f) ? (max_ev / min_ev) : 0.0f;
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
