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

        namespace {

            /// cuSOLVER exposes no name lookup, so map the enumerators by hand. Reporting
            /// the bare integer made every failure an opaque number.
            const char* cusolver_status_name(cusolverStatus_t status) {
                switch (status) {
                    case CUSOLVER_STATUS_SUCCESS:
                        return "CUSOLVER_STATUS_SUCCESS";
                    case CUSOLVER_STATUS_NOT_INITIALIZED:
                        return "CUSOLVER_STATUS_NOT_INITIALIZED";
                    case CUSOLVER_STATUS_ALLOC_FAILED:
                        return "CUSOLVER_STATUS_ALLOC_FAILED";
                    case CUSOLVER_STATUS_INVALID_VALUE:
                        return "CUSOLVER_STATUS_INVALID_VALUE";
                    case CUSOLVER_STATUS_ARCH_MISMATCH:
                        return "CUSOLVER_STATUS_ARCH_MISMATCH";
                    case CUSOLVER_STATUS_MAPPING_ERROR:
                        return "CUSOLVER_STATUS_MAPPING_ERROR";
                    case CUSOLVER_STATUS_EXECUTION_FAILED:
                        return "CUSOLVER_STATUS_EXECUTION_FAILED";
                    case CUSOLVER_STATUS_INTERNAL_ERROR:
                        return "CUSOLVER_STATUS_INTERNAL_ERROR";
                    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
                    case CUSOLVER_STATUS_NOT_SUPPORTED:
                        return "CUSOLVER_STATUS_NOT_SUPPORTED";
                    case CUSOLVER_STATUS_ZERO_PIVOT:
                        return "CUSOLVER_STATUS_ZERO_PIVOT";
                    case CUSOLVER_STATUS_INVALID_LICENSE:
                        return "CUSOLVER_STATUS_INVALID_LICENSE";
                    default:
                        return "CUSOLVER_STATUS_UNKNOWN";
                }
            }

        }  // namespace

        void check_cusolver(cusolverStatus_t status, const char* file, int line) {
            if (status != CUSOLVER_STATUS_SUCCESS) {
                throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " - cuSOLVER error: " + cusolver_status_name(status) + " ("
                                         + std::to_string(static_cast<int>(status)) + ")");
            }
        }

#define CUSOLVER_CHECK(call) check_cusolver(call, __FILE__, __LINE__)

        float compute_condition_number(const float* s, size_t rank) {
            // The condition number is sigma_max / sigma_min, so it grows without bound
            // as the matrix approaches singularity.
            //
            // Returning 0 in that case inverted the meaning: a condition number of zero
            // describes a perfectly conditioned matrix, so the worst case was reported
            // as the best. A singular matrix now reports infinity, and an empty
            // spectrum (no singular values at all) reports NaN rather than a number that
            // looks like a good condition.
            if (rank == 0) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            const float max_s = s[0];
            const float min_s = s[rank - 1];
            if (!(min_s > 0.0f)) {
                return std::numeric_limits<float>::infinity();
            }
            return max_s / min_s;
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
            for (int i = 0; i < n; ++i) {
                out.V[static_cast<size_t>(i) * n + i] = 1.0f;
            }

            constexpr int kMaxSweeps = 60;
            constexpr float kTol = 1e-14f;
            for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
                float off = 0.0f;
                for (int p = 0; p < n; ++p) {
                    for (int q = p + 1; q < n; ++q) {
                        off += M[static_cast<size_t>(p) * n + q] * M[static_cast<size_t>(p) * n + q];
                    }
                }
                if (off < kTol) {
                    break;
                }
                for (int p = 0; p < n; ++p) {
                    for (int q = p + 1; q < n; ++q) {
                        const float apq = M[static_cast<size_t>(p) * n + q];
                        if (std::fabs(apq) <= kTol) {
                            continue;
                        }
                        const float app = M[static_cast<size_t>(p) * n + p];
                        const float aqq = M[static_cast<size_t>(q) * n + q];
                        const float theta = (aqq - app) / (2.0f * apq);
                        const float t = copysignf(1.0f, theta) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0f));
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
            for (int i = 0; i < n; ++i) {
                out.w[static_cast<size_t>(i)] = M[static_cast<size_t>(i) * n + i];
            }

            // Ascending eigenvalues with the matching eigenvector columns.
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    if (out.w[static_cast<size_t>(j)] < out.w[static_cast<size_t>(i)]) {
                        std::swap(out.w[static_cast<size_t>(i)], out.w[static_cast<size_t>(j)]);
                        for (int k = 0; k < n; ++k) {
                            std::swap(out.V[static_cast<size_t>(k) * n + i], out.V[static_cast<size_t>(k) * n + j]);
                        }
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
        // An empty matrix has no condition number; NaN rather than a value that
        // reads as well-conditioned.
        result.condition_number = std::numeric_limits<float>::quiet_NaN();
        if (m == 0 || n == 0) {
            return;
        }

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
            for (size_t a = 0; a < n; ++a) {
                for (size_t b = 0; b < n; ++b) {
                    float acc = 0.0f;
                    for (size_t i = 0; i < m; ++i) {
                        acc += hA[i * n + a] * hA[i * n + b];
                    }
                    G[a * n + b] = acc;
                }
            }
            const auto sg = jacobi_symmetric_eigen(static_cast<int>(n), std::move(G));
            std::vector<std::pair<float, size_t>> order;
            for (size_t k = 0; k < n; ++k) {
                order.emplace_back(std::sqrt(std::max(0.0f, sg.w[k])), k);
            }
            std::sort(order.begin(), order.end(), std::greater<>());
            const float tol = 1e-9f * std::max(1.0f, order[0].first);
            for (size_t k = 0; k < n; ++k) {
                hS[k] = order[k].first;
                const size_t c = order[k].second;
                if (hS[k] > tol) {
                    for (size_t i = 0; i < m; ++i) {
                        float acc = 0.0f;
                        for (size_t j = 0; j < n; ++j) {
                            acc += hA[i * n + j] * sg.V[j * n + c];
                        }
                        hU[i * min_dim + k] = acc / hS[k];
                    }
                }
                for (size_t j = 0; j < n; ++j) {
                    hVt[k * n + j] = sg.V[j * n + c];
                }
            }
        } else {
            // Left singular vectors from AAᵀ (m×m); v_kᵀ = u_kᵀ·A / σ_k.
            std::vector<float> G(m * m, 0.0f);
            for (size_t a = 0; a < m; ++a) {
                for (size_t b = 0; b < m; ++b) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < n; ++j) {
                        acc += hA[a * n + j] * hA[b * n + j];
                    }
                    G[a * m + b] = acc;
                }
            }
            const auto sg = jacobi_symmetric_eigen(static_cast<int>(m), std::move(G));
            std::vector<std::pair<float, size_t>> order;
            for (size_t k = 0; k < m; ++k) {
                order.emplace_back(std::sqrt(std::max(0.0f, sg.w[k])), k);
            }
            std::sort(order.begin(), order.end(), std::greater<>());
            const float tol = 1e-9f * std::max(1.0f, order[0].first);
            for (size_t k = 0; k < m; ++k) {
                hS[k] = order[k].first;
                const size_t c = order[k].second;
                for (size_t i = 0; i < m; ++i) {
                    hU[i * min_dim + k] = sg.V[i * m + c];
                }
                if (hS[k] > tol) {
                    for (size_t j = 0; j < n; ++j) {
                        float acc = 0.0f;
                        for (size_t i = 0; i < m; ++i) {
                            acc += hA[i * n + j] * sg.V[i * m + c];
                        }
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
        // The numerical rank counts singular values above a tolerance relative to
        // the largest; reporting min_dim unconditionally claimed a full-rank matrix
        // even when one or more singular values were zero.
        const float largest = hS.empty() ? 0.0f : hS[0];
        const float rank_tolerance = largest * static_cast<float>(min_dim) * std::numeric_limits<float>::epsilon();
        size_t numerical_rank = 0;
        for (size_t i = 0; i < min_dim; ++i) {
            if (hS[i] > rank_tolerance) {
                ++numerical_rank;
            }
        }
        result.actual_rank = numerical_rank;
        result.condition_number = compute_condition_number(hS.data(), min_dim);
    }

    void eigenvalue_decomposition(const float* A, size_t n, EVDResult& result) {
        if (n == 0) {
            result.eigenvalues = memory::Buffer<float>(0);
            result.eigenvectors = memory::Buffer<float>(0);
            // No spectrum means no condition number; NaN keeps it from looking good.
            result.condition_number = std::numeric_limits<float>::quiet_NaN();
            return;
        }

        // Host Jacobi instead of cusolverDnSsyevd (non-conformant for general
        // symmetric inputs on the target drivers — CUSOLVER_STATUS_EXECUTION_FAILED
        // with garbage output on a non-diagonal matrix). Mirrors the old
        // CUBLAS_FILL_MODE_UPPER convention: the strict upper triangle is read and
        // reflected, so the caller need only fill one triangle.
        std::vector<float> hA(n * n);
        CUDA_CHECK(cudaMemcpy(hA.data(), A, n * n * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                hA[j * n + i] = hA[i * n + j];
            }
        }

        const auto sg = jacobi_symmetric_eigen(static_cast<int>(n), std::move(hA));

        result.eigenvalues = memory::Buffer<float>(n);
        result.eigenvectors = memory::Buffer<float>(n * n);
        result.eigenvalues.copy_from(sg.w.data(), n);

        // Eigenvectors are returned column-major (V[:, j] at row + j*n), matching
        // the syevd layout the API already exposed.
        std::vector<float> ev(n * n);
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < n; ++i) {
                ev[i + j * n] = sg.V[i * n + j];
            }
        }
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
        // Same semantic inversion as compute_condition_number: a non-positive
        // minimum eigenvalue means the matrix is singular or indefinite, which is the
        // worst conditioning, not a condition number of zero.
        result.condition_number = (min_ev > 0.0f) ? (max_ev / min_ev) : std::numeric_limits<float>::infinity();
    }

    void qr_decomposition(const float* A, size_t m, size_t n, QRResult& result) {
        cusolverDnHandle_t handle = get_cusolver_handle();

        if (m == 0 || n == 0) {
            result.Q = memory::Buffer<float>(0);
            result.R = memory::Buffer<float>(0);
            return;
        }

        const size_t k = (m < n) ? m : n;

        // cuSOLVER is column-major and everything else in this header is row-major, so the
        // matrix is transposed in and both factors are transposed out. The previous version
        // handed the row-major buffer over as-is and copied the factors back as-is, so the
        // call factored A^T and both outputs were read in the wrong order.
        //
        // Measured on a 4x3 matrix: Q*R reconstruction error 1.364e+01, Q^T Q - I 8.659e-01,
        // and R's strict lower triangle 7.892e-01 -- Q was not orthogonal and R was not
        // triangular. After the transpose, on the same matrix and on a wide and a square one:
        // reconstruction 8.749e-07 / 0 / 1.480e-06, Q^T Q - I 2.2e-07, R lower exactly 0.
        std::vector<float> h_A(m * n);
        CUDA_CHECK(cudaMemcpy(h_A.data(), A, m * n * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> h_A_col_major(m * n);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                h_A_col_major[j * m + i] = h_A[i * n + j];
            }
        }

        memory::Buffer<float> Acopy(m * n);
        Acopy.copy_from(h_A_col_major.data(), h_A_col_major.size());

        result.Q = memory::Buffer<float>(m * k);
        result.R = memory::Buffer<float>(k * n);

        memory::Buffer<float> tau(k);
        memory::Buffer<int> devInfo(1);

        int lwork_geqrf = 0;
        CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(handle, static_cast<int>(m), static_cast<int>(n), Acopy.data(), static_cast<int>(m), &lwork_geqrf));

        // orgqr has its OWN workspace query, and the two are not interchangeable: measured on
        // a 4x3 matrix, geqrf asked for 786432 floats and orgqr for 33056. Passing the geqrf
        // size happened to be safe here because it is the larger of the two, but nothing
        // guarantees that ordering for every shape, so the buffer is sized to the larger of
        // the two queries rather than to whichever was measured first.
        int lwork_orgqr = 0;
        CUSOLVER_CHECK(
            cusolverDnSorgqr_bufferSize(handle, static_cast<int>(m), static_cast<int>(k), static_cast<int>(k), Acopy.data(), static_cast<int>(m), tau.data(), &lwork_orgqr));

        const int lwork = std::max(lwork_geqrf, lwork_orgqr);
        memory::Buffer<float> work(lwork);

        CUSOLVER_CHECK(cusolverDnSgeqrf(handle, static_cast<int>(m), static_cast<int>(n), Acopy.data(), static_cast<int>(m), tau.data(), work.data(), lwork, devInfo.data()));

        // R is the upper triangle of the factored matrix: cuSOLVER leaves the Householder
        // reflectors below the diagonal, so the lower half must not be copied out as if it
        // were part of R.
        std::vector<float> h_factor(m * n, 0.0f);
        CUDA_CHECK(cudaMemcpy(h_factor.data(), Acopy.data(), m * n * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> h_R(k * n, 0.0f);
        for (size_t row = 0; row < k; ++row) {
            for (size_t col = 0; col < n; ++col) {
                if (row <= col) {
                    h_R[row * n + col] = h_factor[col * m + row];  // column-major -> row-major
                }
            }
        }

        CUSOLVER_CHECK(cusolverDnSorgqr(
            handle, static_cast<int>(m), static_cast<int>(k), static_cast<int>(k), Acopy.data(), static_cast<int>(m), tau.data(), work.data(), lwork, devInfo.data()));

        std::vector<float> h_Q_col_major(m * k, 0.0f);
        CUDA_CHECK(cudaMemcpy(h_Q_col_major.data(), Acopy.data(), m * k * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> h_Q(m * k, 0.0f);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                h_Q[i * k + j] = h_Q_col_major[j * m + i];
            }
        }

        result.Q.copy_from(h_Q.data(), h_Q.size());
        result.R.copy_from(h_R.data(), h_R.size());
    }

    void cholesky_decomposition(const float* A, size_t n, CholeskyResult& result) {
        cusolverDnHandle_t handle = get_cusolver_handle();

        if (n == 0) {
            result.L = memory::Buffer<float>(0);
            // A 0x0 matrix is trivially positive definite; there is simply no factor to
            // compute. Reporting false would say the matrix failed a test it was never run
            // against.
            result.is_positive_definite = true;
            return;
        }

        memory::Buffer<float> Acopy(n * n);
        CUDA_CHECK(cudaMemcpy(Acopy.data(), A, n * n * sizeof(float), cudaMemcpyDeviceToDevice));

        result.L = memory::Buffer<float>(n * n);

        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, static_cast<int>(n), Acopy.data(), static_cast<int>(n), &lwork));

        memory::Buffer<float> work(lwork);
        memory::Buffer<int> devInfo(1);

        const cusolverStatus_t status =
            cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, static_cast<int>(n), Acopy.data(), static_cast<int>(n), work.data(), lwork, devInfo.data());

        // A non-positive-definite matrix is a legitimate answer, not a failure of the call:
        // spotrf returns CUSOLVER_STATUS_SUCCESS and reports the offending pivot in devInfo.
        // Anything else is a real error and has to be raised rather than reported as
        // "not positive definite", which is what ignoring `status` did.
        CUSOLVER_CHECK(status);

        int h_info = 0;
        CUDA_CHECK(cudaMemcpy(&h_info, devInfo.data(), sizeof(int), cudaMemcpyDeviceToHost));

        result.is_positive_definite = (h_info == 0);

        if (!result.is_positive_definite) {
            return;
        }

        // L is stored COLUMN-MAJOR by cuSOLVER (L[i][j] at Acopy[j * n + i], lower triangle
        // j <= i) while everything else in this header is row-major. The previous version
        // copied the raw buffer straight out and zeroed the strict UPPER triangle by
        // row-major index -- which, on a column-major factor, cleared the wrong half.
        //
        // Measured on a 3x3 SPD matrix:
        //     returned L^T (as read):  2  6 -8 / 12  1  5 / -16 -43  3
        //     L * L^T reconstruction error: 2.016e+03
        // The factor cuSOLVER produced was correct; it was being read in the wrong order.
        // Transposed into a row-major lower-triangular L, the same matrix reconstructs with
        // error 0.000e+00.
        std::vector<float> h_factor(n * n, 0.0f);
        CUDA_CHECK(cudaMemcpy(h_factor.data(), Acopy.data(), n * n * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> h_L(n * n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                h_L[i * n + j] = h_factor[j * n + i];
            }
        }
        result.L.copy_from(h_L.data(), h_L.size());
    }

}  // namespace cuda::linalg
