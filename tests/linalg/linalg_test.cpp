#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "cuda/linalg/linalg.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

using namespace cuda::linalg;
using cuda::memory::Buffer;

namespace {

    class LinalgTest : public ::testing::Test {
    protected:
        void SetUp() override { CUDA_CHECK(cudaSetDevice(0)); }

        void TearDown() override {
            (void)cudaDeviceSynchronize();
            cudaGetLastError();
        }
    };

    // Thin SVD must reconstruct A = U * diag(S) * Vt for a TALL (m > n) matrix.
    // The old SVD requested the full 'A'/'A' decomposition into thin buffers
    // (m*min_dim), so cuSOLVER wrote past the end of U and ldvt violated the
    // jobvt='A' contract.
    TEST_F(LinalgTest, SvdThinReconstructsTallMatrix) {
        const int m = 4, n = 3;
        std::vector<float> A = {
            1,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            3,
            1,
            1,
            1,
        };
        Buffer<float> dA(static_cast<size_t>(m) * n);
        dA.copy_from(A.data(), dA.size());

        SVDResult r;
        svd(dA.data(), static_cast<size_t>(m), static_cast<size_t>(n), r, SVDMode::Thin);

        ASSERT_EQ(r.S.size(), static_cast<size_t>(n));  // min_dim = 3 singular values
        ASSERT_EQ(r.U.size(), static_cast<size_t>(m * n));
        ASSERT_EQ(r.Vt.size(), static_cast<size_t>(n * n));
        ASSERT_GT(r.condition_number, 0.0f);

        std::vector<float> hU(r.U.size()), hS(r.S.size()), hVt(r.Vt.size());
        r.U.copy_to(hU.data(), hU.size());
        r.S.copy_to(hS.data(), hS.size());
        r.Vt.copy_to(hVt.data(), hVt.size());

        // The decomposition is stored row-major: U is m×min_dim, Vt min_dim×n.
        // A = U diag(S) Vt must reconstruct the input exactly.
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float recon = 0.0f;
                for (int k = 0; k < n; ++k) {
                    recon += hU[static_cast<size_t>(i) * n + k] * hS[static_cast<size_t>(k)] * hVt[static_cast<size_t>(k) * n + j];
                }
                EXPECT_NEAR(recon, A[static_cast<size_t>(i) * n + j], 1e-4f) << "recon at (" << i << "," << j << ")";
            }
        }
    }

    TEST_F(LinalgTest, SvdThinReconstructsWideMatrix) {
        // m < n exercises the jobvt='A'-violating case that previously passed
        // ldvt = min_dim < n.
        const int m = 2, n = 4;
        std::vector<float> A = {
            1,
            0,
            0,
            2,
            0,
            3,
            1,
            0,
        };
        Buffer<float> dA(static_cast<size_t>(m) * n);
        dA.copy_from(A.data(), dA.size());

        SVDResult r;
        svd(dA.data(), static_cast<size_t>(m), static_cast<size_t>(n), r, SVDMode::Thin);

        ASSERT_EQ(r.S.size(), static_cast<size_t>(m));
        std::vector<float> hU(r.U.size()), hS(r.S.size()), hVt(r.Vt.size());
        r.U.copy_to(hU.data(), hU.size());
        r.S.copy_to(hS.data(), hS.size());
        r.Vt.copy_to(hVt.data(), hVt.size());

        // Thin layout: U m×min_dim, Vt min_dim×n, both row-major.
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float recon = 0.0f;
                for (int k = 0; k < m; ++k) {
                    recon += hU[static_cast<size_t>(i) * m + k] * hS[static_cast<size_t>(k)] * hVt[static_cast<size_t>(k) * n + j];
                }
                EXPECT_NEAR(recon, A[static_cast<size_t>(i) * n + j], 1e-4f) << "recon at (" << i << "," << j << ")";
            }
        }
    }

    TEST_F(LinalgTest, EvdConditionNumberIsNotInverted) {
        // Syevd returns eigenvalues in ASCENDING order {2, 4, 8}. The old shared
        // helper assumed the DESCENDING SVD order and reported min/max = 0.25
        // (1/κ) instead of κ = 8/2 = 4.
        std::vector<float> A = {8, 0, 0, 0, 4, 0, 0, 0, 2};
        Buffer<float> dA(A.size());
        dA.copy_from(A.data(), A.size());

        EVDResult r;
        eigenvalue_decomposition(dA.data(), 3, r);

        EXPECT_NEAR(r.condition_number, 4.0f, 1e-4f);

        auto* ev = r.eigenvalues.data();
        std::vector<float> h_ev(3);
        r.eigenvalues.copy_to(h_ev.data(), 3);
        (void)ev;
        EXPECT_NEAR(h_ev[0], 2.0f, 1e-4f);
        EXPECT_NEAR(h_ev[2], 8.0f, 1e-4f);
    }

}  // namespace

// ---------------------------------------------------------------------------
// QR and Cholesky had no tests at all, and both were wrong. They handed a
// row-major buffer to column-major cuSOLVER routines and copied the factors back
// without transposing.
//
// Measured on a 4x3 matrix, before the fix:
//     Q*R reconstruction error  1.364e+01
//     Q^T Q - I                 8.659e-01   (Q was not orthogonal)
//     R strict lower triangle   7.892e-01   (R was not triangular)
// and on a 3x3 SPD matrix, before the fix:
//     L*L^T reconstruction error 2.016e+03
// ---------------------------------------------------------------------------

namespace {

    /// A = Q*R, Q orthonormal columns, R upper-triangular. All row-major.
    void check_qr(const std::vector<float>& A, int m, int n) {
        const int k = (m < n) ? m : n;

        Buffer<float> dA(static_cast<size_t>(m) * n);
        dA.copy_from(A.data(), dA.size());

        QRResult r;
        qr_decomposition(dA.data(), static_cast<size_t>(m), static_cast<size_t>(n), r);
        ASSERT_EQ(r.Q.size(), static_cast<size_t>(m) * k);
        ASSERT_EQ(r.R.size(), static_cast<size_t>(k) * n);

        std::vector<float> Q(r.Q.size()), R(r.R.size());
        r.Q.copy_to(Q.data(), Q.size());
        r.R.copy_to(R.data(), R.size());

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double acc = 0.0;
                for (int t = 0; t < k; ++t) {
                    acc += static_cast<double>(Q[static_cast<size_t>(i) * k + t]) * R[static_cast<size_t>(t) * n + j];
                }
                EXPECT_NEAR(static_cast<float>(acc), A[static_cast<size_t>(i) * n + j], 1e-4f) << "Q*R at (" << i << "," << j << ")";
            }
        }

        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < k; ++b) {
                double acc = 0.0;
                for (int i = 0; i < m; ++i) {
                    acc += static_cast<double>(Q[static_cast<size_t>(i) * k + a]) * Q[static_cast<size_t>(i) * k + b];
                }
                EXPECT_NEAR(static_cast<float>(acc), (a == b) ? 1.0f : 0.0f, 1e-4f) << "Q^T Q at (" << a << "," << b << ")";
            }
        }

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < i; ++j) {
                EXPECT_FLOAT_EQ(R[static_cast<size_t>(i) * n + j], 0.0f) << "R must be upper triangular, at (" << i << "," << j << ")";
            }
        }
    }

    /// A = L*L^T with L row-major lower-triangular. All row-major.
    void check_cholesky(const std::vector<float>& A, int n) {
        Buffer<float> dA(static_cast<size_t>(n) * n);
        dA.copy_from(A.data(), dA.size());

        CholeskyResult r;
        cholesky_decomposition(dA.data(), static_cast<size_t>(n), r);
        ASSERT_TRUE(r.is_positive_definite);
        ASSERT_EQ(r.L.size(), static_cast<size_t>(n) * n);

        std::vector<float> L(r.L.size());
        r.L.copy_to(L.data(), L.size());

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double acc = 0.0;
                for (int t = 0; t < n; ++t) {
                    acc += static_cast<double>(L[static_cast<size_t>(i) * n + t]) * L[static_cast<size_t>(j) * n + t];
                }
                EXPECT_NEAR(static_cast<float>(acc), A[static_cast<size_t>(i) * n + j], 1e-4f) << "L*L^T at (" << i << "," << j << ")";
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                EXPECT_FLOAT_EQ(L[static_cast<size_t>(i) * n + j], 0.0f) << "L must be lower triangular, at (" << i << "," << j << ")";
            }
        }

        // The diagonal of a Cholesky factor is positive by construction.
        for (int i = 0; i < n; ++i) {
            EXPECT_GT(L[static_cast<size_t>(i) * n + i], 0.0f) << "diagonal at " << i;
        }
    }

}  // namespace

TEST_F(LinalgTest, QrDecomposesATallMatrix) {
    check_qr({1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 0, 1}, 4, 3);
}

// The wide case is the one whose ldvt/workspace contract differs; the old code got it wrong
// in the same way, so it is covered separately rather than left to the tall case.
TEST_F(LinalgTest, QrDecomposesAWideMatrix) {
    check_qr({1, 0, 0, 2, 0, 3, 1, 0}, 2, 4);
}

TEST_F(LinalgTest, QrDecomposesASquareMatrix) {
    check_qr({4, 3, 2, 1, 5, 7, 9, 6, 8}, 3, 3);
}

TEST_F(LinalgTest, CholeskyFactorsASpdMatrix) {
    // A = [4 12 -16; 12 37 -43; -16 -43 98] is the classic textbook SPD example; its
    // factor is L = [2 0 0; 6 1 0; -8 5 3].
    std::vector<float> A = {4, 12, -16, 12, 37, -43, -16, -43, 98};
    check_cholesky(A, 3);

    Buffer<float> dA(A.size());
    dA.copy_from(A.data(), A.size());
    CholeskyResult r;
    cholesky_decomposition(dA.data(), 3, r);
    std::vector<float> L(9);
    r.L.copy_to(L.data(), L.size());
    EXPECT_NEAR(L[0], 2.0f, 1e-4f);
    EXPECT_NEAR(L[3], 6.0f, 1e-4f);
    EXPECT_NEAR(L[4], 1.0f, 1e-4f);
    EXPECT_NEAR(L[6], -8.0f, 1e-4f);
    EXPECT_NEAR(L[7], 5.0f, 1e-4f);
    EXPECT_NEAR(L[8], 3.0f, 1e-4f);
}

// A diagonal matrix makes the column-major/row-major confusion invisible (it is symmetric and
// triangular at once), so the factorisation is checked on a matrix whose off-diagonals are all
// non-zero.
TEST_F(LinalgTest, CholeskyFactorsAMatrixWithNoZeroOffDiagonals) {
    // A = [[5,2,1],[2,4,1],[1,1,3]], SPD.
    std::vector<float> A = {5, 2, 1, 2, 4, 1, 1, 1, 3};
    check_cholesky(A, 3);
}

TEST_F(LinalgTest, CholeskyReportsAnIndefiniteMatrixWithoutThrowing) {
    // Eigenvalues 3 and -1: not positive definite. spotrf reports the failing pivot through
    // info rather than failing the call, so this is a reported answer, not an exception.
    std::vector<float> A = {1, 2, 2, 1};
    Buffer<float> dA(A.size());
    dA.copy_from(A.data(), A.size());

    CholeskyResult r;
    EXPECT_NO_THROW(cholesky_decomposition(dA.data(), 2, r));
    EXPECT_FALSE(r.is_positive_definite);
}

TEST_F(LinalgTest, CholeskyAcceptsAnEmptyMatrix) {
    CholeskyResult r;
    EXPECT_NO_THROW(cholesky_decomposition(nullptr, 0, r));
    EXPECT_EQ(r.L.size(), 0u);
    EXPECT_TRUE(r.is_positive_definite) << "a 0x0 matrix is trivially positive definite; there is just no factor";
}

TEST_F(LinalgTest, QrAcceptsAnEmptyMatrix) {
    QRResult r;
    EXPECT_NO_THROW(qr_decomposition(nullptr, 0, 0, r));
    EXPECT_EQ(r.Q.size(), 0u);
    EXPECT_EQ(r.R.size(), 0u);
}
