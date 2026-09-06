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
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }

    void TearDown() override {
        CUDA_CHECK(cudaDeviceSynchronize());
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
        1, 0, 0,
        0, 2, 0,
        0, 0, 3,
        1, 1, 1,
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
                recon += hU[static_cast<size_t>(i) * n + k] *
                         hS[static_cast<size_t>(k)] *
                         hVt[static_cast<size_t>(k) * n + j];
            }
            EXPECT_NEAR(recon, A[static_cast<size_t>(i) * n + j], 1e-4f)
                << "recon at (" << i << "," << j << ")";
        }
    }
}

TEST_F(LinalgTest, SvdThinReconstructsWideMatrix) {
    // m < n exercises the jobvt='A'-violating case that previously passed
    // ldvt = min_dim < n.
    const int m = 2, n = 4;
    std::vector<float> A = {
        1, 0, 0, 2,
        0, 3, 1, 0,
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
                recon += hU[static_cast<size_t>(i) * m + k] *
                         hS[static_cast<size_t>(k)] *
                         hVt[static_cast<size_t>(k) * n + j];
            }
            EXPECT_NEAR(recon, A[static_cast<size_t>(i) * n + j], 1e-4f)
                << "recon at (" << i << "," << j << ")";
        }
    }
}

TEST_F(LinalgTest, EvdConditionNumberIsNotInverted) {
    // Syevd returns eigenvalues in ASCENDING order {2, 4, 8}. The old shared
    // helper assumed the DESCENDING SVD order and reported min/max = 0.25
    // (1/κ) instead of κ = 8/2 = 4.
    std::vector<float> A = {8, 0, 0,
                            0, 4, 0,
                            0, 0, 2};
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
