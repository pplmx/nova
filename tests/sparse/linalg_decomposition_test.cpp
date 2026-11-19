/**
 * @file linalg_decomposition_test.cpp
 * @brief Contracts for the SVD, eigendecomposition, QR and Cholesky entry points
 *
 * src/cuda/linalg/linalg.cu implements the matrix decompositions. Only one of its
 * seven functions was referenced from a test, and the condition number had its
 * meaning inverted:
 *
 *   return (min_s > 0) ? (max_s / min_s) : 0.0f;
 *
 * The condition number is sigma_max / sigma_min, so it grows without bound as a
 * matrix approaches singularity. Returning 0 for a singular matrix describes a
 * PERFECTLY conditioned one, so the worst case was reported as the best.
 * Measured on a rank-deficient 2x2 with singular values {1, 0}: the pre-fix code
 * reported condition_number = 0, i.e. "ideal".
 *
 * The same inversion appeared twice more: in eigenvalue_decomposition's final
 * expression, and in the empty-matrix early returns of both svd and
 * eigenvalue_decomposition, which set the field to 0 as well.
 *
 * svd also reported actual_rank = min_dim unconditionally, so a rank-deficient
 * matrix claimed full rank. It now counts singular values above a relative
 * tolerance.
 */

#include <cuda/linalg/linalg.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace cuda::linalg::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Owns a device copy of a row-major matrix.
        class DeviceMatrix {
        public:
            DeviceMatrix(std::initializer_list<float> values)
                : data_(values),
                  count_(values.size()) {
                cudaMalloc(&ptr_, count_ * sizeof(float));
                cudaMemcpy(ptr_, data_.data(), count_ * sizeof(float), cudaMemcpyHostToDevice);
            }

            ~DeviceMatrix() {
                if (ptr_ != nullptr) {
                    cudaFree(ptr_);
                }
            }

            DeviceMatrix(const DeviceMatrix&) = delete;
            DeviceMatrix& operator=(const DeviceMatrix&) = delete;

            const float* data() const { return ptr_; }

        private:
            std::vector<float> data_;
            float* ptr_ = nullptr;
            size_t count_ = 0;
        };

    }  // namespace

    // Regression: this returned 0, which reads as "perfectly conditioned".
    TEST(SvdConditionNumberTest, SingularMatrixReportsInfiniteCondition) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // [[1,0],[0,0]] has singular values {1, 0}: singular.
        DeviceMatrix matrix = {1.0f, 0.0f, 0.0f, 0.0f};

        SVDResult result;
        ASSERT_NO_THROW(svd(matrix.data(), 2, 2, result));

        std::vector<float> singular(result.S.size());
        result.S.copy_to(singular.data(), singular.size());
        ASSERT_EQ(singular.size(), 2u);
        EXPECT_NEAR(singular[0], 1.0f, 1e-5f);
        EXPECT_NEAR(singular[1], 0.0f, 1e-5f);

        EXPECT_TRUE(std::isinf(result.condition_number)) << "a singular matrix has an infinite condition number, got " << result.condition_number;
        EXPECT_FALSE(result.condition_number == 0.0f) << "zero describes a perfectly conditioned matrix";
    }

    TEST(SvdConditionNumberTest, WellConditionedMatrixReportsTheRatio) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // diag(2, 1): singular values {2, 1}, condition number 2.
        DeviceMatrix matrix = {2.0f, 0.0f, 0.0f, 1.0f};

        SVDResult result;
        ASSERT_NO_THROW(svd(matrix.data(), 2, 2, result));

        EXPECT_NEAR(result.condition_number, 2.0f, 1e-4f);
    }

    TEST(SvdConditionNumberTest, IdentityHasConditionNumberOne) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        DeviceMatrix matrix = {1.0f, 0.0f, 0.0f, 1.0f};

        SVDResult result;
        ASSERT_NO_THROW(svd(matrix.data(), 2, 2, result));

        EXPECT_NEAR(result.condition_number, 1.0f, 1e-4f);
    }

    // Regression: actual_rank was set to min_dim regardless of the spectrum.
    TEST(SvdRankTest, RankDeficientMatrixReportsItsNumericalRank) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        DeviceMatrix matrix = {1.0f, 0.0f, 0.0f, 0.0f};

        SVDResult result;
        ASSERT_NO_THROW(svd(matrix.data(), 2, 2, result));

        EXPECT_EQ(result.actual_rank, 1u) << "one zero singular value means rank 1, not 2";
    }

    TEST(SvdRankTest, FullRankMatrixReportsFullRank) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        DeviceMatrix matrix = {2.0f, 0.0f, 0.0f, 1.0f};

        SVDResult result;
        ASSERT_NO_THROW(svd(matrix.data(), 2, 2, result));

        EXPECT_EQ(result.actual_rank, 2u);
    }

    TEST(SvdShapeTest, EmptyMatrixHasNoConditionNumber) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        SVDResult result;
        // An empty shape needs no device data.
        svd(nullptr, 0, 0, result);

        EXPECT_EQ(result.actual_rank, 0u);
        EXPECT_TRUE(std::isnan(result.condition_number)) << "an empty matrix has no condition number, got " << result.condition_number;
    }

    TEST(SvdShapeTest, RectangularShapesProduceThinFactors) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 3x2: thin SVD gives U(3x2), S(2), Vt(2x2).
        DeviceMatrix matrix = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

        SVDResult result;
        ASSERT_NO_THROW(svd(matrix.data(), 3, 2, result));

        EXPECT_EQ(result.U.size(), 6u) << "U is m x min(m,n)";
        EXPECT_EQ(result.S.size(), 2u) << "two singular values";
        EXPECT_EQ(result.Vt.size(), 4u) << "Vt is min(m,n) x n";
    }

    // ------------------------------------------------ eigendecomposition --------

    TEST(EigenDecompositionTest, DiagonalMatrixReturnsItsEigenvaluesAscending) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // diag(3, 1, 2): eigenvalues 1, 2, 3 ascending.
        DeviceMatrix matrix = {3.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 2.0f};

        EVDResult result;
        ASSERT_NO_THROW(eigenvalue_decomposition(matrix.data(), 3, result));

        std::vector<float> values(3);
        result.eigenvalues.copy_to(values.data(), values.size());

        EXPECT_NEAR(values[0], 1.0f, 1e-4f);
        EXPECT_NEAR(values[1], 2.0f, 1e-4f);
        EXPECT_NEAR(values[2], 3.0f, 1e-4f);
    }

    TEST(EigenDecompositionTest, SymmetricMatrixHasRealEigenvalues) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // [[2,1],[1,2]] has eigenvalues 1 and 3.
        DeviceMatrix matrix = {2.0f, 1.0f, 1.0f, 2.0f};

        EVDResult result;
        ASSERT_NO_THROW(eigenvalue_decomposition(matrix.data(), 2, result));

        std::vector<float> values(2);
        result.eigenvalues.copy_to(values.data(), values.size());

        EXPECT_NEAR(values[0], 1.0f, 1e-3f);
        EXPECT_NEAR(values[1], 3.0f, 1e-3f);
        EXPECT_NEAR(result.condition_number, 3.0f, 1e-2f) << "condition number is max/min eigenvalue";
    }

    // Regression: a non-positive eigenvalue returned a condition number of zero.
    TEST(EigenDecompositionTest, IndefiniteMatrixDoesNotReportAZeroCondition) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // [[0,0],[0,-1]] has eigenvalues -1 and 0: singular and indefinite.
        DeviceMatrix matrix = {0.0f, 0.0f, 0.0f, -1.0f};

        EVDResult result;
        ASSERT_NO_THROW(eigenvalue_decomposition(matrix.data(), 2, result));

        EXPECT_FALSE(result.condition_number == 0.0f) << "zero would describe a perfectly conditioned matrix";
        EXPECT_TRUE(std::isinf(result.condition_number) || std::isnan(result.condition_number))
            << "an indefinite matrix has no finite condition number, got " << result.condition_number;
    }

    TEST(EigenDecompositionTest, EmptyMatrixHasNoConditionNumber) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        EVDResult result;
        eigenvalue_decomposition(nullptr, 0, result);

        EXPECT_TRUE(std::isnan(result.condition_number));
    }

    // ----------------------------------------------------------- QR ------------

    TEST(QrDecompositionTest, TallMatrixProducesThinFactors) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 3x2: Q(3x2), R(2x2).
        DeviceMatrix matrix = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

        QRResult result;
        ASSERT_NO_THROW(qr_decomposition(matrix.data(), 3, 2, result));

        EXPECT_EQ(result.Q.size(), 6u);
        EXPECT_EQ(result.R.size(), 4u);
    }

    TEST(QrDecompositionTest, QIsOrthonormal) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        DeviceMatrix matrix = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

        QRResult result;
        ASSERT_NO_THROW(qr_decomposition(matrix.data(), 3, 2, result));

        std::vector<float> q(6);
        result.Q.copy_to(q.data(), q.size());

        // Q is 3x2 stored column-major: column j starts at j*3.
        for (int j = 0; j < 2; ++j) {
            float norm = 0.0f;
            for (int i = 0; i < 3; ++i) {
                norm += q[i + j * 3] * q[i + j * 3];
            }
            EXPECT_NEAR(norm, 1.0f, 1e-3f) << "column " << j << " is not unit length";
        }
        float dot = 0.0f;
        for (int i = 0; i < 3; ++i) {
            dot += q[i] * q[i + 3];
        }
        EXPECT_NEAR(dot, 0.0f, 1e-3f) << "the two columns are not orthogonal";
    }

    // ------------------------------------------------------ Cholesky ----------

    TEST(CholeskyDecompositionTest, PositiveDefiniteMatrixSucceeds) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // [[4,2],[2,3]] is positive definite (determinant 8 > 0).
        DeviceMatrix matrix = {4.0f, 2.0f, 2.0f, 3.0f};

        CholeskyResult result;
        ASSERT_NO_THROW(cholesky_decomposition(matrix.data(), 2, result));

        EXPECT_TRUE(result.is_positive_definite);
    }

    TEST(CholeskyDecompositionTest, IndefiniteMatrixIsReported) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // [[1,2],[2,1]] has determinant -3, so it is not positive definite.
        DeviceMatrix matrix = {1.0f, 2.0f, 2.0f, 1.0f};

        CholeskyResult result;
        ASSERT_NO_THROW(cholesky_decomposition(matrix.data(), 2, result));

        EXPECT_FALSE(result.is_positive_definite) << "an indefinite matrix must not be reported as factorizable";
    }

}  // namespace cuda::linalg::test
