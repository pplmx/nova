#include <gtest/gtest.h>
#include "matrix/ops.h"
#include "cuda/device/device_utils.h"
#include <vector>
#include <cmath>

class MatrixOpsTest : public ::testing::Test {
protected:
    int rows_ = 32;
    int cols_ = 32;
    int size_;
    std::vector<float> h_a_;
    std::vector<float> h_b_;
    std::vector<float> h_c_;
    float *d_a_ = nullptr;
    float *d_b_ = nullptr;
    float *d_c_ = nullptr;

    void SetUp() override {
        size_ = rows_ * cols_;
        h_a_.resize(size_);
        h_b_.resize(size_);
        h_c_.resize(size_);

        CUDA_CHECK(cudaMalloc(&d_a_, size_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b_, size_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c_, size_ * sizeof(float)));
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(d_a_));
        CUDA_CHECK(cudaFree(d_b_));
        CUDA_CHECK(cudaFree(d_c_));
    }

    void uploadAB() {
        CUDA_CHECK(cudaMemcpy(d_a_, h_a_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b_, h_b_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    void downloadC() {
        CUDA_CHECK(cudaMemcpy(h_c_.data(), d_c_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

TEST_F(MatrixOpsTest, ElementwiseAdd) {
    for (int i = 0; i < size_; ++i) {
        h_a_[i] = static_cast<float>(i);
        h_b_[i] = static_cast<float>(i * 2);
    }
    uploadAB();

    matrixElementwiseAdd(d_a_, d_b_, d_c_, rows_, cols_);
    downloadC();

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_c_[i], h_a_[i] + h_b_[i]);
    }
}

TEST_F(MatrixOpsTest, ElementwiseAddAllZeros) {
    std::fill(h_a_.begin(), h_a_.end(), 0.0f);
    std::fill(h_b_.begin(), h_b_.end(), 0.0f);
    uploadAB();

    matrixElementwiseAdd(d_a_, d_b_, d_c_, rows_, cols_);
    downloadC();

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_c_[i], 0.0f);
    }
}

TEST_F(MatrixOpsTest, ElementwiseAddNegative) {
    for (int i = 0; i < size_; ++i) {
        h_a_[i] = static_cast<float>(i);
        h_b_[i] = -static_cast<float>(i);
    }
    uploadAB();

    matrixElementwiseAdd(d_a_, d_b_, d_c_, rows_, cols_);
    downloadC();

    for (int i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_c_[i], 0.0f, 1e-6);
    }
}

TEST_F(MatrixOpsTest, ElementwiseMultiply) {
    for (int i = 0; i < size_; ++i) {
        h_a_[i] = 2.0f;
        h_b_[i] = 3.0f;
    }
    uploadAB();

    matrixElementwiseMultiply(d_a_, d_b_, d_c_, rows_, cols_);
    downloadC();

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_c_[i], 6.0f);
    }
}

TEST_F(MatrixOpsTest, ElementwiseMultiplyZeros) {
    std::fill(h_a_.begin(), h_a_.end(), 5.0f);
    std::fill(h_b_.begin(), h_b_.end(), 0.0f);
    uploadAB();

    matrixElementwiseMultiply(d_a_, d_b_, d_c_, rows_, cols_);
    downloadC();

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_c_[i], 0.0f);
    }
}

TEST_F(MatrixOpsTest, ElementwiseMultiplyOnes) {
    std::fill(h_a_.begin(), h_a_.end(), 1.0f);
    std::fill(h_b_.begin(), h_b_.end(), 1.0f);
    uploadAB();

    matrixElementwiseMultiply(d_a_, d_b_, d_c_, rows_, cols_);
    downloadC();

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_c_[i], 1.0f);
    }
}

TEST_F(MatrixOpsTest, ScaleByTwo) {
    for (int i = 0; i < size_; ++i) {
        h_a_[i] = static_cast<float>(i);
    }
    CUDA_CHECK(cudaMemcpy(d_a_, h_a_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));

    matrixScale(d_a_, 2.0f, size_);

    std::vector<float> h_result(size_);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_a_, size_ * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_a_[i] * 2.0f);
    }
}

TEST_F(MatrixOpsTest, ScaleByZero) {
    std::fill(h_a_.begin(), h_a_.end(), 100.0f);
    CUDA_CHECK(cudaMemcpy(d_a_, h_a_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));

    matrixScale(d_a_, 0.0f, size_);

    std::vector<float> h_result(size_);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_a_, size_ * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 0.0f);
    }
}

TEST_F(MatrixOpsTest, ScaleNegative) {
    for (int i = 0; i < size_; ++i) {
        h_a_[i] = static_cast<float>(i - size_ / 2);
    }
    CUDA_CHECK(cudaMemcpy(d_a_, h_a_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));

    matrixScale(d_a_, -1.0f, size_);

    std::vector<float> h_result(size_);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_a_, size_ * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], -h_a_[i]);
    }
}

TEST_F(MatrixOpsTest, TransposeBasic) {
    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            h_a_[r * cols_ + c] = static_cast<float>(r * cols_ + c);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_a_, h_a_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));

    transposeMatrix(d_a_, d_c_, rows_, cols_);
    downloadC();

    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            float expected = h_a_[c * rows_ + r];
            EXPECT_FLOAT_EQ(h_c_[r * cols_ + c], expected);
        }
    }
}

TEST_F(MatrixOpsTest, TransposeAllOnes) {
    std::fill(h_a_.begin(), h_a_.end(), 1.0f);
    CUDA_CHECK(cudaMemcpy(d_a_, h_a_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));

    transposeMatrix(d_a_, d_c_, rows_, cols_);
    downloadC();

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_c_[i], 1.0f);
    }
}

TEST_F(MatrixOpsTest, TransposeAllZeros) {
    std::fill(h_a_.begin(), h_a_.end(), 0.0f);
    CUDA_CHECK(cudaMemcpy(d_a_, h_a_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));

    transposeMatrix(d_a_, d_c_, rows_, cols_);
    downloadC();

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_c_[i], 0.0f);
    }
}

TEST_F(MatrixOpsTest, TransposeTiled) {
    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            h_a_[r * cols_ + c] = static_cast<float>(r + c);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_a_, h_a_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));

    transposeMatrixTiled(d_a_, d_c_, rows_, cols_);
    downloadC();

    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            float expected = h_a_[c * rows_ + r];
            EXPECT_FLOAT_EQ(h_c_[r * cols_ + c], expected);
        }
    }
}

TEST_F(MatrixOpsTest, TransposeTiledIdentity) {
    for (int i = 0; i < size_; ++i) {
        h_a_[i] = static_cast<float>(i);
    }
    CUDA_CHECK(cudaMemcpy(d_a_, h_a_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));

    transposeMatrixTiled(d_a_, d_b_, rows_, cols_);

    transposeMatrixTiled(d_b_, d_c_, cols_, rows_);
    downloadC();

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_c_[i], h_a_[i]);
    }
}

TEST_F(MatrixOpsTest, NonSquareMatrix) {
    int old_rows = rows_;
    int old_cols = cols_;
    rows_ = 16;
    cols_ = 64;
    size_ = rows_ * cols_;
    h_a_.resize(size_);
    h_b_.resize(size_);
    h_c_.resize(cols_ * rows_);

    CUDA_CHECK(cudaFree(d_a_));
    CUDA_CHECK(cudaFree(d_b_));
    CUDA_CHECK(cudaFree(d_c_));
    CUDA_CHECK(cudaMalloc(&d_a_, size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_, size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c_, cols_ * rows_ * sizeof(float)));

    for (int i = 0; i < size_; ++i) {
        h_a_[i] = static_cast<float>(i);
    }
    CUDA_CHECK(cudaMemcpy(d_a_, h_a_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));

    transposeMatrix(d_a_, d_c_, rows_, cols_);

    std::vector<float> h_result(cols_ * rows_);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_c_, cols_ * rows_ * sizeof(float), cudaMemcpyDeviceToHost));

    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            float expected = h_a_[r * cols_ + c];
            float actual = h_result[c * rows_ + r];
            EXPECT_EQ(actual, expected) << " at r=" << r << " c=" << c;
        }
    }

    rows_ = old_rows;
    cols_ = old_cols;
}

TEST_F(MatrixOpsTest, ScaleByFraction) {
    for (int i = 0; i < size_; ++i) {
        h_a_[i] = 10.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_a_, h_a_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));

    matrixScale(d_a_, 0.5f, size_);

    std::vector<float> h_result(size_);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_a_, size_ * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 5.0f);
    }
}
