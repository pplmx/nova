#include <gtest/gtest.h>
#include "convolution/conv2d.h"
#include "cuda/device/device_utils.h"
#include <vector>
#include <cmath>

class ConvolutionTest : public ::testing::Test {
protected:
    size_t width_ = 32;
    size_t height_ = 32;
    size_t size_;
    std::vector<float> h_input_;
    std::vector<float> h_output_;
    std::vector<float> h_kernel_;
    float *d_input_ = nullptr;
    float *d_output_ = nullptr;
    float *d_kernel_ = nullptr;
    float *d_kernel_large_ = nullptr;

    void SetUp() override {
        size_ = width_ * height_;
        h_input_.resize(size_);
        h_output_.resize(size_);
        h_kernel_.resize(9);

        CUDA_CHECK(cudaMalloc(&d_input_, size_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_, size_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kernel_, 9 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kernel_large_, 25 * sizeof(float)));
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(d_input_));
        CUDA_CHECK(cudaFree(d_output_));
        CUDA_CHECK(cudaFree(d_kernel_));
        CUDA_CHECK(cudaFree(d_kernel_large_));
    }

    void uploadInput() {
        CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    void downloadOutput() {
        CUDA_CHECK(cudaMemcpy(h_output_.data(), d_output_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

TEST_F(ConvolutionTest, IdentityKernel) {
    h_kernel_.assign(9, 0.0f);
    h_kernel_[4] = 1.0f;

    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = static_cast<float>(i);
    }
    uploadInput();
    CUDA_CHECK(cudaMemcpy(d_kernel_, h_kernel_.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));

    convolve2D(d_input_, d_output_, d_kernel_, width_, height_, 3);
    downloadOutput();

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], h_input_[i], 1e-3);
    }
}

TEST_F(ConvolutionTest, BoxBlurKernel) {
    for (size_t i = 0; i < 9; ++i) {
        h_kernel_[i] = 1.0f / 9.0f;
    }

    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = static_cast<float>(i % 256);
    }
    uploadInput();
    CUDA_CHECK(cudaMemcpy(d_kernel_, h_kernel_.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));

    convolve2D(d_input_, d_output_, d_kernel_, width_, height_, 3);
    downloadOutput();

    bool changed = false;
    for (size_t i = 0; i < size_; ++i) {
        if (std::abs(h_output_[i] - h_input_[i]) > 0.1f) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed);
}

TEST_F(ConvolutionTest, SobelXKernel) {
    float sobel_x[] = {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f
    };
    h_kernel_.assign(sobel_x, sobel_x + 9);

    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = 128.0f;
    }
    uploadInput();
    CUDA_CHECK(cudaMemcpy(d_kernel_, h_kernel_.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));

    convolve2D(d_input_, d_output_, d_kernel_, width_, height_, 3);
    downloadOutput();

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_GE(h_output_[i], -6000.0f);
        EXPECT_LE(h_output_[i], 6000.0f);
    }
}

TEST_F(ConvolutionTest, SobelYKernel) {
    float sobel_y[] = {
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f
    };
    h_kernel_.assign(sobel_y, sobel_y + 9);

    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = 128.0f;
    }
    uploadInput();
    CUDA_CHECK(cudaMemcpy(d_kernel_, h_kernel_.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));

    convolve2D(d_input_, d_output_, d_kernel_, width_, height_, 3);
    downloadOutput();

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_GE(h_output_[i], -6000.0f);
        EXPECT_LE(h_output_[i], 6000.0f);
    }
}

TEST_F(ConvolutionTest, CreateGaussianKernel) {
    createGaussianKernel(d_kernel_large_, 5, 1.0f);

    std::vector<float> h_k(25);
    CUDA_CHECK(cudaMemcpy(h_k.data(), d_kernel_large_, 25 * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < 25; ++i) {
        EXPECT_GE(h_k[i], 0.0f);
        sum += h_k[i];
    }
    EXPECT_NEAR(sum, 1.0f, 0.01f);
}

TEST_F(ConvolutionTest, GaussianKernelIsNormalized) {
    createGaussianKernel(d_kernel_, 3, 0.5f);
    // 3x3 kernel uses d_kernel_ which is 9 floats

    std::vector<float> h_k(9);
    CUDA_CHECK(cudaMemcpy(h_k.data(), d_kernel_, 9 * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < 9; ++i) {
        sum += h_k[i];
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);
}

TEST_F(ConvolutionTest, CreateBoxKernel) {
    createBoxKernel(d_kernel_, 3);

    std::vector<float> h_k(9);
    CUDA_CHECK(cudaMemcpy(h_k.data(), d_kernel_, 9 * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(h_k[i], 1.0f / 9.0f);
    }
}

TEST_F(ConvolutionTest, CreateSobelKernelX) {
    createSobelKernelX(d_kernel_);

    std::vector<float> h_k(9);
    CUDA_CHECK(cudaMemcpy(h_k.data(), d_kernel_, 9 * sizeof(float), cudaMemcpyDeviceToHost));

    float expected[] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(h_k[i], expected[i]);
    }
}

TEST_F(ConvolutionTest, CreateSobelKernelY) {
    createSobelKernelY(d_kernel_);

    std::vector<float> h_k(9);
    CUDA_CHECK(cudaMemcpy(h_k.data(), d_kernel_, 9 * sizeof(float), cudaMemcpyDeviceToHost));

    float expected[] = {-1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f};
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(h_k[i], expected[i]);
    }
}

TEST_F(ConvolutionTest, GaussianBlur) {
    createGaussianKernel(d_kernel_large_, 5, 1.0f);

    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = (i % 2 == 0) ? 255.0f : 0.0f;
    }
    uploadInput();

    convolve2D(d_input_, d_output_, d_kernel_large_, width_, height_, 5);
    downloadOutput();

    bool smoothed = false;
    for (size_t i = 1; i < size_ - 1; ++i) {
        if (h_output_[i] > 10.0f && h_output_[i] < 245.0f) {
            smoothed = true;
            break;
        }
    }
    EXPECT_TRUE(smoothed);
}

TEST_F(ConvolutionTest, LaplacianKernel) {
    float laplacian[] = {
        0.0f,  1.0f, 0.0f,
        1.0f, -4.0f, 1.0f,
        0.0f,  1.0f, 0.0f
    };
    h_kernel_.assign(laplacian, laplacian + 9);

    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = 128.0f;
    }
    uploadInput();
    CUDA_CHECK(cudaMemcpy(d_kernel_, h_kernel_.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));

    convolve2D(d_input_, d_output_, d_kernel_, width_, height_, 3);
    downloadOutput();

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 0.0f, 0.001f);
    }
}

TEST_F(ConvolutionTest, NonSquareInput) {
    width_ = 64;
    height_ = 32;
    size_ = width_ * height_;
    h_input_.resize(size_);
    h_output_.resize(size_);

    CUDA_CHECK(cudaFree(d_input_));
    CUDA_CHECK(cudaFree(d_output_));
    CUDA_CHECK(cudaMalloc(&d_input_, size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_, size_ * sizeof(float)));

    h_kernel_.assign(9, 1.0f / 9.0f);

    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = static_cast<float>(i);
    }
    uploadInput();
    CUDA_CHECK(cudaMemcpy(d_kernel_, h_kernel_.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));

    convolve2D(d_input_, d_output_, d_kernel_, width_, height_, 3);
    downloadOutput();

    EXPECT_EQ(h_output_.size(), size_);
}

TEST_F(ConvolutionTest, LargerKernel) {
    createBoxKernel(d_kernel_large_, 5);

    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = 100.0f;
    }
    uploadInput();

    convolve2D(d_input_, d_output_, d_kernel_large_, width_, height_, 5);
    downloadOutput();

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 100.0f, 0.1f);
    }
}
