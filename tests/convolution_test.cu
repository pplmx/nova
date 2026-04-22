#include <gtest/gtest.h>
#include "convolution/conv2d.h"
#include "cuda/memory/buffer.h"
#include <vector>
#include <cmath>

using cuda::memory::Buffer;

class ConvolutionTest : public ::testing::Test {
protected:
    size_t width_ = 32;
    size_t height_ = 32;
    std::vector<float> h_input_;
    std::vector<float> h_output_;
};

TEST_F(ConvolutionTest, IdentityKernel) {
    const size_t size = width_ * height_;
    h_input_.resize(size);
    h_output_.resize(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = static_cast<float>(i);
    }

    Buffer<float> input(size);
    Buffer<float> output(size);
    Buffer<float> kernel(9);
    input.copy_from(h_input_.data(), size);

    std::vector<float> h_kernel(9, 0.0f);
    h_kernel[4] = 1.0f;
    kernel.copy_from(h_kernel.data(), 9);

    cuda::algo::convolve2D(input, output, kernel, width_, height_, 3);
    output.copy_to(h_output_.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(h_output_[i], h_input_[i], 1e-3);
    }
}

TEST_F(ConvolutionTest, BoxBlurKernel) {
    const size_t size = width_ * height_;
    h_input_.resize(size);
    h_output_.resize(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = static_cast<float>(i % 256);
    }

    Buffer<float> input(size);
    Buffer<float> output(size);
    Buffer<float> kernel(9);
    kernel.fill(1.0f / 9.0f);
    input.copy_from(h_input_.data(), size);

    cuda::algo::convolve2D(input, output, kernel, width_, height_, 3);
    output.copy_to(h_output_.data(), size);

    bool changed = false;
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(h_output_[i] - h_input_[i]) > 0.1f) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed);
}

TEST_F(ConvolutionTest, CreateGaussianKernel) {
    Buffer<float> kernel;
    cuda::algo::createGaussianKernel(kernel, 5, 1.0f);

    EXPECT_EQ(kernel.size(), 25u);

    std::vector<float> h_k(25);
    kernel.copy_to(h_k.data(), 25);

    float sum = 0.0f;
    for (int i = 0; i < 25; ++i) {
        EXPECT_GE(h_k[i], 0.0f);
        sum += h_k[i];
    }
    EXPECT_NEAR(sum, 1.0f, 0.01f);
}

TEST_F(ConvolutionTest, GaussianKernelIsNormalized) {
    Buffer<float> kernel;
    cuda::algo::createGaussianKernel(kernel, 3, 0.5f);

    EXPECT_EQ(kernel.size(), 9u);

    std::vector<float> h_k(9);
    kernel.copy_to(h_k.data(), 9);

    float sum = 0.0f;
    for (int i = 0; i < 9; ++i) {
        sum += h_k[i];
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);
}

TEST_F(ConvolutionTest, CreateBoxKernel) {
    Buffer<float> kernel;
    cuda::algo::createBoxKernel(kernel, 3);

    EXPECT_EQ(kernel.size(), 9u);

    std::vector<float> h_k(9);
    kernel.copy_to(h_k.data(), 9);

    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(h_k[i], 1.0f / 9.0f);
    }
}

TEST_F(ConvolutionTest, CreateSobelKernelX) {
    Buffer<float> kernel;
    cuda::algo::createSobelKernelX(kernel);

    EXPECT_EQ(kernel.size(), 9u);

    std::vector<float> h_k(9);
    kernel.copy_to(h_k.data(), 9);

    float expected[] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(h_k[i], expected[i]);
    }
}

TEST_F(ConvolutionTest, CreateSobelKernelY) {
    Buffer<float> kernel;
    cuda::algo::createSobelKernelY(kernel);

    EXPECT_EQ(kernel.size(), 9u);

    std::vector<float> h_k(9);
    kernel.copy_to(h_k.data(), 9);

    float expected[] = {-1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f};
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(h_k[i], expected[i]);
    }
}

TEST_F(ConvolutionTest, GaussianBlur) {
    const size_t size = width_ * height_;
    h_input_.resize(size);
    h_output_.resize(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = (i % 2 == 0) ? 255.0f : 0.0f;
    }

    Buffer<float> input(size);
    Buffer<float> output(size);
    Buffer<float> kernel;
    input.copy_from(h_input_.data(), size);
    cuda::algo::createGaussianKernel(kernel, 5, 1.0f);

    cuda::algo::convolve2D(input, output, kernel, width_, height_, 5);
    output.copy_to(h_output_.data(), size);

    bool smoothed = false;
    for (size_t i = 1; i < size - 1; ++i) {
        if (h_output_[i] > 10.0f && h_output_[i] < 245.0f) {
            smoothed = true;
            break;
        }
    }
    EXPECT_TRUE(smoothed);
}

TEST_F(ConvolutionTest, LaplacianKernel) {
    const size_t size = width_ * height_;
    h_input_.resize(size);
    h_output_.resize(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = 128.0f;
    }

    Buffer<float> input(size);
    Buffer<float> output(size);
    Buffer<float> kernel(9);
    input.copy_from(h_input_.data(), size);

    std::vector<float> h_kernel = {0.0f, 1.0f, 0.0f, 1.0f, -4.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    kernel.copy_from(h_kernel.data(), 9);

    cuda::algo::convolve2D(input, output, kernel, width_, height_, 3);
    output.copy_to(h_output_.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(h_output_[i], 0.0f, 0.001f);
    }
}

TEST_F(ConvolutionTest, NonSquareInput) {
    width_ = 64;
    height_ = 32;
    const size_t size = width_ * height_;
    h_input_.resize(size);
    h_output_.resize(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = static_cast<float>(i);
    }

    Buffer<float> input(size);
    Buffer<float> output(size);
    Buffer<float> kernel(9);
    kernel.fill(1.0f / 9.0f);
    input.copy_from(h_input_.data(), size);

    cuda::algo::convolve2D(input, output, kernel, width_, height_, 3);
    output.copy_to(h_output_.data(), size);

    EXPECT_EQ(h_output_.size(), size);
}

TEST_F(ConvolutionTest, LargerKernel) {
    const size_t size = width_ * height_;
    h_input_.resize(size);
    h_output_.resize(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = 100.0f;
    }

    Buffer<float> input(size);
    Buffer<float> output(size);
    Buffer<float> kernel;
    input.copy_from(h_input_.data(), size);
    cuda::algo::createBoxKernel(kernel, 5);

    cuda::algo::convolve2D(input, output, kernel, width_, height_, 5);
    output.copy_to(h_output_.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(h_output_[i], 100.0f, 0.1f);
    }
}
