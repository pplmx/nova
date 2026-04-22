#include <gtest/gtest.h>
#include <memory>
#include <cstddef>
#include <cstdint>

#include "image/types.h"
#include "image/gaussian_blur.h"
#include "test_patterns.cuh"

class GaussianBlurTest : public ::testing::Test {
protected:
    void SetUp() override {
        width_ = 64;
        height_ = 64;
        size_ = width_ * height_ * 3;

        h_input_ = std::make_unique<unsigned char[]>(size_);
        h_output_ = std::make_unique<unsigned char[]>(size_);

        CUDA_CHECK_IMAGE(cudaMalloc(&d_input_, size_));
        CUDA_CHECK_IMAGE(cudaMalloc(&d_output_, size_));
    }

    void TearDown() override {
        CUDA_CHECK_IMAGE(cudaFree(d_input_));
        CUDA_CHECK_IMAGE(cudaFree(d_output_));
    }

    void runAndDownload(float sigma = 1.0f, int kernel_size = 3) {
        CUDA_CHECK_IMAGE(cudaMemcpy(d_input_, h_input_.get(), size_, cudaMemcpyHostToDevice));
        gaussianBlur(d_input_, d_output_, width_, height_, sigma, kernel_size);
        CUDA_CHECK_IMAGE(cudaMemcpy(h_output_.get(), d_output_, size_, cudaMemcpyDeviceToHost));
    }

    size_t width_;
    size_t height_;
    size_t size_;
    std::unique_ptr<unsigned char[]> h_input_;
    std::unique_ptr<unsigned char[]> h_output_;
    uint8_t *d_input_;
    uint8_t *d_output_;
};

TEST_F(GaussianBlurTest, SolidImage) {
    generateSolid(h_input_.get(), width_, height_, 128);

    runAndDownload(1.0f, 3);

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 128, 2);
    }
}

TEST_F(GaussianBlurTest, SmallKernel) {
    generateSolid(h_input_.get(), width_, height_, 200);

    runAndDownload(0.5f, 3);

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 200, 3);
    }
}

TEST_F(GaussianBlurTest, LargerKernel) {
    generateSolid(h_input_.get(), width_, height_, 100);

    runAndDownload(2.0f, 5);

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 100, 5);
    }
}

TEST_F(GaussianBlurTest, Checkerboard) {
    generateCheckerboard(h_input_.get(), width_, height_, 8);

    runAndDownload(1.5f, 5);

    int nonZeroCount = 0;
    for (size_t i = 0; i < size_; i += 3) {
        if (h_output_[i] > 10 && h_output_[i] < 245) {
            nonZeroCount++;
        }
    }
    EXPECT_GT(nonZeroCount, size_ / 10);
}

TEST_F(GaussianBlurTest, SinglePixel) {
    size_t size = 3;
    std::vector<unsigned char> input(size, 128);
    std::vector<unsigned char> output(size, 0);

    uint8_t *d_input, *d_output;
    CUDA_CHECK_IMAGE(cudaMalloc(&d_input, size));
    CUDA_CHECK_IMAGE(cudaMalloc(&d_output, size));

    CUDA_CHECK_IMAGE(cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice));
    gaussianBlur(d_input, d_output, 1, 1, 1.0f, 3);
    CUDA_CHECK_IMAGE(cudaMemcpy(output.data(), d_output, size, cudaMemcpyDeviceToHost));

    EXPECT_NEAR(output[0], 128, 5);

    CUDA_CHECK_IMAGE(cudaFree(d_input));
    CUDA_CHECK_IMAGE(cudaFree(d_output));
}

TEST_F(GaussianBlurTest, NonSquareImage) {
    width_ = 100;
    height_ = 50;
    size_ = width_ * height_ * 3;

    h_input_ = std::make_unique<unsigned char[]>(size_);
    h_output_ = std::make_unique<unsigned char[]>(size_);

    generateSolid(h_input_.get(), width_, height_, 128);

    CUDA_CHECK_IMAGE(cudaMalloc(&d_input_, size_));
    CUDA_CHECK_IMAGE(cudaMalloc(&d_output_, size_));

    CUDA_CHECK_IMAGE(cudaMemcpy(d_input_, h_input_.get(), size_, cudaMemcpyHostToDevice));
    gaussianBlur(d_input_, d_output_, width_, height_, 1.0f, 3);
    CUDA_CHECK_IMAGE(cudaMemcpy(h_output_.get(), d_output_, size_, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 128, 3);
    }
}

TEST_F(GaussianBlurTest, LargeKernel) {
    width_ = 128;
    height_ = 128;
    size_ = width_ * height_ * 3;

    h_input_ = std::make_unique<unsigned char[]>(size_);
    h_output_ = std::make_unique<unsigned char[]>(size_);

    generateSolid(h_input_.get(), width_, height_, 100);

    CUDA_CHECK_IMAGE(cudaMalloc(&d_input_, size_));
    CUDA_CHECK_IMAGE(cudaMalloc(&d_output_, size_));

    CUDA_CHECK_IMAGE(cudaMemcpy(d_input_, h_input_.get(), size_, cudaMemcpyHostToDevice));
    gaussianBlur(d_input_, d_output_, width_, height_, 3.0f, 7);
    CUDA_CHECK_IMAGE(cudaMemcpy(h_output_.get(), d_output_, size_, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 100, 10);
    }
}
