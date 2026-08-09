#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <gtest/gtest.h>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "image/gaussian_blur.h"
#include "test_patterns.cuh"
#include "cuda/device/error.h"

using cuda::memory::Buffer;

class GaussianBlurTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaDeviceSynchronize());

        width_ = 64;
        height_ = 64;
        size_ = width_ * height_ * 3;

        h_input_.resize(size_);
        h_output_.resize(size_);

        d_input_ = cuda::memory::Buffer<uint8_t>(size_);
        d_output_ = cuda::memory::Buffer<uint8_t>(size_);
    }

    void runAndDownload(float sigma = 1.0f, int kernel_size = 3) {
        d_input_.copy_from(h_input_.data(), size_);
        cuda::algo::gaussianBlur(d_input_, d_output_, width_, height_, sigma, kernel_size);
        d_output_.copy_to(h_output_.data(), size_);
    }

    size_t width_;
    size_t height_;
    size_t size_;
    std::vector<uint8_t> h_input_;
    std::vector<uint8_t> h_output_;
    cuda::memory::Buffer<uint8_t> d_input_;
    cuda::memory::Buffer<uint8_t> d_output_;
};

TEST_F(GaussianBlurTest, SolidImage) {
    generateSolid(h_input_.data(), width_, height_, 128);

    runAndDownload(1.0f, 3);

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 128, 2);
    }
}

TEST_F(GaussianBlurTest, SmallKernel) {
    generateSolid(h_input_.data(), width_, height_, 200);

    runAndDownload(0.5f, 3);

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 200, 3);
    }
}

TEST_F(GaussianBlurTest, LargerKernel) {
    generateSolid(h_input_.data(), width_, height_, 100);

    runAndDownload(2.0f, 5);

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 100, 5);
    }
}

TEST_F(GaussianBlurTest, Checkerboard) {
    generateCheckerboard(h_input_.data(), width_, height_, 8);

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

    cuda::memory::Buffer<uint8_t> d_input(size);
    cuda::memory::Buffer<uint8_t> d_output(size);

    d_input.copy_from(input.data(), size);
    cuda::algo::gaussianBlur(d_input, d_output, 1, 1, 1.0f, 3);
    d_output.copy_to(output.data(), size);

    EXPECT_NEAR(output[0], 128, 5);
}

TEST_F(GaussianBlurTest, NonSquareImage) {
    width_ = 100;
    height_ = 50;
    size_ = width_ * height_ * 3;

    h_input_.resize(size_);
    h_output_.resize(size_);
    d_input_ = cuda::memory::Buffer<uint8_t>(size_);
    d_output_ = cuda::memory::Buffer<uint8_t>(size_);

    generateSolid(h_input_.data(), width_, height_, 128);

    runAndDownload(1.0f, 3);

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 128, 3);
    }
}

TEST_F(GaussianBlurTest, LargeKernel) {
    width_ = 128;
    height_ = 128;
    size_ = width_ * height_ * 3;

    h_input_.resize(size_);
    h_output_.resize(size_);
    d_input_ = cuda::memory::Buffer<uint8_t>(size_);
    d_output_ = cuda::memory::Buffer<uint8_t>(size_);

    generateSolid(h_input_.data(), width_, height_, 100);

    runAndDownload(3.0f, 7);

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 100, 10);
    }
}
