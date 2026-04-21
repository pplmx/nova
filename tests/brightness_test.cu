#include <gtest/gtest.h>
#include <memory>
#include <cstring>
#include <cstddef>
#include <cstdint>

#include "image_utils.h"
#include "brightness.h"
#include "test_patterns.cuh"

class BrightnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        width_ = 64;
        height_ = 64;
        size_ = width_ * height_ * 3;

        h_input_ = std::make_unique<unsigned char[]>(size_);
        h_output_ = std::make_unique<unsigned char[]>(size_);
        h_expected_ = std::make_unique<unsigned char[]>(size_);

        CUDA_CHECK_IMAGE(cudaMalloc(&d_input_, size_));
        CUDA_CHECK_IMAGE(cudaMalloc(&d_output_, size_));
    }

    void TearDown() override {
        CUDA_CHECK_IMAGE(cudaFree(d_input_));
        CUDA_CHECK_IMAGE(cudaFree(d_output_));
    }

    void runAndDownload() {
        CUDA_CHECK_IMAGE(cudaMemcpy(d_input_, h_input_.get(), size_, cudaMemcpyHostToDevice));
        adjustBrightnessContrast(d_input_, d_output_, width_, height_, alpha_, beta_);
        CUDA_CHECK_IMAGE(cudaMemcpy(h_output_.get(), d_output_, size_, cudaMemcpyDeviceToHost));
    }

    size_t width_;
    size_t height_;
    size_t size_;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    std::unique_ptr<unsigned char[]> h_input_;
    std::unique_ptr<unsigned char[]> h_output_;
    std::unique_ptr<unsigned char[]> h_expected_;
    uint8_t* d_input_;
    uint8_t* d_output_;
};

TEST_F(BrightnessTest, Identity) {
    generateSolid(h_input_.get(), width_, height_, 128);
    std::memset(h_expected_.get(), 128, size_);

    alpha_ = 1.0f;
    beta_ = 0.0f;
    runAndDownload();

    EXPECT_TRUE(compareBuffers(h_expected_.get(), h_output_.get(), size_));
}

TEST_F(BrightnessTest, BrightnessIncrease) {
    generateSolid(h_input_.get(), width_, height_, 100);
    std::memset(h_expected_.get(), 150, size_);

    alpha_ = 1.0f;
    beta_ = 50.0f;
    runAndDownload();

    EXPECT_TRUE(compareBuffers(h_expected_.get(), h_output_.get(), size_));
}

TEST_F(BrightnessTest, BrightnessDecrease) {
    generateSolid(h_input_.get(), width_, height_, 200);
    std::memset(h_expected_.get(), 150, size_);

    alpha_ = 1.0f;
    beta_ = -50.0f;
    runAndDownload();

    EXPECT_TRUE(compareBuffers(h_expected_.get(), h_output_.get(), size_));
}

TEST_F(BrightnessTest, ContrastIncrease) {
    generateSolid(h_input_.get(), width_, height_, 64);
    std::memset(h_expected_.get(), 128, size_);

    alpha_ = 2.0f;
    beta_ = 0.0f;
    runAndDownload();

    EXPECT_TRUE(compareBuffers(h_expected_.get(), h_output_.get(), size_));
}

TEST_F(BrightnessTest, BrightnessAndContrastCombined) {
    generateSolid(h_input_.get(), width_, height_, 100);

    alpha_ = 1.5f;
    beta_ = 30.0f;
    runAndDownload();

    for (size_t i = 0; i < size_; ++i) {
        int expected = static_cast<int>(1.5f * 100.0f + 30.0f);
        expected = std::min(255, std::max(0, expected));
        EXPECT_EQ(h_output_[i], static_cast<unsigned char>(expected));
    }
}

TEST_F(BrightnessTest, ClampingAtUpperBound) {
    generateSolid(h_input_.get(), width_, height_, 200);
    std::memset(h_expected_.get(), 255, size_);

    alpha_ = 1.5f;
    beta_ = 100.0f;
    runAndDownload();

    EXPECT_TRUE(compareBuffers(h_expected_.get(), h_output_.get(), size_));
}

TEST_F(BrightnessTest, ClampingAtLowerBound) {
    generateSolid(h_input_.get(), width_, height_, 50);
    std::memset(h_expected_.get(), 0, size_);

    alpha_ = 0.5f;
    beta_ = -100.0f;
    runAndDownload();

    EXPECT_TRUE(compareBuffers(h_expected_.get(), h_output_.get(), size_));
}
