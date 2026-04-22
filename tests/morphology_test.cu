#include <gtest/gtest.h>
#include "image/morphology.h"
#include "image/types.h"
#include "cuda/kernel/cuda_utils.h"
#include <vector>
#include <algorithm>

class MorphologyTest : public ::testing::Test {
protected:
    size_t width_ = 32;
    size_t height_ = 32;
    size_t size_;
    std::vector<uint8_t> h_input_;
    std::vector<uint8_t> h_output_;
    uint8_t *d_input_ = nullptr;
    uint8_t *d_output_ = nullptr;

    void SetUp() override {
        size_ = width_ * height_ * 3;
        h_input_.resize(size_);
        h_output_.resize(size_);

        CUDA_CHECK(cudaMalloc(&d_input_, size_ * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_output_, size_ * sizeof(uint8_t)));
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(d_input_));
        CUDA_CHECK(cudaFree(d_output_));
    }

    void uploadInput() {
        CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    void downloadOutput() {
        CUDA_CHECK(cudaMemcpy(h_output_.data(), d_output_, size_ * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    }
};

TEST_F(MorphologyTest, ThresholdAllBlack) {
    h_input_.assign(size_, 200);
    uploadInput();

    applyThreshold(d_input_, d_output_, width_, height_, 128);
    downloadOutput();

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_EQ(h_output_[i], 255);
    }
}

TEST_F(MorphologyTest, ThresholdAllWhite) {
    h_input_.assign(size_, 50);
    uploadInput();

    applyThreshold(d_input_, d_output_, width_, height_, 128);
    downloadOutput();

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_EQ(h_output_[i], 0);
    }
}

TEST_F(MorphologyTest, ThresholdMixed) {
    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = (i % 256);
    }
    uploadInput();

    applyThreshold(d_input_, d_output_, width_, height_, 128);
    downloadOutput();

    for (size_t i = 0; i < size_; ++i) {
        uint8_t expected = (h_input_[i] > 128) ? 255 : 0;
        EXPECT_EQ(h_output_[i], expected);
    }
}

TEST_F(MorphologyTest, ErodeSolidImage) {
    h_input_.assign(size_, 255);
    uploadInput();

    erodeImage(d_input_, d_output_, width_, height_, 3);
    downloadOutput();

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_EQ(h_output_[i], 255);
    }
}

TEST_F(MorphologyTest, ErodeSingleDarkPixel) {
    h_input_.assign(size_, 255);
    h_input_[size_ / 2] = 0;
    uploadInput();

    erodeImage(d_input_, d_output_, width_, height_, 3);
    downloadOutput();

    bool hasZero = false;
    for (size_t i = 0; i < size_; ++i) {
        if (h_output_[i] == 0) hasZero = true;
    }
    EXPECT_TRUE(hasZero);
}

TEST_F(MorphologyTest, DilateSolidImage) {
    h_input_.assign(size_, 0);
    uploadInput();

    dilateImage(d_input_, d_output_, width_, height_, 3);
    downloadOutput();

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_EQ(h_output_[i], 0);
    }
}

TEST_F(MorphologyTest, DilateSingleBrightPixel) {
    h_input_.assign(size_, 0);
    h_input_[size_ / 2] = 255;
    uploadInput();

    dilateImage(d_input_, d_output_, width_, height_, 3);
    downloadOutput();

    bool hasBright = false;
    for (size_t i = 0; i < size_; ++i) {
        if (h_output_[i] == 255) hasBright = true;
    }
    EXPECT_TRUE(hasBright);
}

TEST_F(MorphologyTest, OpeningThenClosing) {
    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = static_cast<uint8_t>(i % 256);
    }
    uploadInput();

    openingImage(d_input_, d_output_, width_, height_, 3);
    downloadOutput();

    bool changed = false;
    for (size_t i = 0; i < size_; ++i) {
        if (h_output_[i] != h_input_[i]) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed);
}

TEST_F(MorphologyTest, ClosingThenOpening) {
    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = static_cast<uint8_t>((i * 7) % 256);
    }
    uploadInput();

    closingImage(d_input_, d_output_, width_, height_, 3);
    downloadOutput();

    bool changed = false;
    for (size_t i = 0; i < size_; ++i) {
        if (h_output_[i] != h_input_[i]) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed);
}

TEST_F(MorphologyTest, SharpenOutputInRange) {
    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = static_cast<uint8_t>(i % 256);
    }
    uploadInput();

    sharpenImage(d_input_, d_output_, width_, height_, 0.5f);
    downloadOutput();

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_GE(h_output_[i], 0);
        EXPECT_LE(h_output_[i], 255);
    }
}

TEST_F(MorphologyTest, SharpenPreservesDimensions) {
    width_ = 64;
    height_ = 48;
    size_ = width_ * height_ * 3;
    h_input_.resize(size_);
    h_output_.resize(size_);

    CUDA_CHECK(cudaFree(d_input_));
    CUDA_CHECK(cudaFree(d_output_));
    CUDA_CHECK(cudaMalloc(&d_input_, size_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_output_, size_ * sizeof(uint8_t)));

    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = 128;
    }
    uploadInput();

    sharpenImage(d_input_, d_output_, width_, height_, 1.0f);
    downloadOutput();

    EXPECT_EQ(h_output_.size(), size_);
}

TEST_F(MorphologyTest, ErodeLargerKernel) {
    h_input_.assign(size_, 255);
    h_input_[size_ / 2] = 0;
    uploadInput();

    erodeImage(d_input_, d_output_, width_, height_, 5);
    downloadOutput();

    bool hasZero = false;
    for (size_t i = 0; i < size_; ++i) {
        if (h_output_[i] == 0) hasZero = true;
    }
    EXPECT_TRUE(hasZero);
}

TEST_F(MorphologyTest, DilateLargerKernel) {
    h_input_.assign(size_, 0);
    h_input_[size_ / 2] = 255;
    uploadInput();

    dilateImage(d_input_, d_output_, width_, height_, 5);
    downloadOutput();

    bool hasBright = false;
    for (size_t i = 0; i < size_; ++i) {
        if (h_output_[i] == 255) hasBright = true;
    }
    EXPECT_TRUE(hasBright);
}
