#include <gtest/gtest.h>
#include <memory>
#include <cstddef>
#include <cstdint>

#include "image/types.h"
#include "image/sobel_edge.h"
#include "test_patterns.cuh"

class SobelTest : public ::testing::Test {
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

    void runAndDownload(float threshold = 30.0f) {
        CUDA_CHECK_IMAGE(cudaMemcpy(d_input_, h_input_.get(), size_, cudaMemcpyHostToDevice));
        sobelEdgeDetection(d_input_, d_output_, width_, height_, threshold);
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

TEST_F(SobelTest, SolidImageHasNoEdges) {
    generateSolid(h_input_.get(), width_, height_, 128);
    runAndDownload(30.0f);

    int edgeCount = 0;
    for (size_t i = 0; i < size_; i += 3) {
        if (h_output_[i] > 0) {
            edgeCount++;
        }
    }
    EXPECT_EQ(edgeCount, 0);
}

TEST_F(SobelTest, CheckerboardHasEdges) {
    generateCheckerboard(h_input_.get(), width_, height_, 8);
    runAndDownload(30.0f);

    int edgeCount = 0;
    for (size_t i = 0; i < size_; i += 3) {
        if (h_output_[i] > 0) {
            edgeCount++;
        }
    }
    EXPECT_GT(edgeCount, 0);
}

TEST_F(SobelTest, HighThresholdSuppressesEdges) {
    generateCheckerboard(h_input_.get(), width_, height_, 8);

    runAndDownload(1200.0f);

    int edgeCount = 0;
    for (size_t i = 0; i < size_; i += 3) {
        if (h_output_[i] > 0) {
            edgeCount++;
        }
    }
    EXPECT_LT(edgeCount, 100);
}

TEST_F(SobelTest, LowThresholdFindsMoreEdges) {
    generateCheckerboard(h_input_.get(), width_, height_, 8);

    runAndDownload(10.0f);

    int edgeCount = 0;
    for (size_t i = 0; i < size_; i += 3) {
        if (h_output_[i] > 0) {
            edgeCount++;
        }
    }
    EXPECT_GT(edgeCount, 100);
}

TEST_F(SobelTest, SinglePixel) {
    size_ = 3;
    h_input_ = std::make_unique<unsigned char[]>(size_);
    h_output_ = std::make_unique<unsigned char[]>(size_);

    h_input_.get()[0] = 128;

    uint8_t *d_input, *d_output;
    CUDA_CHECK_IMAGE(cudaMalloc(&d_input, size_));
    CUDA_CHECK_IMAGE(cudaMalloc(&d_output, size_));

    CUDA_CHECK_IMAGE(cudaMemcpy(d_input, h_input_.get(), size_, cudaMemcpyHostToDevice));
    sobelEdgeDetection(d_input, d_output, 1, 1, 30.0f);
    CUDA_CHECK_IMAGE(cudaMemcpy(h_output_.get(), d_output, size_, cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_output_.get()[0], 0);

    CUDA_CHECK_IMAGE(cudaFree(d_input));
    CUDA_CHECK_IMAGE(cudaFree(d_output));
}

TEST_F(SobelTest, EdgeAtBoundary) {
    width_ = 16;
    height_ = 16;
    size_ = width_ * height_ * 3;

    h_input_ = std::make_unique<unsigned char[]>(size_);
    h_output_ = std::make_unique<unsigned char[]>(size_);

    for (size_t y = 0; y < height_; ++y) {
        for (size_t x = 0; x < width_; ++x) {
            size_t idx = (y * width_ + x) * 3;
            unsigned char val = (x < 8) ? 0 : 255;
            h_input_[idx] = h_input_[idx + 1] = h_input_[idx + 2] = val;
        }
    }

    uint8_t *d_input, *d_output;
    CUDA_CHECK_IMAGE(cudaMalloc(&d_input, size_));
    CUDA_CHECK_IMAGE(cudaMalloc(&d_output, size_));

    CUDA_CHECK_IMAGE(cudaMemcpy(d_input, h_input_.get(), size_, cudaMemcpyHostToDevice));
    sobelEdgeDetection(d_input, d_output, width_, height_, 30.0f);
    CUDA_CHECK_IMAGE(cudaMemcpy(h_output_.get(), d_output, size_, cudaMemcpyDeviceToHost));

    int edgeCount = 0;
    for (size_t y = 0; y < height_; ++y) {
        for (size_t x = 0; x < width_; ++x) {
            size_t idx = (y * width_ + x) * 3;
            if (h_output_[idx] > 0) edgeCount++;
        }
    }
    EXPECT_GT(edgeCount, 0);

    CUDA_CHECK_IMAGE(cudaFree(d_input));
    CUDA_CHECK_IMAGE(cudaFree(d_output));
}

TEST_F(SobelTest, HighThreshold) {
    generateCheckerboard(h_input_.get(), width_, height_, 8);

    runAndDownload(10.0f);
    int lowThresholdEdges = 0;
    for (size_t i = 0; i < size_; i += 3) {
        if (h_output_[i] > 0) lowThresholdEdges++;
    }

    runAndDownload(10000.0f);
    int highThresholdEdges = 0;
    for (size_t i = 0; i < size_; i += 3) {
        if (h_output_[i] > 0) highThresholdEdges++;
    }

    EXPECT_LE(highThresholdEdges, lowThresholdEdges);
}
