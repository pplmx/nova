#include <gtest/gtest.h>
#include <memory>
#include <cstddef>
#include <cstdint>

#include "image_utils.h"
#include "sobel_edge.h"
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
