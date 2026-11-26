#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <gtest/gtest.h>
#include <stdexcept>
#include "image/morphology.h"
#include "image/types.h"
#include "cuda/device/device_utils.h"
#include <vector>
#include <algorithm>

class MorphologyTest : public ::testing::Test {
protected:
    size_t width_ = 32;
    size_t height_ = 32;
    size_t size_;
    std::vector<uint8_t> h_input_;
    std::vector<uint8_t> h_output_;
    uint8_t* d_input_ = nullptr;
    uint8_t* d_output_ = nullptr;

    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        (void)cudaDeviceSynchronize();

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

    void uploadInput() { CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(uint8_t), cudaMemcpyHostToDevice)); }

    void downloadOutput() { CUDA_CHECK(cudaMemcpy(h_output_.data(), d_output_, size_ * sizeof(uint8_t), cudaMemcpyDeviceToHost)); }
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

// The four morphology entry points in src/image/morphology.cu take kernel_size with no validation:
//
//     int half = kernel_size / 2;          // -3 -> -1
//     for (int dy = -half; dy <= half; ...) // never runs
//     minVal stays at its 255 initialiser    // erode writes an all-white image
//
// A negative kernel therefore produces an all-white erosion (all-black dilation) and reports
// success, which is a silently wrong result rather than a diagnosable error. kernel_size = 2 does
// real 3x3 work while the caller asked for a 2-wide window.
//
// The sibling filters in this tree do check: gaussianBlur rejects `kernel_size <= 0 || even`, and
// conv2d's require_odd_kernel_size is shared by three entry points. These four were the ones left
// out, so the same parameter is validated on one side of the image layer and trusted on the other.
TEST_F(MorphologyTest, MorphologyRejectsANonPositiveKernelSize) {
    h_input_.assign(size_, 128);
    uploadInput();

    EXPECT_THROW(erodeImage(d_input_, d_output_, width_, height_, -3), std::invalid_argument) << "a negative kernel collapsed the window and returned an all-white image";
    EXPECT_THROW(erodeImage(d_input_, d_output_, width_, height_, 0), std::invalid_argument);
    EXPECT_THROW(dilateImage(d_input_, d_output_, width_, height_, -3), std::invalid_argument) << "the same window collapse made dilation write an all-black image";
    EXPECT_THROW(dilateImage(d_input_, d_output_, width_, height_, 0), std::invalid_argument);
    EXPECT_THROW(openingImage(d_input_, d_output_, width_, height_, -3), std::invalid_argument);
    EXPECT_THROW(closingImage(d_input_, d_output_, width_, height_, -3), std::invalid_argument);
}

TEST_F(MorphologyTest, ErodeSingleDarkPixel) {
    h_input_.assign(size_, 255);
    h_input_[size_ / 2] = 0;
    uploadInput();

    erodeImage(d_input_, d_output_, width_, height_, 3);
    downloadOutput();

    bool hasZero = false;
    for (size_t i = 0; i < size_; ++i) {
        if (h_output_[i] == 0) {
            hasZero = true;
        }
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
        if (h_output_[i] == 255) {
            hasBright = true;
        }
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

    // The two assertions here used to be EXPECT_GE(h_output_[i], 0) and EXPECT_LE(..., 255) on a
    // std::vector<uint8_t>, which are true for every possible byte: sharpenImage could have written
    // nothing, written zeros, or written uninitialised memory and this case stayed green.
    //
    // The input is a sawtooth (i % 256), which has hard discontinuities every 256 pixels, so the
    // sharpened image must differ from it somewhere. Counting the changes distinguishes a sharpen
    // that ran from one that returned the input untouched or wrote a constant.
    size_t changed = 0;
    for (size_t i = 0; i < size_; ++i) {
        if (h_output_[i] != h_input_[i]) {
            ++changed;
        }
    }
    EXPECT_GT(changed, 0u) << "sharpenImage did not change a sawtooth input, so it did nothing";
    EXPECT_LT(changed, size_) << "every pixel changed, which a Laplacian sharpen does not do to flat regions";
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

    // The only assertion here used to be that the output vector kept its size, which the fixture
    // guarantees. A uniform image has a zero Laplacian everywhere, so any strength must leave it
    // untouched -- a sharpen that scaled, inverted or cleared the image fails this.
    for (size_t i = 0; i < size_; ++i) {
        EXPECT_EQ(h_output_[i], 128) << "a uniform image has no edges to sharpen (index " << i << ")";
    }
}

TEST_F(MorphologyTest, ErodeLargerKernel) {
    h_input_.assign(size_, 255);
    h_input_[size_ / 2] = 0;
    uploadInput();

    erodeImage(d_input_, d_output_, width_, height_, 5);
    downloadOutput();

    bool hasZero = false;
    for (size_t i = 0; i < size_; ++i) {
        if (h_output_[i] == 0) {
            hasZero = true;
        }
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
        if (h_output_[i] == 255) {
            hasBright = true;
        }
    }
    EXPECT_TRUE(hasBright);
}
