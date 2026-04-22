#include <gtest/gtest.h>
#include "parallel/histogram.h"
#include "cuda/device/device_utils.h"
#include <vector>

class HistogramTest : public ::testing::Test {
protected:
    size_t width_ = 32;
    size_t height_ = 32;
    size_t size_;
    size_t pixelCount_;
    std::vector<uint8_t> h_input_;
    std::vector<uint32_t> h_histogram_;
    uint8_t *d_input_ = nullptr;
    uint32_t *d_histogram_ = nullptr;

    static constexpr int NUM_BINS = 256;

    void SetUp() override {
        pixelCount_ = width_ * height_;
        size_ = pixelCount_ * 3;
        h_input_.resize(size_);
        h_histogram_.resize(NUM_BINS);

        CUDA_CHECK(cudaMalloc(&d_input_, size_ * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_histogram_, NUM_BINS * sizeof(uint32_t)));
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(d_input_));
        CUDA_CHECK(cudaFree(d_histogram_));
    }

    void uploadInput() {
        CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    void downloadHistogram() {
        CUDA_CHECK(cudaMemcpy(h_histogram_.data(), d_histogram_, NUM_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
};

TEST_F(HistogramTest, UniformDistribution) {
    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = 128;
    }
    uploadInput();

    computeHistogram(d_input_, d_histogram_, width_, height_);
    downloadHistogram();

    EXPECT_EQ(h_histogram_[128], size_);
    for (int i = 0; i < NUM_BINS; ++i) {
        if (i != 128) {
            EXPECT_EQ(h_histogram_[i], 0);
        }
    }
}

TEST_F(HistogramTest, AllZeros) {
    h_input_.assign(size_, 0);
    uploadInput();

    computeHistogram(d_input_, d_histogram_, width_, height_);
    downloadHistogram();

    EXPECT_EQ(h_histogram_[0], size_);
}

TEST_F(HistogramTest, AllOnes) {
    h_input_.assign(size_, 1);
    uploadInput();

    computeHistogram(d_input_, d_histogram_, width_, height_);
    downloadHistogram();

    EXPECT_EQ(h_histogram_[1], size_);
}

TEST_F(HistogramTest, MixedValues) {
    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = static_cast<uint8_t>(i % 256);
    }
    uploadInput();

    computeHistogram(d_input_, d_histogram_, width_, height_);
    downloadHistogram();

    uint64_t total = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        total += h_histogram_[i];
    }
    EXPECT_EQ(total, size_);
}

TEST_F(HistogramTest, CountsMatchTotal) {
    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = static_cast<uint8_t>(i % 100);
    }
    uploadInput();

    computeHistogram(d_input_, d_histogram_, width_, height_);
    downloadHistogram();

    uint64_t total = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        total += h_histogram_[i];
    }
    EXPECT_EQ(total, static_cast<uint64_t>(size_));
}

TEST_F(HistogramTest, PerChannelBasic) {
    uint32_t *d_hist_r, *d_hist_g, *d_hist_b;
    CUDA_CHECK(cudaMalloc(&d_hist_r, NUM_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hist_g, NUM_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hist_b, NUM_BINS * sizeof(uint32_t)));

    for (size_t i = 0; i < pixelCount_; ++i) {
        size_t base = i * 3;
        h_input_[base] = 100;
        h_input_[base + 1] = 150;
        h_input_[base + 2] = 200;
    }
    uploadInput();

    computeHistogramPerChannel(d_input_, d_hist_r, d_hist_g, d_hist_b, width_, height_);

    std::vector<uint32_t> h_r(NUM_BINS), h_g(NUM_BINS), h_b(NUM_BINS);
    CUDA_CHECK(cudaMemcpy(h_r.data(), d_hist_r, NUM_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_g.data(), d_hist_g, NUM_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_hist_b, NUM_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_r[100], pixelCount_);
    EXPECT_EQ(h_g[150], pixelCount_);
    EXPECT_EQ(h_b[200], pixelCount_);

    CUDA_CHECK(cudaFree(d_hist_r));
    CUDA_CHECK(cudaFree(d_hist_g));
    CUDA_CHECK(cudaFree(d_hist_b));
}

TEST_F(HistogramTest, PerChannelSum) {
    for (size_t i = 0; i < pixelCount_; ++i) {
        size_t base = i * 3;
        h_input_[base] = static_cast<uint8_t>(i % 256);
        h_input_[base + 1] = static_cast<uint8_t>((i + 50) % 256);
        h_input_[base + 2] = static_cast<uint8_t>((i + 100) % 256);
    }
    uploadInput();

    uint32_t *d_hist_r, *d_hist_g, *d_hist_b;
    CUDA_CHECK(cudaMalloc(&d_hist_r, NUM_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hist_g, NUM_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hist_b, NUM_BINS * sizeof(uint32_t)));

    computeHistogramPerChannel(d_input_, d_hist_r, d_hist_g, d_hist_b, width_, height_);

    std::vector<uint32_t> h_r(NUM_BINS), h_g(NUM_BINS), h_b(NUM_BINS);
    CUDA_CHECK(cudaMemcpy(h_r.data(), d_hist_r, NUM_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_g.data(), d_hist_g, NUM_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_hist_b, NUM_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint64_t total_r = 0, total_g = 0, total_b = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        total_r += h_r[i];
        total_g += h_g[i];
        total_b += h_b[i];
    }

    EXPECT_EQ(total_r, pixelCount_);
    EXPECT_EQ(total_g, pixelCount_);
    EXPECT_EQ(total_b, pixelCount_);

    CUDA_CHECK(cudaFree(d_hist_r));
    CUDA_CHECK(cudaFree(d_hist_g));
    CUDA_CHECK(cudaFree(d_hist_b));
}

TEST_F(HistogramTest, EqualizeOutputRange) {
    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = static_cast<uint8_t>(i % 256);
    }
    uploadInput();

    computeHistogram(d_input_, d_histogram_, width_, height_);
    downloadHistogram();

    CUDA_CHECK(cudaMemcpy(d_histogram_, h_histogram_.data(), NUM_BINS * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint8_t *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, size_ * sizeof(uint8_t)));

    equalizeHistogram(d_input_, d_output, d_histogram_, width_, height_);

    std::vector<uint8_t> h_output(size_);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_ * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_GE(h_output[i], 0);
        EXPECT_LE(h_output[i], 255);
    }

    CUDA_CHECK(cudaFree(d_output));
}

TEST_F(HistogramTest, EqualizeChangesImage) {
    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = static_cast<uint8_t>(i % 256);
    }
    uploadInput();

    computeHistogram(d_input_, d_histogram_, width_, height_);
    CUDA_CHECK(cudaMemcpy(d_histogram_, h_histogram_.data(), NUM_BINS * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint8_t *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, size_ * sizeof(uint8_t)));

    equalizeHistogram(d_input_, d_output, d_histogram_, width_, height_);

    std::vector<uint8_t> h_output(size_);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_ * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    bool changed = false;
    for (size_t i = 0; i < size_; ++i) {
        if (h_output[i] != h_input_[i]) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed);

    CUDA_CHECK(cudaFree(d_output));
}

TEST_F(HistogramTest, LargeImage) {
    width_ = 512;
    height_ = 512;
    pixelCount_ = width_ * height_;
    size_ = pixelCount_ * 3;
    h_input_.resize(size_);

    CUDA_CHECK(cudaFree(d_input_));
    CUDA_CHECK(cudaFree(d_histogram_));
    CUDA_CHECK(cudaMalloc(&d_input_, size_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_histogram_, NUM_BINS * sizeof(uint32_t)));

    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = static_cast<uint8_t>(i % 256);
    }
    uploadInput();

    computeHistogram(d_input_, d_histogram_, width_, height_);
    downloadHistogram();

    uint64_t total = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        total += h_histogram_[i];
    }
    EXPECT_EQ(total, size_);
}
