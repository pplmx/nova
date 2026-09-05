#include <cuda/quantize/calibrator.hpp>
#include <cuda/quantize/int8_kernels.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdint>

namespace nova {
namespace quantize {
namespace test {

class CalibratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA not available";
        }
    }

    std::vector<float> create_test_data(float min_val, float max_val, size_t n) {
        std::vector<float> data(n);
        for (size_t i = 0; i < n; ++i) {
            float t = static_cast<float>(i) / n;
            data[i] = min_val + t * (max_val - min_val);
        }
        return data;
    }
};

TEST_F(CalibratorTest, MinMaxSymmetricCalibration) {
    size_t n = 1024;
    auto data = create_test_data(-10.0f, 10.0f, n);

    MinMaxCalibrator calibrator(true);
    auto result = calibrator.calibrate(data.data(), n);

    EXPECT_NEAR(result.scale, 10.0f / 127.0f, 0.01f);
    EXPECT_NEAR(result.zero_point, 0.0f, 0.01f);
    EXPECT_TRUE(result.symmetric);
}

TEST_F(CalibratorTest, MinMaxAsymmetricCalibration) {
    size_t n = 1024;
    auto data = create_test_data(-5.0f, 15.0f, n);

    MinMaxCalibrator calibrator(false);
    auto result = calibrator.calibrate(data.data(), n);

    EXPECT_NEAR(result.scale, 20.0f / 254.0f, 0.01f);
    EXPECT_TRUE(!result.symmetric);
}

TEST_F(CalibratorTest, MinMaxCacheRoundtrip) {
    size_t n = 1024;
    auto data = create_test_data(-8.0f, 12.0f, n);

    MinMaxCalibrator calibrator(true);
    calibrator.calibrate(data.data(), n);

    float expected_min = calibrator.get_min();
    float expected_max = calibrator.get_max();

    calibrator.save_cache("/tmp/minmax_cache.bin");

    MinMaxCalibrator calibrator2(true);
    calibrator2.load_cache("/tmp/minmax_cache.bin");

    EXPECT_NEAR(calibrator2.get_min(), expected_min, 0.001f);
    EXPECT_NEAR(calibrator2.get_max(), expected_max, 0.001f);
}

TEST_F(CalibratorTest, HistogramCalibration) {
    size_t n = 1024;
    auto data = create_test_data(-10.0f, 10.0f, n);

    HistogramCalibrator calibrator(256, 99.99f, true);
    auto result = calibrator.calibrate(data.data(), n);

    EXPECT_GT(result.scale, 0.0f);
    EXPECT_TRUE(result.symmetric);
    EXPECT_EQ(calibrator.get_num_bins(), 256);
}

TEST_F(CalibratorTest, HistogramFullCoverage_2048Bins) {
    // Regression (TASK-076): build_histogram_kernel used a fixed 256-entry
    // shared buffer but binned into up to num_bins-1 == 2047 — an out-of-bounds
    // shared-memory write, and bins >= 256 were never flushed to the global
    // histogram, so calibration over the DEFAULT 2048 bins silently under-
    // counted (~7/8 of the values were lost and the percentile threshold was
    // wrong). The kernel now sizes its shared histogram to num_bins.
    const size_t n = 4096;
    const float lo = -50.0f, hi = 50.0f;
    std::vector<float> data = create_test_data(lo, hi, n);

    // build_histogram's histogram argument must be a DEVICE buffer (the host
    // wrapper memsets it with cudaMemset); the calibrator uses a device one.
    uint32_t* d_hist = nullptr;
    ASSERT_EQ(cudaMalloc(&d_hist, 2048 * sizeof(uint32_t)), cudaSuccess);
    std::vector<uint32_t> hist(2048);
    cuda::build_histogram(data.data(), d_hist, n, lo, hi, 2048, 0);
    ASSERT_EQ(cudaMemcpy(hist.data(), d_hist, 2048 * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_hist), cudaSuccess);

    uint64_t total = 0;
    for (uint32_t c : hist) {
        total += c;
    }
    EXPECT_EQ(total, n)
        << "every value must land in one of the 2048 bins (the old kernel "
           "dropped every value whose bin was >= 256)";

    // Values spread across the full range must populate bins far past 256.
    uint64_t tail = 0;
    for (int b = 256; b < 2048; ++b) {
        tail += hist[b];
    }
    EXPECT_GT(tail, 0u) << "bins 256..2047 must receive counts";

    // And a default (2048-bin) calibrate() must neither crash (the OOB shared
    // write) nor produce a degenerate scale.
    HistogramCalibrator calibrator;  // default num_bins = 2048
    auto result = calibrator.calibrate(data.data(), n);
    EXPECT_GT(result.scale, 0.0f);
    EXPECT_TRUE(result.symmetric);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(CalibratorTest, HistogramPercentileSelection) {
    size_t n = 10000;
    std::vector<float> data(n);

    for (size_t i = 0; i < n; ++i) {
        data[i] = (rand() % 100 < 95) ?
            static_cast<float>(rand()) / RAND_MAX * 10.0f :
            static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }

    HistogramCalibrator calibrator1(256, 99.0f, true);
    auto result1 = calibrator1.calibrate(data.data(), n);

    HistogramCalibrator calibrator2(256, 99.9f, true);
    auto result2 = calibrator2.calibrate(data.data(), n);

    EXPECT_GE(result2.scale, result1.scale);
}

TEST_F(CalibratorTest, MSECalibration) {
    size_t n = 1024;
    auto data = create_test_data(-5.0f, 5.0f, n);

    MSECalibrator calibrator(true);
    auto result = calibrator.calibrate(data.data(), n);

    EXPECT_GT(result.scale, 0.0f);
    EXPECT_TRUE(result.symmetric);
}

TEST_F(CalibratorTest, PerChannelCalibration) {
    size_t n = 256;
    auto data = create_test_data(-10.0f, 10.0f, n);

    PerChannelCalibrator calibrator(0, true);
    std::vector<int> shape = {2, 128};
    auto results = calibrator.calibrate_per_channel(data.data(), shape);

    EXPECT_EQ(results.size(), 2u);
    for (const auto& result : results) {
        EXPECT_GT(result.scale, 0.0f);
        EXPECT_TRUE(result.symmetric);
    }
}

TEST_F(CalibratorTest, SmallValueHandling) {
    size_t n = 100;
    std::vector<float> data(n, 0.001f);

    MinMaxCalibrator calibrator(true);
    auto result = calibrator.calibrate(data.data(), n);

    EXPECT_GE(result.scale, 1e-6f);
}

TEST_F(CalibratorTest, ZeroRangeHandling) {
    size_t n = 100;
    std::vector<float> data(n, 5.0f);

    MinMaxCalibrator calibrator(true);
    auto result = calibrator.calibrate(data.data(), n);

    EXPECT_GE(result.scale, 1e-6f);
}

TEST_F(CalibratorTest, HistogramCalibratorDefaultConstruction) {
    HistogramCalibrator calibrator;
    EXPECT_EQ(calibrator.get_num_bins(), 2048);
    EXPECT_NEAR(calibrator.get_percentile(), 99.99f, 0.01f);
}

TEST_F(CalibratorTest, HistogramCalibratorCustomParameters) {
    HistogramCalibrator calibrator(512, 99.9f, false);
    EXPECT_EQ(calibrator.get_num_bins(), 512);
    EXPECT_NEAR(calibrator.get_percentile(), 99.9f, 0.01f);
}

TEST_F(CalibratorTest, MSECalibratorConstantData) {
    size_t n = 100;
    std::vector<float> data(n, 1.0f);

    MSECalibrator calibrator(true);
    auto result = calibrator.calibrate(data.data(), n);

    EXPECT_GE(result.scale, 1e-6f);
    EXPECT_TRUE(result.symmetric);
}

TEST_F(CalibratorTest, PerChannelCalibratorChannelDim) {
    PerChannelCalibrator calibrator(1, true);
    EXPECT_EQ(calibrator.get_channel_dim(), 1);
}

TEST_F(CalibratorTest, PerChannelCalibratorBatched) {
    size_t n = 512;
    auto data = create_test_data(-10.0f, 10.0f, n);

    PerChannelCalibrator calibrator(0, true);
    std::vector<int> shape = {4, 128};
    auto results = calibrator.calibrate_per_channel(data.data(), shape);

    EXPECT_EQ(results.size(), 4u);
}

} // namespace test
} // namespace quantize
} // namespace nova
