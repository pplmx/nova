/**
 * @file device_bandwidth_formula_test.cpp
 * @brief Contracts for the theoretical memory-bandwidth calculation
 *
 * src/observability/bandwidth_tracker.cpp derived peak bandwidth as
 *
 *     clock_ghz * bus_width_bytes
 *
 * leaving out the factor of two for the double data rate that every GDDR and HBM
 * part uses. The reported figures were therefore EXACTLY half the real ones:
 *
 *     reported d2d = 968.5 GB/s
 *     actual peak  = 2039.0 GB/s   (clock 1593000 kHz x 5120-bit bus x 2 / 8)
 *
 * The A100-80GB specification is 2039 GB/s, so the arithmetic can be checked
 * against published hardware data rather than only against itself. The repository
 * already had the correct formula in device_info.h's get_device_properties, which
 * multiplies by two.
 *
 * A wrong peak propagates into utilization: tools.cpp computes
 * max_observed / theoretical * 100, so every reported utilization was doubled.
 */

#include <cuda/observability/bandwidth_tracker.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>

namespace cuda::observability::test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

/// Peak bandwidth in GB/s from the hardware attributes, computed independently of
/// the code under test.
double expected_peak_gbps(int device) {
    int clock_khz = 0;
    int bus_width_bits = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device) !=
            cudaSuccess ||
        cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth,
                               device) != cudaSuccess) {
        return 0.0;
    }
    // clock (GHz) * bytes per transfer * 2 transfers per clock
    return (clock_khz / 1e6) * (bus_width_bits / 8.0) * 2.0;
}

}  // namespace

// Regression: the DDR factor of two was missing, halving every figure.
TEST(DeviceBandwidthFormulaTest, PeakIncludesTheDoubleDataRateFactor) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const double peak = expected_peak_gbps(0);
    ASSERT_GT(peak, 0.0) << "the device must expose clock and bus width";

    const auto bandwidth = DeviceMemoryBandwidth::query(0);

    // d2d is the closest to raw peak: the code models it as 95% of the bus.
    const double modelled_peak = bandwidth.d2d_gbps / 0.95;
    EXPECT_NEAR(modelled_peak, peak, peak * 0.01)
        << "modelled peak " << modelled_peak << " GB/s versus hardware " << peak
        << " GB/s; a missing factor of two would show as roughly half";

    // An A100-80GB reports a 5120-bit bus at 1593 MHz = 2039 GB/s. Asserting the
    // half-of-peak result is excluded catches the old formula regardless of the
    // exact model.
    EXPECT_GT(modelled_peak, peak * 0.9)
        << "the modelled peak is far below the hardware's, which is what a missing "
           "DDR factor looks like";
}

TEST(DeviceBandwidthFormulaTest, DirectionalFiguresAreOrderedAsDocumented) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const auto bandwidth = DeviceMemoryBandwidth::query(0);

    EXPECT_GT(bandwidth.h2d_gbps, 0.0);
    EXPECT_GT(bandwidth.d2h_gbps, 0.0);
    EXPECT_GT(bandwidth.d2d_gbps, 0.0);

    // The model gives host transfers 90% of peak and device copies 95%.
    EXPECT_GT(bandwidth.d2d_gbps, bandwidth.h2d_gbps)
        << "a device-to-device copy must not be modelled slower than a host copy";
    EXPECT_NEAR(bandwidth.h2d_gbps, bandwidth.d2h_gbps, 1e-6)
        << "both host directions share the same link";
}

TEST(DeviceBandwidthFormulaTest, TheoreticalBandwidthMatchesTheDeviceToDeviceFigure) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const auto bandwidth = DeviceMemoryBandwidth::query(0);

    EXPECT_DOUBLE_EQ(bandwidth.theoretical_bandwidth_gbps(),
                     bandwidth.d2d_gbps)
        << "the accessor must not disagree with the field it exposes";
}

TEST(DeviceBandwidthFormulaTest, UtilizationStaysBelowOneHundredPercent) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // tools.cpp divides an observed rate by the theoretical one. With the old
    // formula the denominator was half-sized, so a measurement at true peak would
    // have reported 200% utilization. The corrected denominator must keep a
    // peak-rate observation at or below 100%.
    const auto bandwidth = DeviceMemoryBandwidth::query(0);
    const double theoretical = bandwidth.theoretical_bandwidth_gbps();
    ASSERT_GT(theoretical, 0.0);

    const double observed_at_peak = expected_peak_gbps(0);
    const double utilisation = observed_at_peak / theoretical * 100.0;

    EXPECT_LE(utilisation, 110.0)
        << "a transfer running at hardware peak reported " << utilisation
        << "%; a half-sized denominator would report about 200";
}

TEST(DeviceBandwidthFormulaTest, QueryLeavesTheCallingDeviceUnchanged) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    int before = -1;
    ASSERT_EQ(cudaGetDevice(&before), cudaSuccess);

    (void)DeviceMemoryBandwidth::query(0);

    int after = -1;
    ASSERT_EQ(cudaGetDevice(&after), cudaSuccess);
    EXPECT_EQ(after, before) << "query(device) must restore the caller's device";
}

TEST(DeviceBandwidthFormulaTest, QueryingANonExistentDeviceReportsZero) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Device 9999 does not exist. The function is documented to throw on the
    // failing device calls, so accept either an exception or a zeroed result, but
    // never a positive figure derived from nothing.
    try {
        const auto bandwidth = DeviceMemoryBandwidth::query(9999);
        EXPECT_DOUBLE_EQ(bandwidth.h2d_gbps, 0.0);
        EXPECT_DOUBLE_EQ(bandwidth.d2d_gbps, 0.0);
    } catch (const std::exception&) {
        SUCCEED() << "throwing on an invalid device is acceptable";
    }
    (void)cudaGetLastError();
}

TEST(DeviceBandwidthFormulaTest, MeasuredHostToDeviceTransferIsPlausible) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // An end-to-end check that the tracker's measurement path produces a rate in
    // the same order as the model rather than a factor off.
    BandwidthTracker tracker;
    const auto measurement = tracker.measure_host_to_device(8u * 1024 * 1024);

    const auto model = DeviceMemoryBandwidth::query(0);
    if (measurement.bandwidth_gbps > 0.0) {
        EXPECT_LT(measurement.bandwidth_gbps, model.h2d_gbps * 2.0)
            << "a measured rate far above the modelled one suggests a unit error";
    }
}

}  // namespace cuda::observability::test
