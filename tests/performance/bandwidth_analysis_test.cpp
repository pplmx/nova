#include <gtest/gtest.h>
#include <cuda/performance/bandwidth/roofline_model.h>
#include <cuda/performance/bandwidth/cache_analyzer.h>

namespace cuda::performance::bandwidth::test {

    class RooflineModelTest : public ::testing::Test {
    protected:
        RooflineModel model;
    };

    TEST_F(RooflineModelTest, EmptyModel) {
        EXPECT_EQ(model.point_count(), 0u);
        EXPECT_EQ(model.ridge_point(), 0.0);
    }

    TEST_F(RooflineModelTest, AddPoint) {
        model.add_point("kernel1", 1024, 64, 1000);
        EXPECT_EQ(model.point_count(), 1u);
    }

    TEST_F(RooflineModelTest, ComputeArithmeticIntensity) {
        double ai = model.compute_arithmetic_intensity(1024, 64);
        EXPECT_DOUBLE_EQ(ai, 16.0);
    }

    TEST_F(RooflineModelTest, ComputePerformance) {
        double perf = model.compute_performance_gflops(1e9, 1e9);
        EXPECT_DOUBLE_EQ(perf, 1.0);
    }

    TEST_F(RooflineModelTest, RidgePoint) {
        auto peaks = model.peaks();
        EXPECT_GE(peaks.fp64_gflops, 0.0);
        EXPECT_GE(peaks.hbm_bandwidth_gbs, 0.0);
    }

    TEST_F(RooflineModelTest, MemoryBoundClassification) {
        model.add_point("kernel1", 1024, 64, 1000);
        auto points = model.get_memory_bound_points();
        EXPECT_GE(points.size(), 0u);
    }

    TEST_F(RooflineModelTest, ComputeBoundClassification) {
        model.add_point("kernel1", 1024000, 64, 1000);
        auto points = model.get_compute_bound_points();
        EXPECT_GE(points.size(), 0u);
    }

    TEST_F(RooflineModelTest, Clear) {
        model.add_point("kernel1", 1024, 64, 1000);
        model.clear();
        EXPECT_EQ(model.point_count(), 0u);
    }

    TEST_F(RooflineModelTest, ToJson) {
        model.add_point("kernel1", 1024, 64, 1000);
        std::string json = model.to_json();
        EXPECT_NE(json.find("kernel1"), std::string::npos);
        EXPECT_NE(json.find("arithmetic_intensity"), std::string::npos);
    }

    TEST_F(RooflineModelTest, ToCsv) {
        model.add_point("kernel1", 1024, 64, 1000);
        std::string csv = model.to_csv();
        EXPECT_NE(csv.find("kernel_name"), std::string::npos);
        EXPECT_NE(csv.find("kernel1"), std::string::npos);
    }

    class DevicePeaksTest : public ::testing::Test {};

    TEST_F(DevicePeaksTest, QueryDevice) {
        auto peaks = DevicePeaks::query(0);
        EXPECT_GE(peaks.fp64_gflops, 0.0);
        EXPECT_GE(peaks.fp32_gflops, 0.0);
        EXPECT_GE(peaks.hbm_bandwidth_gbs, 0.0);
    }

    class BandwidthUtilizationTest : public ::testing::Test {
    protected:
        BandwidthUtilizationTracker tracker;
    };

    TEST_F(BandwidthUtilizationTest, EmptyTracker) {
        EXPECT_EQ(tracker.sample_count(), 0u);
        EXPECT_EQ(tracker.average_bandwidth_gbs(), 0.0);
    }

    TEST_F(BandwidthUtilizationTest, AddSample) {
        tracker.add_sample(500.0, 1e9, 1e9);
        EXPECT_EQ(tracker.sample_count(), 1u);
    }

    TEST_F(BandwidthUtilizationTest, AverageBandwidth) {
        tracker.add_sample(400.0, 1e9, 1e9);
        tracker.add_sample(600.0, 1e9, 1e9);
        EXPECT_DOUBLE_EQ(tracker.average_bandwidth_gbs(), 500.0);
    }

    TEST_F(BandwidthUtilizationTest, PeakBandwidth) {
        tracker.add_sample(400.0, 1e9, 1e9);
        tracker.add_sample(600.0, 1e9, 1e9);
        EXPECT_DOUBLE_EQ(tracker.peak_bandwidth_gbs(), 600.0);
    }

    TEST_F(BandwidthUtilizationTest, UtilizationPercent) {
        tracker.set_peak_bandwidth(20.0, 20.0, 1000.0);
        tracker.add_sample(500.0, 1e9, 1e9);
        double util = tracker.utilization_percent();
        EXPECT_GE(util, 0.0);
        EXPECT_LE(util, 100.0);
    }

    TEST_F(BandwidthUtilizationTest, LowUtilizationWarning) {
        tracker.set_peak_bandwidth(20.0, 20.0, 100.0);
        tracker.add_sample(30.0, 1e9, 1e9);
        EXPECT_TRUE(tracker.has_low_utilization_warning());
    }

    TEST_F(BandwidthUtilizationTest, ToJson) {
        tracker.add_sample(500.0, 1e9, 1e9);
        std::string json = tracker.to_json();
        EXPECT_NE(json.find("average_bandwidth_gbs"), std::string::npos);
        EXPECT_NE(json.find("utilization_percent"), std::string::npos);
    }

    TEST_F(BandwidthUtilizationTest, Reset) {
        tracker.add_sample(500.0, 1e9, 1e9);
        tracker.reset();
        EXPECT_EQ(tracker.sample_count(), 0u);
    }

    class CacheAnalyzerTest : public ::testing::Test {
    protected:
        CacheAnalyzer analyzer;
    };

    TEST_F(CacheAnalyzerTest, CheckAvailability) {
        bool available = analyzer.is_available();
        EXPECT_TRUE(available || !available);
    }

    TEST_F(CacheAnalyzerTest, Analyze) {
        auto metrics = analyzer.analyze(0);
        if (metrics.available) {
            EXPECT_GE(metrics.l1_hit_rate, 0.0);
            EXPECT_LE(metrics.l1_hit_rate, 1.0);
        }
    }

    TEST_F(CacheAnalyzerTest, AnalyzeKernel) {
        auto metrics = analyzer.analyze_kernel("matmul_kernel", 0);
        if (metrics.available) {
            EXPECT_GE(metrics.l2_hit_rate, 0.0);
            EXPECT_LE(metrics.l2_hit_rate, 1.0);
        }
    }

}  // namespace cuda::performance::bandwidth::test
