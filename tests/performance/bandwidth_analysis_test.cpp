#include <gtest/gtest.h>
#include <cuda_runtime.h>
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

    // `EXPECT_GE(points.size(), 0u)` was tautological for a size_t, and the input made it worse:
    // 1024 flops over 64 bytes is an intensity of 16, which is ABOVE the ridge point (9.56 on the
    // default peaks), so this kernel is compute-bound. get_memory_bound_points() therefore
    // returned zero points and the case named "MemoryBoundClassification" verified nothing about
    // memory-bound classification at all.
    TEST_F(RooflineModelTest, MemoryBoundClassification) {
        // Intensity 64/1024 = 0.0625, comfortably below the ridge point.
        model.add_point("kernel1", 64, 1024, 1000);
        auto memory_bound = model.get_memory_bound_points();
        auto compute_bound = model.get_compute_bound_points();
        EXPECT_EQ(memory_bound.size(), 1u) << "a kernel with intensity below the ridge point is memory-bound";
        EXPECT_EQ(compute_bound.size(), 0u) << "and must not also be reported as compute-bound";
    }

    TEST_F(RooflineModelTest, ComputeBoundClassification) {
        model.add_point("kernel1", 1024000, 64, 1000);
        auto compute_bound = model.get_compute_bound_points();
        auto memory_bound = model.get_memory_bound_points();
        EXPECT_EQ(compute_bound.size(), 1u) << "a kernel far above the ridge point is compute-bound";
        EXPECT_EQ(memory_bound.size(), 0u);
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

    // Each utilization accessor must divide its OWN direction's mean by that link's peak. Before
    // the fix all three divided the mean of every sample by their own peak, so mixing directions
    // produced ratios that describe nothing. Measured: one H2D sample at 10 GB/s and one D2D sample
    // at 450 GB/s, against peaks of 20/20/900, gave h2d_utilization_percent() = 1150% (the honest
    // answer is 50%) and d2d_utilization_percent() = 25.56% (also 50%).
    TEST_F(BandwidthUtilizationTest, DirectionalUtilizationsUseOnlyTheirOwnSamples) {
        tracker.set_peak_bandwidth(20.0, 20.0, 900.0);
        tracker.add_h2d_sample(10.0, 1000, 100);   // 50% of the 20 GB/s host link
        tracker.add_d2d_sample(450.0, 1000, 100);  // 50% of the 900 GB/s device link

        EXPECT_DOUBLE_EQ(tracker.h2d_utilization_percent(), 50.0);
        EXPECT_DOUBLE_EQ(tracker.d2d_utilization_percent(), 50.0);
        // No D2H sample was recorded, so there is nothing to report for that link.
        EXPECT_DOUBLE_EQ(tracker.d2h_utilization_percent(), 0.0);
    }

    // A utilization cannot exceed 100%. With the default placeholder peaks a single sample above
    // them used to produce one anyway, which is the visible symptom of dividing by the wrong peak.
    TEST_F(BandwidthUtilizationTest, UtilizationStaysAtOrBelowOneHundredPercentForAPlausibleSample) {
        tracker.set_peak_bandwidth(20.0, 900.0, 1555.0);
        tracker.add_h2d_sample(12.0, 1000, 100);    // 60% of 20
        tracker.add_d2d_sample(1200.0, 1000, 100);  // ~77% of 1555

        const double h2d = tracker.h2d_utilization_percent();
        const double d2d = tracker.d2d_utilization_percent();
        EXPECT_GE(h2d, 0.0);
        EXPECT_LE(h2d, 100.0) << "an interconnect cannot exceed its own peak";
        EXPECT_GE(d2d, 0.0);
        EXPECT_LE(d2d, 100.0) << "HBM cannot exceed its own peak";
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

    // `bool available = analyzer.is_available(); EXPECT_TRUE(available || !available);` asserted
    // nothing -- both halves of the disjunction are the same fact. Two entry points answer the
    // same question here, so the useful check is that they agree.
    TEST_F(CacheAnalyzerTest, CheckAvailability) {
        EXPECT_EQ(analyzer.is_available(), CacheAnalyzer::check_cupti_availability()) << "the cached instance flag must match the static probe";

        const auto metrics = analyzer.analyze(0);
        EXPECT_EQ(metrics.available, analyzer.is_available()) << "analyze() must not claim counters the analyzer knows it cannot read";
    }

    // This case only asserted anything when counters happened to be available, so on a build
    // without CUPTI it verified nothing at all. Both branches now have a requirement: real
    // numbers when available, and an explanation when not.
    TEST_F(CacheAnalyzerTest, Analyze) {
        auto metrics = analyzer.analyze(0);
        if (metrics.available) {
            EXPECT_GE(metrics.l1_hit_rate, 0.0);
            EXPECT_LE(metrics.l1_hit_rate, 1.0);
            EXPECT_GE(metrics.l2_hit_rate, 0.0);
            EXPECT_LE(metrics.l2_hit_rate, 1.0);
        } else {
            EXPECT_FALSE(metrics.error_message.empty()) << "an unavailable analysis must say why, not return silent zeroes";
            // Unavailable means no measurement was taken; a hit rate would be invented.
            EXPECT_DOUBLE_EQ(metrics.l1_hit_rate, 0.0) << "no counters were read, so no hit rate may be reported";
            EXPECT_DOUBLE_EQ(metrics.l2_hit_rate, 0.0);
        }
    }

    TEST_F(CacheAnalyzerTest, AnalyzeKernel) {
        auto metrics = analyzer.analyze_kernel("matmul_kernel", 0);
        if (metrics.available) {
            EXPECT_GE(metrics.l2_hit_rate, 0.0);
            EXPECT_LE(metrics.l2_hit_rate, 1.0);
        }
    }

    // Regression: BandwidthAnalysis's peak setters stored into members that nothing read.
    //
    // set_roofline_peak_flops() wrote peak_fp64_/peak_fp32_/peak_fp16_ and
    // set_roofline_peak_bandwidth() wrote peak_bandwidth_, but get_report() and
    // generate_roofline_json() both read roofline_->peaks(), which those members were
    // never copied into. A caller that configured device peaks before asking for a report
    // got the RooflineModel's own defaults and no indication the call did nothing.
    TEST(BandwidthAnalysisTest, ConfiguredPeaksReachTheReport) {
        BandwidthAnalysis analysis;
        analysis.set_roofline_peak_flops(/*fp64=*/1000.0, /*fp32=*/2000.0, /*fp16=*/4000.0);
        analysis.set_roofline_peak_bandwidth(/*hbm_gbs=*/3500.0);

        const std::string report = analysis.get_report();
        EXPECT_NE(report.find("1000.0"), std::string::npos) << "fp64 peak missing from report";
        EXPECT_NE(report.find("2000.0"), std::string::npos) << "fp32 peak missing from report";
        EXPECT_NE(report.find("3500.0"), std::string::npos) << "bandwidth peak missing from report";
    }

    // Regression: DevicePeaks::query() reported an FP64 peak eight times too low for
    // Ampere.
    //
    // The table it used was "8 FP64 cores per SM for major >= 7, 32 for major >= 9", with
    // one flop per core per clock. An A100 (sm_80, 108 SMs, 1.41 GHz) has 32 FP64 cores per
    // SM and two flops per FMA, so the real peak is
    //
    //     108 * 32 * 2 * 1.41e9 = 9.75 TFLOPS   (NVIDIA's published FP64 figure is 9.7)
    //
    // and query() produced 108 * 8 * 1.41e9 = 1.22 TFLOPS. Because the roofline's ridge
    // point and its memory-bound/compute-bound split are derived from this number, every
    // classification was made against a device roughly 8x weaker than the real one.
    //
    // The old assertion was EXPECT_GE(peaks.fp64_gflops, 0.0), which any value satisfies.
    TEST(BandwidthAnalysisTest, Fp64PeakMatchesTheDeviceItQueried) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "no CUDA device";
        }

        cudaDeviceProp prop {};
        ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);

        const auto peaks = DevicePeaks::query(0);

        // One FP64 FMA per core per clock is two floating-point operations. The per-SM core
        // count is 32 for every datacenter part from Volta through Hopper and 2 for the
        // consumer dies; this device is a datacenter A100, so use 32.
        const double expected = static_cast<double>(prop.multiProcessorCount) * 32.0 * 2.0 * static_cast<double>(prop.clockRate) * 1e3 / 1e9;

        // Within 20%: the coefficient is an architectural constant, and clockRate is a boost
        // figure, so exact equality is not the claim. Being 8x off is.
        EXPECT_NEAR(peaks.fp64_gflops, expected, 0.2 * expected) << "reported " << peaks.fp64_gflops << " GFLOPS, expected about " << expected << " for " << prop.name;

        // The FP32 rate is twice FP64 on every part this code supports.
        EXPECT_NEAR(peaks.fp32_gflops, peaks.fp64_gflops * 2.0, 1e-6 * peaks.fp32_gflops);
    }

    // Regression: the peaks were FLOP/s stored in fields named *_gflops.
    //
    //     peaks.fp64_gflops = SMs * cores * 2.0 * prop.clockRate * 1e3;
    //
    // clockRate is in kHz, so clockRate * 1e3 is Hz, and the expression is FLOP per second.
    // Assigning it to a field the whole class treats as GFLOPS made every peak 1e9 times
    // too large, and everything derived from it too:
    //
    //     ridge_point() == fp64_gflops / hbm_bandwidth_gbs
    //
    // came out around 1e12 FLOP/Byte. Real arithmetic intensities are 0.1 to 100, so no
    // point could ever reach the ridge and get_compute_bound_points() returned empty for
    // every workload -- the model classified everything as memory-bound.
    //
    // This checks the consequence, which is what a user of the class sees, rather than the
    // unit of one field.
    TEST(BandwidthAnalysisTest, AComputeHeavyKernelIsClassifiedAsComputeBound) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "no CUDA device";
        }

        RooflineModel model;

        // Two points with unmistakable arithmetic intensities: a classic memory-bound stream
        // (1 FLOP/Byte) and a compute-heavy GEMM-like kernel (64 FLOP/Byte).
        model.add_point("stream", /*flops=*/1000, /*bytes=*/1000, /*elapsed_ns=*/1000);
        model.add_point("gemm", /*flops=*/6400, /*bytes=*/100, /*elapsed_ns=*/1000);

        const auto memory_bound = model.get_memory_bound_points();
        const auto compute_bound = model.get_compute_bound_points();

        EXPECT_EQ(memory_bound.size() + compute_bound.size(), 2u) << "every point must land on one side of the ridge";
        EXPECT_EQ(compute_bound.size(), 1u) << "a 64 FLOP/Byte kernel must be compute-bound on any real device; ridge = " << model.ridge_point();
        ASSERT_EQ(compute_bound.size(), 1u);
        EXPECT_EQ(compute_bound.front().kernel_name, "gemm");

        // The ridge point is FLOP/Byte, so it has to be a number a kernel could reach: on
        // every GPU built it is between roughly 1 and 1000.
        EXPECT_GT(model.ridge_point(), 0.1) << "ridge point is not a plausible arithmetic intensity";
        EXPECT_LT(model.ridge_point(), 1e6) << "ridge point is orders of magnitude too large";
    }

}  // namespace cuda::performance::bandwidth::test
