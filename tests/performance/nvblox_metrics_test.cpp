#include <gtest/gtest.h>
#include <algorithm>

#include <cuda/performance/nvblox_metrics.h>
#include <cuda/performance/kernel_profiler.h>
#include <cuda/performance/metric_aggregators.h>

#include <thread>
#include <vector>

namespace cuda::performance::test {

    class NVBloxMetricsTest : public ::testing::Test {
    protected:
        void SetUp() override { NVBloxMetricsCollector::instance().reset(); }

        void TearDown() override { NVBloxMetricsCollector::instance().reset(); }
    };

    TEST_F(NVBloxMetricsTest, KernelMetricsConstruction) {
        KernelMetrics km;
        EXPECT_EQ(km.name, "");
        EXPECT_EQ(km.latency_ns, 0u);
        EXPECT_EQ(km.throughput_gflops, 0.0);
        EXPECT_EQ(km.sm_occupancy, 0.0f);
        EXPECT_EQ(km.arithmetic_intensity, 0.0);
        EXPECT_EQ(km.memory_bandwidth_gbs, 0.0);
    }

    TEST_F(NVBloxMetricsTest, KernelMetricsWithValues) {
        KernelMetrics km;
        km.name = "test_kernel";
        km.latency_ns = 1000;
        km.throughput_gflops = 100.5;
        km.sm_occupancy = 0.75f;
        km.arithmetic_intensity = 32.0;
        km.memory_bandwidth_gbs = 500.0;

        EXPECT_EQ(km.name, "test_kernel");
        EXPECT_EQ(km.latency_ns, 1000u);
        EXPECT_DOUBLE_EQ(km.throughput_gflops, 100.5);
        EXPECT_FLOAT_EQ(km.sm_occupancy, 0.75f);
        EXPECT_DOUBLE_EQ(km.arithmetic_intensity, 32.0);
        EXPECT_DOUBLE_EQ(km.memory_bandwidth_gbs, 500.0);
    }

    TEST_F(NVBloxMetricsTest, RegisterMetric) {
        auto& collector = NVBloxMetricsCollector::instance();

        collector.register_metric("latency", MetricType::Latency);
        collector.register_metric("throughput", MetricType::Throughput);

        EXPECT_EQ(collector.registered_metric_count(), 2u);
    }

    TEST_F(NVBloxMetricsTest, AddSample) {
        auto& collector = NVBloxMetricsCollector::instance();

        collector.register_metric("test_metric", MetricType::Latency);
        collector.add_sample("test_metric", 1.5);
        collector.add_sample("test_metric", 2.5);

        auto samples = collector.get_metric_samples("test_metric");
        EXPECT_EQ(samples.size(), 2u);
        EXPECT_DOUBLE_EQ(samples[0].value, 1.5);
        EXPECT_DOUBLE_EQ(samples[1].value, 2.5);
    }

    TEST_F(NVBloxMetricsTest, RecordKernel) {
        auto& collector = NVBloxMetricsCollector::instance();

        KernelMetrics km;
        km.name = "kernel1";
        km.latency_ns = 100;
        collector.record_kernel(km);

        km.name = "kernel2";
        km.latency_ns = 200;
        collector.record_kernel(km);

        auto metrics = collector.get_kernel_metrics();
        EXPECT_EQ(metrics.size(), 2u);
        EXPECT_EQ(metrics[0].name, "kernel1");
        EXPECT_EQ(metrics[1].name, "kernel2");
    }

    TEST_F(NVBloxMetricsTest, ConcurrentMetricCollection) {
        auto& collector = NVBloxMetricsCollector::instance();

        std::vector<std::thread> threads;
        constexpr int num_threads = 4;
        constexpr int samples_per_thread = 100;

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&collector, t]() {
                for (int i = 0; i < samples_per_thread; ++i) {
                    collector.add_sample("metric_" + std::to_string(t), static_cast<double>(i));
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        EXPECT_EQ(collector.total_kernel_count(), 0u);
        // There is no useful assertion on registered_metric_count() here. The previous line said
        // `EXPECT_GE(..., 0u)`, which is always true for a size_t, and the obvious tightening --
        // requiring it to be non-zero -- is also wrong: add_sample() records samples, not metrics,
        // so the count is legitimately 0 after this case. Asserting either would be noise, so the
        // line is removed rather than made to look meaningful. What this case actually checks is
        // that four threads hammering add_sample() concurrently leave total_kernel_count() at zero,
        // which the line above does.
    }

    TEST_F(NVBloxMetricsTest, ToJson) {
        auto& collector = NVBloxMetricsCollector::instance();

        collector.add_sample("latency", 1.0);
        collector.add_sample("latency", 2.0);

        KernelMetrics km;
        km.name = "test_kernel";
        km.latency_ns = 100;
        km.throughput_gflops = 50.0;
        collector.record_kernel(km);

        std::string json = collector.to_json();
        EXPECT_NE(json.find("test_kernel"), std::string::npos);
        EXPECT_NE(json.find("latency"), std::string::npos);
    }

    TEST_F(NVBloxMetricsTest, ToCsv) {
        auto& collector = NVBloxMetricsCollector::instance();

        KernelMetrics km;
        km.name = "kernel1";
        km.latency_ns = 100;
        km.throughput_gflops = 50.0;
        collector.record_kernel(km);

        std::string csv = collector.to_csv();
        EXPECT_NE(csv.find("kernel_name"), std::string::npos);
        EXPECT_NE(csv.find("kernel1"), std::string::npos);
    }

    TEST_F(NVBloxMetricsTest, Reset) {
        auto& collector = NVBloxMetricsCollector::instance();

        collector.add_sample("test", 1.0);
        KernelMetrics km;
        km.name = "test";
        collector.record_kernel(km);

        EXPECT_EQ(collector.total_kernel_count(), 1u);

        collector.reset();

        EXPECT_EQ(collector.total_kernel_count(), 0u);
        EXPECT_TRUE(collector.get_metric_samples("test").empty());
    }

    TEST_F(NVBloxMetricsTest, CalculateArithmeticIntensity) {
        EXPECT_DOUBLE_EQ(calculate_arithmetic_intensity(1024, 64), 16.0);
        EXPECT_DOUBLE_EQ(calculate_arithmetic_intensity(512, 128), 4.0);
        EXPECT_DOUBLE_EQ(calculate_arithmetic_intensity(1024, 0), 0.0);
    }

    TEST_F(NVBloxMetricsTest, CalculateThroughput) {
        double gflops = calculate_throughput_gflops(1e9, 1e9);
        EXPECT_DOUBLE_EQ(gflops, 1.0);
    }

    TEST_F(NVBloxMetricsTest, CalculateMemoryBandwidth) {
        double gbs = calculate_memory_bandwidth_gbs(1e9, 1e9);
        EXPECT_DOUBLE_EQ(gbs, 1.0);
    }

    class KernelProfilerTest : public ::testing::Test {
    protected:
        void SetUp() override {
            KernelProfiler::instance().reset();
            KernelProfiler::instance().enable();
        }

        void TearDown() override { KernelProfiler::instance().reset(); }
    };

    TEST_F(KernelProfilerTest, EnableDisable) {
        auto& profiler = KernelProfiler::instance();

        EXPECT_TRUE(profiler.is_enabled());

        profiler.disable();
        EXPECT_FALSE(profiler.is_enabled());

        profiler.enable();
        EXPECT_TRUE(profiler.is_enabled());
    }

    // Regression: KernelProfiler::estimate_occupancy() computed
    //
    //     blocks_needed = ceil(threads_per_block / maxThreadsPerBlock)   // always 1
    //     active_blocks = min(max_blocks_per_sm, blocks_needed)          // always 1
    //
    // so the caller's max_blocks_per_sm argument had no effect at all, and the answer was the
    // occupancy of exactly one block per SM. Measured on this device (2048 threads per SM):
    // estimate_occupancy(1024, 1) and estimate_occupancy(1024, 64) both returned 0.5, where
    // two 1024-thread blocks fit and the answer is 1.0.
    //
    // The case is written against the device's own limits so it holds on any GPU, and the
    // discriminating assertion is that the SAME block size gives DIFFERENT answers for
    // different max_blocks_per_sm -- which the old implementation could never produce.
    TEST_F(KernelProfilerTest, OccupancyRespondsToTheBlocksPerSmArgument) {
        auto& profiler = KernelProfiler::instance();

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "no device to query";
        }
        const int block_size = std::min(prop.maxThreadsPerBlock, 256);
        const int blocks_that_fit = prop.maxThreadsPerMultiProcessor / block_size;
        if (blocks_that_fit < 2) {
            GTEST_SKIP() << "device holds fewer than two blocks of this size; the argument cannot be distinguished";
        }

        const float one_block = profiler.estimate_occupancy(block_size, 1);
        const float many_blocks = profiler.estimate_occupancy(block_size, blocks_that_fit);

        EXPECT_GT(one_block, 0.0f);
        EXPECT_LT(one_block, many_blocks) << "asking for more resident blocks must raise occupancy; it was ignored before";
        EXPECT_FLOAT_EQ(many_blocks, 1.0f) << "a block size that tiles the SM exactly reaches full occupancy";

        // Occupancy is a ratio and cannot exceed it.
        EXPECT_LE(profiler.estimate_occupancy(block_size, blocks_that_fit * 8), 1.0f) << "more blocks than fit cannot exceed 100%";

        // A block larger than the device allows cannot be launched; reporting a clamped
        // occupancy for it would describe a kernel that never runs.
        EXPECT_FLOAT_EQ(profiler.estimate_occupancy(prop.maxThreadsPerBlock * 2, 1), 0.0f);
        EXPECT_FLOAT_EQ(profiler.estimate_occupancy(0, 1), 0.0f);
        EXPECT_FLOAT_EQ(profiler.estimate_occupancy(block_size, 0), 0.0f);
    }

    TEST_F(KernelProfilerTest, OccupancyCalculator) {
        float occ = OccupancyCalculator::calculate_theoretical_occupancy(256, 32, 0, 0);
        EXPECT_GE(occ, 0.0f);
        EXPECT_LE(occ, 1.0f);
    }

    TEST_F(KernelProfilerTest, RecommendedBlockSize) {
        int block_size = OccupancyCalculator::recommended_block_size(32, 0, 0);
        EXPECT_GT(block_size, 0);
        EXPECT_LE(block_size, 1024);
    }

    class MetricAggregatorTest : public ::testing::Test {
    protected:
        void SetUp() override {
            ai_agg_.reset();
            flops_agg_.reset();
            bw_agg_.reset();
        }

        ArithmeticIntensityAggregator ai_agg_;
        FLOPsAggregator flops_agg_;
        BandwidthAggregator bw_agg_;
    };

    TEST_F(MetricAggregatorTest, ArithmeticIntensity) {
        ai_agg_.add_sample(1024, 64);
        ai_agg_.add_sample(2048, 128);

        EXPECT_DOUBLE_EQ(ai_agg_.get_average(), 16.0);
        EXPECT_DOUBLE_EQ(ai_agg_.get_peak(), 16.0);
    }

    TEST_F(MetricAggregatorTest, ArithmeticIntensitySummary) {
        ai_agg_.add_sample(1024, 64);
        ai_agg_.add_sample(2048, 128);
        ai_agg_.add_sample(512, 32);

        auto summary = ai_agg_.get_summary();
        EXPECT_EQ(summary.name, "arithmetic_intensity");
        EXPECT_EQ(summary.sample_count, 3u);
        EXPECT_DOUBLE_EQ(summary.mean, 16.0);
    }

    TEST_F(MetricAggregatorTest, FLOPsAggregator) {
        flops_agg_.add_sample(1e9, 1e9);
        flops_agg_.add_sample(2e9, 1e9);

        EXPECT_DOUBLE_EQ(flops_agg_.get_achieved_gflops(), 1.5);
    }

    TEST_F(MetricAggregatorTest, FLOPsAggregatorEfficiency) {
        flops_agg_.set_device_peak_flops(1000.0, 2000.0, 4000.0);
        flops_agg_.add_sample(500e6, 1e9);

        double efficiency = flops_agg_.get_efficiency_percent();
        EXPECT_GT(efficiency, 0.0);
        EXPECT_LT(efficiency, 100.0);
    }

    TEST_F(MetricAggregatorTest, BandwidthAggregator) {
        bw_agg_.set_peak_bandwidths(20.0, 20.0, 900.0);
        bw_agg_.add_sample(1e9, 1e9, BandwidthAggregator::TransferType::HostToDevice);
        bw_agg_.add_sample(2e9, 1e9, BandwidthAggregator::TransferType::DeviceToHost);
        bw_agg_.add_sample(450e9, 1e9, BandwidthAggregator::TransferType::DeviceToDevice);

        EXPECT_DOUBLE_EQ(bw_agg_.get_h2d_bandwidth_gbs(), 1.0);
        EXPECT_DOUBLE_EQ(bw_agg_.get_d2h_bandwidth_gbs(), 2.0);
        EXPECT_NEAR(bw_agg_.get_d2d_bandwidth_gbs(), 450.0, 1.0);
    }

    TEST_F(MetricAggregatorTest, BandwidthUtilization) {
        bw_agg_.set_peak_bandwidths(20.0, 20.0, 900.0);
        bw_agg_.add_sample(10e9, 1e9, BandwidthAggregator::TransferType::HostToDevice);

        EXPECT_DOUBLE_EQ(bw_agg_.get_h2d_utilization_percent(), 50.0);
    }

    TEST_F(MetricAggregatorTest, BandwidthSummary) {
        bw_agg_.add_sample(1e9, 1e9, BandwidthAggregator::TransferType::HostToDevice);

        auto summary = bw_agg_.get_summary(BandwidthAggregator::TransferType::HostToDevice);
        EXPECT_EQ(summary.name, "h2d_bandwidth_gbs");
        EXPECT_EQ(summary.sample_count, 1u);
    }

    TEST_F(MetricAggregatorTest, ComputeMean) {
        std::vector<double> samples = {1.0, 2.0, 3.0, 4.0, 5.0};
        EXPECT_DOUBLE_EQ(compute_mean(samples), 3.0);
    }

    TEST_F(MetricAggregatorTest, ComputeStdDev) {
        std::vector<double> samples = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
        double stddev = compute_stddev(samples);
        EXPECT_NEAR(stddev, 2.138, 0.01);
    }

    TEST_F(MetricAggregatorTest, PipelineAggregation) {
        MetricAggregatorPipeline pipeline;
        pipeline.set_device_peak_specs(1000.0, 2000.0, 4000.0, 20.0, 20.0, 900.0);

        pipeline.add_arithmetic_sample(1024, 64);
        pipeline.add_flops_sample(1e9, 1e9);
        pipeline.add_bandwidth_sample(1e9, 1e9, BandwidthAggregator::TransferType::DeviceToDevice);

        auto summaries = pipeline.get_all_summaries();
        EXPECT_EQ(summaries.size(), 5u);
    }

    TEST_F(MetricAggregatorTest, PipelineReset) {
        MetricAggregatorPipeline pipeline;

        pipeline.add_arithmetic_sample(1024, 64);
        pipeline.add_flops_sample(1e9, 1e9);

        pipeline.reset();

        auto summaries = pipeline.get_all_summaries();
        for (const auto& s : summaries) {
            EXPECT_EQ(s.sample_count, 0u);
        }
    }

}  // namespace cuda::performance::test
