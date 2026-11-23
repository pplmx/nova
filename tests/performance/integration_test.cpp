#include <gtest/gtest.h>
#include <cuda/performance/nvblox_metrics.h>
#include <cuda/performance/kernel_profiler.h>
#include <cuda/performance/metric_aggregators.h>
#include <cuda/performance/fusion/kernel_fusion_analyzer.h>
#include <cuda/performance/fusion/fusion_profitability.h>
#include <cuda/performance/fusion/fusion_patterns.h>
#include <cuda/performance/bandwidth/roofline_model.h>
#include <cuda/performance/bandwidth/cache_analyzer.h>
#include <cuda/performance/dashboard/dashboard_exporter.h>
#include <cuda/performance/dashboard/flame_graph.h>

namespace cuda::performance::test {

    class PerformanceToolingIntegrationTest : public ::testing::Test {
    protected:
        void SetUp() override {
            NVBloxMetricsCollector::instance().reset();
            KernelProfiler::instance().reset();
        }

        void TearDown() override {
            NVBloxMetricsCollector::instance().reset();
            KernelProfiler::instance().reset();
        }
    };

    TEST_F(PerformanceToolingIntegrationTest, MetricsCollectionPipeline) {
        auto& collector = NVBloxMetricsCollector::instance();

        collector.register_metric("latency", MetricType::Latency);
        collector.add_sample("latency", 1.0);

        KernelMetrics km;
        km.name = "matmul";
        km.latency_ns = 1000;
        km.throughput_gflops = 100.0;
        collector.record_kernel(km);

        EXPECT_EQ(collector.total_kernel_count(), 1u);
        EXPECT_EQ(collector.registered_metric_count(), 1u);
    }

    TEST_F(PerformanceToolingIntegrationTest, KernelProfilerPipeline) {
        auto& profiler = KernelProfiler::instance();
        profiler.enable();

        EXPECT_TRUE(profiler.is_enabled());

        profiler.disable();
        EXPECT_FALSE(profiler.is_enabled());
    }

    TEST_F(PerformanceToolingIntegrationTest, FusionAnalysisPipeline) {
        fusion::KernelFusionAnalyzer analyzer;
        fusion::Operation op1 {"matmul", fusion::OpType::Matmul, 100, 1024, 64, 1, 1, 1, 128, 1, 1, 0.75f, false};
        fusion::Operation op2 {"bias", fusion::OpType::ElementWise, 50, 64, 64, 1, 1, 1, 128, 1, 1, 0.8f, false};

        analyzer.add_operation(op1);
        analyzer.add_operation(op2);

        auto opportunities = analyzer.detect_opportunities();
        EXPECT_GE(opportunities.size(), 1u);

        fusion::FusionProfitabilityConfig config;
        config.launch_overhead_us = 100.0;
        fusion::FusionRecommendationEngine engine(config);

        auto recommendations = engine.generate_recommendations(opportunities);
    }

    TEST_F(PerformanceToolingIntegrationTest, BandwidthAnalysisPipeline) {
        bandwidth::RooflineModel roofline;
        roofline.add_point("kernel1", 1024, 64, 1000);

        EXPECT_EQ(roofline.point_count(), 1u);
        EXPECT_GE(roofline.ridge_point(), 0.0);

        bandwidth::BandwidthUtilizationTracker tracker;
        tracker.set_peak_bandwidth(20.0, 20.0, 900.0);
        tracker.add_sample(500.0, 1e9, 1e9);

        EXPECT_GE(tracker.utilization_percent(), 0.0);
    }

    TEST_F(PerformanceToolingIntegrationTest, CacheAnalysisPipeline) {
        bandwidth::CacheAnalyzer analyzer;
        auto metrics = analyzer.analyze(0);

        // The original assertion was `EXPECT_TRUE(metrics.available || !metrics.available)`,
        // which is true regardless. What this pipeline can actually promise is that its two
        // reporting paths agree and that an unavailable analysis explains itself.
        EXPECT_EQ(analyzer.is_available(), metrics.available) << "is_available() and analyze().available must report the same thing";
        if (!metrics.available) {
            EXPECT_FALSE(metrics.error_message.empty()) << "an unavailable analyzer must say why";
        } else {
            EXPECT_GE(metrics.l1_hit_rate, 0.0);
            EXPECT_LE(metrics.l1_hit_rate, 1.0);
        }
    }

    TEST_F(PerformanceToolingIntegrationTest, DashboardExportPipeline) {
        dashboard::DashboardExporter exporter;
        EXPECT_TRUE(exporter.is_empty());

        bandwidth::RooflineModel roofline;
        roofline.add_point("kernel1", 1024, 64, 1000);
        exporter.add_roofline_data(roofline);

        exporter.add_kernel_count(10);

        EXPECT_FALSE(exporter.is_empty());

        std::string json = exporter.to_json();
        EXPECT_NE(json.find("header"), std::string::npos);
        EXPECT_NE(json.find("roofline"), std::string::npos);
    }

    TEST_F(PerformanceToolingIntegrationTest, FlameGraphPipeline) {
        dashboard::FlameGraphGenerator generator;
        dashboard::ChromeTraceEvent event;
        event.name = "kernel1";
        event.category = "performance";
        event.ph = "X";
        event.ts = 1000;
        event.dur = 100;
        event.pid = 1;
        event.tid = 1;

        generator.add_event(event);
        EXPECT_EQ(generator.event_count(), 1u);

        auto flame_graph = generator.build_flame_graph();
        EXPECT_EQ(flame_graph.name, "root");
        EXPECT_EQ(flame_graph.value, 100u);

        std::string json = generator.to_json();
        EXPECT_NE(json.find("root"), std::string::npos);
    }

    TEST_F(PerformanceToolingIntegrationTest, FullPipeline) {
        auto& collector = NVBloxMetricsCollector::instance();

        collector.register_metric("test_metric", MetricType::Throughput);
        collector.add_sample("test_metric", 100.0);

        KernelMetrics km;
        km.name = "matmul_kernel";
        km.latency_ns = 1000;
        km.throughput_gflops = 500.0;
        collector.record_kernel(km);

        fusion::KernelFusionAnalyzer fusion_analyzer;
        fusion::Operation matmul {"matmul_kernel", fusion::OpType::Matmul, 1000, 1024, 64, 1, 1, 1, 128, 1, 1, 0.75f, false};
        fusion::Operation bias {"bias_add", fusion::OpType::ElementWise, 100, 64, 64, 1, 1, 1, 128, 1, 1, 0.8f, false};
        fusion_analyzer.add_operation(matmul);
        fusion_analyzer.add_operation(bias);

        auto opportunities = fusion_analyzer.detect_opportunities();

        bandwidth::RooflineModel roofline;
        roofline.add_point("matmul_kernel", 1024, 64, 1000);

        dashboard::DashboardExporter exporter;
        exporter.add_roofline_data(roofline);
        exporter.add_kernel_count(collector.total_kernel_count());

        std::string json = exporter.to_json();
        EXPECT_FALSE(json.empty());

        dashboard::FlameGraphGenerator flame_gen;
        dashboard::ChromeTraceEvent event {"matmul_kernel", "algo", "X", 1000, 1000, 1, 1};
        flame_gen.add_event(event);

        std::string flame_json = flame_gen.to_json();
        EXPECT_FALSE(flame_json.empty());
    }

    TEST_F(PerformanceToolingIntegrationTest, FusionPatternsIntegration) {
        auto patterns = fusion::FusionPatterns::all();
        EXPECT_EQ(patterns.size(), 10u);

        auto* matmul_pattern = fusion::FusionPatterns::find_by_name("matmul_bias_act_relu");
        EXPECT_NE(matmul_pattern, nullptr);

        auto matmul_patterns = fusion::FusionPatterns::find_by_op_type(fusion::OpType::Matmul);
        EXPECT_GE(matmul_patterns.size(), 1u);
    }

    TEST_F(PerformanceToolingIntegrationTest, DevicePeaksIntegration) {
        auto peaks = bandwidth::DevicePeaks::query(0);
        EXPECT_GE(peaks.fp64_gflops, 0.0);
        EXPECT_GE(peaks.fp32_gflops, 0.0);
        EXPECT_GE(peaks.hbm_bandwidth_gbs, 0.0);
    }

    TEST_F(PerformanceToolingIntegrationTest, DashboardGeneratorIntegration) {
        dashboard::DashboardGenerator generator;

        dashboard::DashboardExporter exporter1;
        bandwidth::RooflineModel roofline;
        roofline.add_point("kernel1", 1024, 64, 1000);
        exporter1.add_roofline_data(roofline);

        dashboard::DashboardExporter exporter2;
        fusion::KernelFusionAnalyzer analyzer;
        exporter2.add_fusion_data({});

        generator.add_exporter(exporter1);
        generator.add_exporter(exporter2);

        std::string json = generator.generate_json();
        EXPECT_NE(json.find("dashboards"), std::string::npos);
    }

}  // namespace cuda::performance::test
