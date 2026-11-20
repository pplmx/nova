#include <gtest/gtest.h>

#include "test_temp_paths.h"
#include <fstream>
#include <cuda_runtime.h>
#include <cuda/tools/bank_conflict_analyzer.h>
#include <cuda/tools/timeline_visualizer.h>

namespace cuda::tools::test {

    class ToolsTest : public ::testing::Test {
    protected:
        void SetUp() override {}
        void TearDown() override {
            TimelineVisualizer::instance().clear();
            BandwidthAnalyzer::instance().clear();
        }
    };

    TEST_F(ToolsTest, SharedMemoryAnalyzerConstruction) {
        SharedMemoryAnalyzer& analyzer = SharedMemoryAnalyzer::instance();

        BankConflictConfig config;
        config.block_size = 256;
        config.check_padding = true;
        analyzer.set_config(config);

        auto retrieved = analyzer.get_config();
        EXPECT_EQ(retrieved.block_size, 256);
        EXPECT_TRUE(retrieved.check_padding);
    }

    TEST_F(ToolsTest, SharedMemoryAnalyzerPaddingHints) {
        SharedMemoryAnalyzer& analyzer = SharedMemoryAnalyzer::instance();

        analyzer.enable_padding_hints(true);
        EXPECT_TRUE(analyzer.is_padding_hints_enabled());

        analyzer.enable_padding_hints(false);
        EXPECT_FALSE(analyzer.is_padding_hints_enabled());
    }

    TEST_F(ToolsTest, SharedMemoryAnalyzerSuggestPadding) {
        SharedMemoryAnalyzer& analyzer = SharedMemoryAnalyzer::instance();

        int padding = analyzer.suggest_padding(4, 1024);
        EXPECT_GE(padding, 0);
    }

    TEST_F(ToolsTest, BankConflictDetection) {
        int conflicts = detect_bank_conflicts(nullptr, 64, 32);
        EXPECT_GE(conflicts, 0);
    }

    TEST_F(ToolsTest, BankConflictAnalysis) {
        BankConflictConfig config;
        config.block_size = 256;
        config.num_threads = 64;

        auto result = analyze_bank_conflicts(nullptr, 4096, config);
        EXPECT_GE(result.potential_conflicts, 0);
        EXPECT_GE(result.suggested_padding, 0);
        EXPECT_FALSE(result.analysis.empty());
    }

    TEST_F(ToolsTest, TimelineVisualizerConstruction) {
        TimelineVisualizer& visualizer = TimelineVisualizer::instance();
        visualizer.clear();
        EXPECT_EQ(visualizer.get_event_count(), 0);
    }

    TEST_F(ToolsTest, TimelineVisualizerRecordKernel) {
        TimelineVisualizer& visualizer = TimelineVisualizer::instance();
        visualizer.clear();

        visualizer.record_kernel("test_kernel", 1.5f);
        EXPECT_EQ(visualizer.get_event_count(), 1);
    }

    TEST_F(ToolsTest, TimelineVisualizerRecordMemoryOp) {
        TimelineVisualizer& visualizer = TimelineVisualizer::instance();
        visualizer.clear();

        visualizer.record_memory_op("memcpy", 1024, 0.5f);
        EXPECT_EQ(visualizer.get_event_count(), 1);
    }

    TEST_F(ToolsTest, TimelineVisualizerExport) {
        TimelineVisualizer& visualizer = TimelineVisualizer::instance();
        visualizer.clear();

        visualizer.record_kernel("kernel1", 1.0f);
        visualizer.record_memory_op("memcpy", 1024, 0.5f);

        // The two exports returned void and nothing was checked, so a writer that produced an
        // empty file — or failed to open it — passed. Both paths are now inspected: the files
        // must exist and contain the recorded names.
        const std::string trace_path = nova::test::temp_path("nova_tools_trace_", ".json");
        const std::string json_path = nova::test::temp_path("nova_tools_timeline_", ".json");

        visualizer.export_chrome_trace(trace_path);
        visualizer.export_json(json_path);

        for (const std::string& path : {trace_path, json_path}) {
            std::ifstream file(path);
            ASSERT_TRUE(file.good()) << "the export did not create " << path;
            const std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            EXPECT_FALSE(contents.empty()) << path << " is empty";
            EXPECT_NE(contents.find("kernel1"), std::string::npos) << path << " does not mention the recorded kernel";
            file.close();
            std::remove(path.c_str());
        }
    }

    TEST_F(ToolsTest, TimelineVisualizerEnable) {
        TimelineVisualizer& visualizer = TimelineVisualizer::instance();
        visualizer.clear();

        visualizer.enable();
        visualizer.record_kernel("kernel1", 1.0f);
        EXPECT_EQ(visualizer.get_event_count(), 1);
    }

    TEST_F(ToolsTest, BandwidthAnalyzerConstruction) {
        BandwidthAnalyzer& analyzer = BandwidthAnalyzer::instance();
        analyzer.clear();

        auto stats = analyzer.get_stats();
        EXPECT_TRUE(stats.empty());
    }

    TEST_F(ToolsTest, BandwidthAnalyzerRecordOperation) {
        BandwidthAnalyzer& analyzer = BandwidthAnalyzer::instance();
        analyzer.clear();

        analyzer.record_operation("H2D", 1024 * 1024, 1.0f);
        analyzer.record_operation("D2H", 512 * 1024, 0.5f);

        auto stats = analyzer.get_stats();
        EXPECT_EQ(stats.size(), 2);
    }

    TEST_F(ToolsTest, BandwidthAnalyzerUtilization) {
        BandwidthAnalyzer& analyzer = BandwidthAnalyzer::instance();
        analyzer.clear();

        analyzer.record_operation("test", 1024, 1.0f);

        float utilization = analyzer.get_utilization_percentage(0);
        EXPECT_GE(utilization, 0.0f);
    }

    TEST_F(ToolsTest, BandwidthAnalyzerReport) {
        BandwidthAnalyzer& analyzer = BandwidthAnalyzer::instance();
        analyzer.clear();

        // 4096 bytes in 2 ms is 2.048 MB/s, i.e. 0.002048 GB/s.
        analyzer.record_operation("test_op", 4096, 2.0f);

        // This case used to stop after writing a file, so it passed whether the report
        // contained the operation, was empty, or failed to write at all. Asserting on the
        // recorded statistics is what makes it a test.
        const auto stats = analyzer.get_stats();
        ASSERT_FALSE(stats.empty()) << "the recorded operation produced no statistics";
        ASSERT_EQ(stats.front().name, "test_op");
        EXPECT_EQ(stats.front().total_bytes, 4096u);
        EXPECT_FLOAT_EQ(stats.front().total_duration_ms, 2.0f);

        // bytes / (ms * 1e6) = GB/s
        const float expected_gbps = 4096.0f / (2.0f * 1e6f);
        EXPECT_NEAR(stats.front().bandwidth_gbps, expected_gbps, 1e-6f) << "bandwidth must be bytes over seconds in GB/s";

        analyzer.export_report("/tmp/bandwidth_report.md");
    }

    TEST_F(ToolsTest, TraceEventsGeneration) {
        TimelineVisualizer& visualizer = TimelineVisualizer::instance();
        visualizer.clear();

        visualizer.record_kernel("kernel1", 1.0f);
        visualizer.record_kernel("kernel2", 2.0f);

        auto traces = visualizer.get_trace_events();
        EXPECT_EQ(traces.size(), 2);
    }

    // The existing hint case only checked that enable_padding_hints() and
    // is_padding_hints_enabled() agree with each other, which they always did: the flag was
    // stored and read by its own accessor and by nothing else. suggest_padding() is the
    // function that produces a padding hint, and it never consulted the switch.
    TEST_F(ToolsTest, DisablingPaddingHintsStopsSuggestPaddingFromSuggesting) {
        SharedMemoryAnalyzer& analyzer = SharedMemoryAnalyzer::instance();

        analyzer.enable_padding_hints(true);
        const int with_hints = analyzer.suggest_padding(4, 1024);
        EXPECT_GT(with_hints, 0) << "a layout that conflicts should get a non-zero suggestion";

        analyzer.enable_padding_hints(false);
        EXPECT_EQ(analyzer.suggest_padding(4, 1024), 0) << "hints are disabled, so no padding should be suggested";

        // Restore, since this is a singleton and other cases use it.
        analyzer.enable_padding_hints(true);
    }

}  // namespace cuda::tools::test
