#include <gtest/gtest.h>
#include <cuda/performance/dashboard/dashboard_exporter.h>
#include <cuda/performance/dashboard/flame_graph.h>

namespace cuda::performance::dashboard::test {

    class DashboardExporterTest : public ::testing::Test {
    protected:
        DashboardExporter exporter;
    };

    class FlameGraphGeneratorTest : public ::testing::Test {};

    TEST_F(DashboardExporterTest, DefaultConfig) {
        auto config = exporter.get_config();
        EXPECT_TRUE(config.include_roofline);
        EXPECT_TRUE(config.include_fusion);
        EXPECT_TRUE(config.include_bandwidth);
        EXPECT_EQ(config.output_format, "json");
    }

    // add_kernel_count() wrote into the exporter's own DashboardData, and get_data() built its
    // result from a fresh one that never incorporated it. Measured before the fix:
    // add_kernel_count(12345) flipped is_empty() to false -- so the value was stored -- while
    // get_data().kernels.kernel_count stayed 0 and to_json() emitted "count": 0. A caller's kernel
    // count reached nothing, which is the kind of report that silently understates a run.
    TEST_F(DashboardExporterTest, KernelCountReachesTheExportedData) {
        exporter.add_kernel_count(12345);

        EXPECT_EQ(exporter.get_data().kernels.kernel_count, 12345u) << "the count must survive into get_data()";
        const std::string json = exporter.to_json();
        EXPECT_NE(json.find("12345"), std::string::npos) << "the count must appear in the JSON, got: " << json;
    }

    TEST_F(DashboardExporterTest, KernelCountDoesNotHideTheComputedSections) {
        // Starting get_data() from the stored accumulators must not stop the populate_* passes from
        // running: the roofline/bandwidth/fusion sections are computed from the registered models,
        // not stored in that struct.
        exporter.add_kernel_count(7);
        const auto data = exporter.get_data();

        EXPECT_EQ(data.kernels.kernel_count, 7u);
        EXPECT_FALSE(data.header.version.empty()) << "populate_header still has to run";
        EXPECT_FALSE(data.header.timestamp.empty());
    }

    // from_chrome_trace() is the declared inverse of to_chrome_trace(), and the three parsers that
    // exist alongside it accepted their argument and threw it away:
    //
    //     FlameGraphGenerator FlameGraphGenerator::from_chrome_trace(const std::string& trace_json) {
    //         FlameGraphGenerator generator;
    //         return generator;                 // trace_json never read
    //     }
    //
    // Measured before the fix: two events serialised by to_chrome_trace() (277 bytes) came back
    // through from_chrome_trace() as event_count() == 0 -- the same answer a legitimately empty
    // trace gives, so nothing could tell the difference.
    TEST_F(FlameGraphGeneratorTest, ChromeTraceRoundTripsThroughItsInverse) {
        FlameGraphGenerator source;
        ChromeTraceEvent e;
        e.name = "matmul_kernel";
        e.category = "gpu";
        e.ph = "X";
        e.ts = 1234;
        e.dur = 567;
        e.pid = 3;
        e.tid = 9;
        source.add_event(e);

        const std::string trace = source.to_chrome_trace();
        ASSERT_FALSE(trace.empty());

        const auto recovered = FlameGraphGenerator::from_chrome_trace(trace);
        EXPECT_EQ(recovered.event_count(), 1u) << "the inverse must recover the events it was given";

        // Values, not just the count: the duration is what build_flame_graph() sums.
        const auto node = recovered.build_flame_graph();
        EXPECT_EQ(node.value, 567u) << "the duration must survive the round trip";
    }

    TEST_F(FlameGraphGeneratorTest, ParserReadsJsonIntoEvents) {
        NVTraceParser parser;
        const std::string json = R"([
            {"name": "kernel_a", "cat": "gpu", "ph": "X", "ts": 10, "dur": 100, "pid": 1, "tid": 2},
            {"name": "kernel_b", "cat": "gpu", "ph": "X", "ts": 20, "dur": 200, "pid": 1, "tid": 2}
        ])";

        const auto events = parser.parse_json(json);
        ASSERT_EQ(events.size(), 2u) << "parse_json used to return an empty vector without reading its argument";
        EXPECT_EQ(events[0].name, "kernel_a");
        EXPECT_EQ(events[0].dur, 100u);
        EXPECT_EQ(events[1].name, "kernel_b");
        EXPECT_EQ(events[1].tid, 2);
    }

    TEST_F(FlameGraphGeneratorTest, ParserReportsNothingForAMissingFile) {
        NVTraceParser parser;
        EXPECT_TRUE(parser.parse_file("/tmp/nova-no-such-trace-file.json").empty());
    }

    TEST_F(DashboardExporterTest, EmptyExporter) {
        EXPECT_TRUE(exporter.is_empty());
    }

    TEST_F(DashboardExporterTest, ToJson) {
        std::string json = exporter.to_json();
        EXPECT_NE(json.find("header"), std::string::npos);
        EXPECT_NE(json.find("roofline"), std::string::npos);
        EXPECT_NE(json.find("fusion"), std::string::npos);
        EXPECT_NE(json.find("bandwidth"), std::string::npos);
    }

    TEST_F(DashboardExporterTest, ToCsv) {
        std::string csv = exporter.to_csv();
        EXPECT_NE(csv.find("section"), std::string::npos);
        EXPECT_NE(csv.find("roofline"), std::string::npos);
    }

    TEST_F(DashboardExporterTest, Clear) {
        exporter.clear();
        EXPECT_TRUE(exporter.is_empty());
    }

    class FlameGraphTest : public ::testing::Test {
    protected:
        FlameGraphGenerator generator;
    };

    TEST_F(FlameGraphTest, EmptyGenerator) {
        EXPECT_EQ(generator.event_count(), 0u);
    }

    TEST_F(FlameGraphTest, AddEvent) {
        ChromeTraceEvent event;
        event.name = "kernel1";
        event.category = "performance";
        event.ph = "X";
        event.ts = 1000;
        event.dur = 100;
        event.pid = 1;
        event.tid = 1;

        generator.add_event(event);
        EXPECT_EQ(generator.event_count(), 1u);
    }

    TEST_F(FlameGraphTest, BuildFlameGraph) {
        ChromeTraceEvent event;
        event.name = "matmul";
        event.category = "algo";
        event.ph = "X";
        event.ts = 1000;
        event.dur = 100;
        event.pid = 1;
        event.tid = 1;

        generator.add_event(event);
        auto flame_graph = generator.build_flame_graph();

        EXPECT_EQ(flame_graph.name, "root");
        EXPECT_EQ(flame_graph.value, 100u);
        // `>= 0u` on a size_t is always true. One event was added, so the root has exactly one
        // child; zero children would mean the event was dropped on the way into the tree.
        EXPECT_EQ(flame_graph.children.size(), 1u) << "the single added event must appear as one child of the root";
    }

    TEST_F(FlameGraphTest, ToJson) {
        ChromeTraceEvent event;
        event.name = "kernel1";
        event.category = "test";
        event.ph = "X";
        event.ts = 1000;
        event.dur = 50;
        event.pid = 1;
        event.tid = 1;

        generator.add_event(event);
        std::string json = generator.to_json();
        EXPECT_NE(json.find("root"), std::string::npos);
    }

    TEST_F(FlameGraphTest, ToChromeTrace) {
        ChromeTraceEvent event;
        event.name = "kernel1";
        event.category = "test";
        event.ph = "X";
        event.ts = 1000;
        event.dur = 50;
        event.pid = 1;
        event.tid = 1;

        generator.add_event(event);
        std::string trace = generator.to_chrome_trace();
        EXPECT_NE(trace.find("kernel1"), std::string::npos);
    }

    TEST_F(FlameGraphTest, Clear) {
        ChromeTraceEvent event;
        event.name = "kernel1";
        event.category = "test";
        event.ph = "X";
        generator.add_event(event);

        generator.clear();
        EXPECT_EQ(generator.event_count(), 0u);
    }

    TEST_F(FlameGraphTest, FlameGraphNodeJson) {
        FlameGraphNode node;
        node.name = "test";
        node.value = 100;
        node.self_value = 50;

        std::string json = node.to_json();
        EXPECT_NE(json.find("test"), std::string::npos);
        EXPECT_NE(json.find("value"), std::string::npos);
    }

    TEST_F(FlameGraphTest, NVTraceParser) {
        NVTraceParser parser;
        auto domain = parser.get_nvtx_domain("performance.nvblox");
        EXPECT_EQ(domain, "nova.performance.nvblox");

        domain = parser.get_nvtx_domain("memory");
        EXPECT_EQ(domain, "nova.memory");
    }

    class DashboardGeneratorTest : public ::testing::Test {
    protected:
        DashboardGenerator generator;
    };

    TEST_F(DashboardGeneratorTest, EmptyGenerator) {
        std::string json = generator.generate_json();
        EXPECT_NE(json.find("dashboards"), std::string::npos);
    }

    TEST_F(DashboardGeneratorTest, GenerateHtml) {
        std::string html = generator.generate_html();
        EXPECT_FALSE(html.empty());
    }

    TEST_F(DashboardGeneratorTest, AddExporter) {
        DashboardExporter exporter;
        generator.add_exporter(exporter);

        std::string json = generator.generate_json();
        EXPECT_NE(json.find("dashboards"), std::string::npos);
    }

}  // namespace cuda::performance::dashboard::test
