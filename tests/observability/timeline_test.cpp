#include <gtest/gtest.h>

#include <unistd.h>
#include <fstream>
#include <filesystem>

#include "cuda/observability/timeline.h"

namespace cuda::observability::test {

class TimelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        TimelineExporter::instance().clear();
        TimelineExporter::instance().set_process_id(1);
        TimelineExporter::instance().set_thread_id(1);
    }

    void TearDown() override {
        TimelineExporter::instance().clear();
        // SetUp overrides the process id with 1 so events are reproducible, and only the
        // events were being cleared afterwards. The overridden id stayed on the singleton
        // and every later test in the binary saw pid 1 — a test in another file that
        // asserts the exported JSON carries ::getpid() failed only when this suite ran
        // first. Restore what the singleton would have reported on its own.
        TimelineExporter::instance().set_process_id(static_cast<int>(::getpid()));
    }
};

TEST_F(TimelineTest, ScopedEventRecordsDuration) {
    {
        NOVA_TIMELINE_SCOPED("test_kernel", "test");
    }

    EXPECT_EQ(TimelineExporter::instance().event_count(), 1);
}

TEST_F(TimelineTest, ManualBeginEndEvents) {
    NOVA_TIMELINE_BEGIN("kernel_start", "test");
    NOVA_TIMELINE_END("kernel_start", "test");

    EXPECT_EQ(TimelineExporter::instance().event_count(), 1);
}

TEST_F(TimelineTest, MultipleNestedEvents) {
    NOVA_TIMELINE_BEGIN("outer", "test");
    NOVA_TIMELINE_BEGIN("inner1", "test");
    NOVA_TIMELINE_END("inner1", "test");
    NOVA_TIMELINE_BEGIN("inner2", "test");
    NOVA_TIMELINE_END("inner2", "test");
    NOVA_TIMELINE_END("outer", "test");

    // One event per opened/closed scope (outer, inner1, inner2) - consistent
    // with ManualBeginEndEvents where one begin/end pair records exactly one
    // event.
    EXPECT_EQ(TimelineExporter::instance().event_count(), 3);
}

TEST_F(TimelineTest, RecordInstantEvent) {
    NOVA_TIMELINE_RECORD("instant", "test", 100);

    EXPECT_EQ(TimelineExporter::instance().event_count(), 1);
}

TEST_F(TimelineTest, ExportToFileCreatesValidJson) {
    NOVA_TIMELINE_SCOPED("test1", "cat1");
    NOVA_TIMELINE_SCOPED("test2", "cat2");

    std::string test_file = "/tmp/nova_timeline_test.json";
    bool success = NOVA_TIMELINE_EXPORT(test_file);

    EXPECT_TRUE(success);
    EXPECT_TRUE(std::filesystem::exists(test_file));

    std::ifstream file(test_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("\"traceEvents\"") != std::string::npos);
    EXPECT_TRUE(content.find("\"cat\":") != std::string::npos);
    EXPECT_TRUE(content.find("\"name\":") != std::string::npos);
    EXPECT_TRUE(content.find("\"ts\":") != std::string::npos);

    std::filesystem::remove(test_file);
}

TEST_F(TimelineTest, ExportEmptyTimeline) {
    std::string test_file = "/tmp/nova_timeline_empty.json";
    bool success = NOVA_TIMELINE_EXPORT(test_file);

    EXPECT_TRUE(success);

    std::ifstream file(test_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    // The exporter emits "\"traceEvents\": [\n" then entries then "],".
    EXPECT_TRUE(content.find("\"traceEvents\": [\n]") != std::string::npos ||
                content.find("\"traceEvents\": []") != std::string::npos);

    std::filesystem::remove(test_file);
}

TEST_F(TimelineTest, JsonStringEscaping) {
    NOVA_TIMELINE_SCOPED("test\"with\"quotes", "cat\"with\"quotes");

    std::string test_file = "/tmp/nova_timeline_escape.json";
    bool success = NOVA_TIMELINE_EXPORT(test_file);

    EXPECT_TRUE(success);

    std::ifstream file(test_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("\\\"") != std::string::npos);

    std::filesystem::remove(test_file);
}

TEST_F(TimelineTest, ClearResetsEventCount) {
    NOVA_TIMELINE_SCOPED("test", "cat");
    EXPECT_EQ(TimelineExporter::instance().event_count(), 1);

    TimelineExporter::instance().clear();
    EXPECT_EQ(TimelineExporter::instance().event_count(), 0);
}

}  // namespace cuda::observability::test
