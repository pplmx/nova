#include <gtest/gtest.h>
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

    EXPECT_EQ(TimelineExporter::instance().event_count(), 4);
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

    EXPECT_TRUE(content.find("\"traceEvents\": []") != std::string::npos ||
                content.find("\"traceEvents\":[\n]") != std::string::npos);

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
