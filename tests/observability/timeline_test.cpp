#include <gtest/gtest.h>

#include "test_temp_paths.h"
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
            TimelineExporter::instance().set_process_id(nova::test::process_id());
        }
    };

    TEST_F(TimelineTest, ScopedEventRecordsDuration) {
        { NOVA_TIMELINE_SCOPED("test_kernel", "test"); }

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
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

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
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        // The exporter emits "\"traceEvents\": [\n" then entries then "],".
        EXPECT_TRUE(content.find("\"traceEvents\": [\n]") != std::string::npos || content.find("\"traceEvents\": []") != std::string::npos);

        std::filesystem::remove(test_file);
    }

    TEST_F(TimelineTest, JsonStringEscaping) {
        NOVA_TIMELINE_SCOPED("test\"with\"quotes", "cat\"with\"quotes");

        std::string test_file = "/tmp/nova_timeline_escape.json";
        bool success = NOVA_TIMELINE_EXPORT(test_file);

        EXPECT_TRUE(success);

        std::ifstream file(test_file);
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        EXPECT_TRUE(content.find("\\\"") != std::string::npos);

        std::filesystem::remove(test_file);
    }

    // ScopedTimelineEvent closes an event in its destructor by calling end_event(), which pops the
    // exporter's open-event stack. It declared no copy or move operations, so the compiler generated
    // a copy: N copies each pop the stack once, closing records that belong to other scopes and
    // leaving the trace with mismatched or negative durations and no error anywhere.
    //
    // Moving is refused too, for the same reason as ScopedKernelTiming: the record boundary is where
    // the object dies, so relocating it would move that boundary.
    TEST_F(TimelineTest, ScopedEventIsExclusiveToItsScope) {
        static_assert(!std::is_copy_constructible_v<ScopedTimelineEvent>, "a copy would close another scope's event");
        static_assert(!std::is_copy_assignable_v<ScopedTimelineEvent>, "a copy would close another scope's event");
        static_assert(!std::is_move_constructible_v<ScopedTimelineEvent>, "the event closes on destruction, so it must not be relocatable");
        static_assert(!std::is_move_assignable_v<ScopedTimelineEvent>, "the event closes on destruction, so it must not be relocatable");

        auto& exporter = TimelineExporter::instance();
        {
            ScopedTimelineEvent scoped("exclusive-probe", "cat");
            EXPECT_EQ(exporter.event_count(), 1u);
        }
        // One construction, one close -- no more.
        EXPECT_EQ(exporter.event_count(), 1u) << "the scope must produce exactly one record";

        const std::string path = nova::test::temp_path("timeline-exclusive-", ".json");
        ASSERT_TRUE(exporter.export_to_file(path));
        std::ifstream in(path);
        const std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::filesystem::remove(path);
        EXPECT_NE(json.find("exclusive-probe"), std::string::npos);
    }

    TEST_F(TimelineTest, ClearResetsEventCount) {
        NOVA_TIMELINE_SCOPED("test", "cat");
        EXPECT_EQ(TimelineExporter::instance().event_count(), 1);

        TimelineExporter::instance().clear();
        EXPECT_EQ(TimelineExporter::instance().event_count(), 0);
    }

    // end_event() used to overwrite the record begin_event() had opened, setting ph = 'E' on it and
    // storing the duration there. Chrome's B/E form needs a matching "B" for every "E", so a trace
    // built from begin/end pairs contained only "E" records with nothing to pair them with. Measured
    // before the fix: two nested pairs exported as two entries, both ph "E", no "B" anywhere.
    //
    // The records are 'X' now -- one complete event carrying its own duration -- which needs no
    // pairing and cannot be left half-closed.
    TEST_F(TimelineTest, BeginEndProduceCompleteEventsNotUnpairedEnds) {
        auto& timeline = TimelineExporter::instance();
        timeline.begin_event("outer", "gpu");
        timeline.begin_event("inner", "gpu");
        timeline.end_event("inner", "gpu");
        timeline.end_event("outer", "gpu");

        ASSERT_TRUE(timeline.export_to_file("/tmp/nova_timeline_complete.json"));
        std::ifstream in("/tmp/nova_timeline_complete.json");
        ASSERT_TRUE(in.good());
        std::ostringstream buffer;
        buffer << in.rdbuf();
        const std::string json = buffer.str();

        EXPECT_NE(json.find("\"ph\": \"X\""), std::string::npos) << "records must be complete events: " << json;
        EXPECT_EQ(json.find("\"ph\": \"E\""), std::string::npos) << "no end may be left unpaired: " << json;
        EXPECT_NE(json.find("\"dur\": "), std::string::npos) << "the duration must be exported: " << json;
        EXPECT_NE(json.find("outer"), std::string::npos);
        EXPECT_NE(json.find("inner"), std::string::npos);
    }

}  // namespace cuda::observability::test
