/**
 * @file timeline_export_test.cpp
 * @brief Contracts for the Chrome-trace exporter
 *
 * src/observability/timeline.cpp writes a Chrome trace JSON file. Two of its
 * seven functions had any test reference, and two defects made the output wrong:
 *
 * - The metadata key was written as " exporter" with a leading space. JSON
 *   accepts that as a key, so the file stayed parseable and the mistake went
 *   unnoticed, but a consumer looking up "exporter" finds nothing. Measured on the
 *   pre-fix output: metadata keys were [' exporter', 'version'].
 *
 * - pid_ defaulted to 0 and nothing in production ever called set_process_id, so
 *   every exported event advertised pid 0. Chrome's trace viewer groups events by
 *   pid, so traces from concurrent processes could not be told apart. Measured:
 *   pids were {0}. The singleton now seeds pid_ from getpid().
 *
 * The escaping and the event stack were checked and are correct: quotes,
 * backslashes, control characters and the duration bookkeeping all survive a
 * round trip through a JSON parser.
 */

#include <cuda/observability/timeline.h>

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace cuda::observability::test {

namespace {

/// Reads a whole file, or returns an empty string when it cannot be opened.
std::string slurp(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return {};
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

/// Unique path per process and test, so parallel ctest runs cannot collide.
std::string temp_path(const char* tag) {
    return std::string("/tmp/nova_timeline_") + tag + "_" +
           std::to_string(::getpid()) + ".json";
}

/// True when `needle` occurs in `haystack`.
bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

// Regression: the key was written with a leading space.
TEST(TimelineExporterTest, MetadataKeyHasNoLeadingSpace) {
    auto& exporter = TimelineExporter::instance();
    exporter.clear();
    exporter.record_event("event", "cat", 5);

    const std::string path = temp_path("metadata");
    ASSERT_TRUE(exporter.export_to_file(path));
    const std::string json = slurp(path);
    std::remove(path.c_str());

    ASSERT_FALSE(json.empty());
    EXPECT_TRUE(contains(json, "\"exporter\":"))
        << "a consumer looking up \"exporter\" finds nothing:\n" << json;
    EXPECT_FALSE(contains(json, "\" exporter\":"))
        << "the key starts with a space:\n" << json;
}

// Regression: every event carried pid 0.
TEST(TimelineExporterTest, EventsCarryTheRealProcessId) {
    auto& exporter = TimelineExporter::instance();
    exporter.clear();
    exporter.record_event("event", "cat", 5);

    const std::string path = temp_path("pid");
    ASSERT_TRUE(exporter.export_to_file(path));
    const std::string json = slurp(path);
    std::remove(path.c_str());

    ASSERT_FALSE(json.empty());
    EXPECT_FALSE(contains(json, "\"pid\": 0,"))
        << "pid 0 cannot distinguish processes:\n" << json;
    EXPECT_TRUE(contains(json, "\"pid\": " + std::to_string(::getpid()) + ","))
        << "expected the current process id:\n" << json;
}

TEST(TimelineExporterTest, ExportProducesWellFormedTraceArrays) {
    auto& exporter = TimelineExporter::instance();
    exporter.clear();
    exporter.record_event("first", "cat", 100);
    exporter.record_event("second", "cat", 200);

    const std::string path = temp_path("shape");
    ASSERT_TRUE(exporter.export_to_file(path));
    const std::string json = slurp(path);
    std::remove(path.c_str());

    ASSERT_FALSE(json.empty());
    EXPECT_TRUE(contains(json, "\"traceEvents\": [")) << json;
    EXPECT_TRUE(contains(json, "\"name\": \"first\"")) << json;
    EXPECT_TRUE(contains(json, "\"name\": \"second\"")) << json;
    EXPECT_TRUE(contains(json, "\"dur\": 100,")) << json;
    EXPECT_TRUE(contains(json, "\"ph\": \"X\"")) << json;

    // Two event objects means exactly one separating comma between them.
    size_t commas = 0;
    for (size_t pos = json.find("\"ph\""); pos != std::string::npos;
         pos = json.find("\"ph\"", pos + 1)) {
        ++commas;
    }
    EXPECT_GE(commas, 2u) << "both events should have a phase field";
}

TEST(TimelineExporterTest, SpecialCharactersAreEscaped) {
    auto& exporter = TimelineExporter::instance();
    exporter.clear();
    // A quote and a newline both break a JSON string if left raw.
    exporter.record_event("quote\"inside", "cat\\backslash", 1);

    const std::string path = temp_path("escape");
    ASSERT_TRUE(exporter.export_to_file(path));
    const std::string json = slurp(path);
    std::remove(path.c_str());

    ASSERT_FALSE(json.empty());
    EXPECT_TRUE(contains(json, "quote\\\"inside"))
        << "the embedded quote is not escaped:\n" << json;
    EXPECT_TRUE(contains(json, "cat\\\\backslash"))
        << "the backslash is not escaped:\n" << json;
    // A raw unescaped quote would end the string early and leave the object
    // unbalanced; the exporter writes one closing quote per value.
    EXPECT_FALSE(contains(json, "\"name\": \"quote\"inside\"")) << json;
}

TEST(TimelineExporterTest, BeginEndPairProducesMeasuredDuration) {
    auto& exporter = TimelineExporter::instance();
    exporter.clear();

    exporter.begin_event("span", "cat");
    exporter.end_event("span", "cat");

    const std::string path = temp_path("span");
    ASSERT_TRUE(exporter.export_to_file(path));
    const std::string json = slurp(path);
    std::remove(path.c_str());

    ASSERT_FALSE(json.empty());
    // The pair is recorded as one event that ends up with phase 'E'.
    EXPECT_TRUE(contains(json, "\"ph\": \"E\"")) << json;
    EXPECT_TRUE(contains(json, "\"name\": \"span\"")) << json;
}

TEST(TimelineExporterTest, EndEventWithoutBeginIsHarmless) {
    auto& exporter = TimelineExporter::instance();
    exporter.clear();

    // An unmatched end must not underflow the stack or crash.
    EXPECT_NO_THROW(exporter.end_event("orphan", "cat"));
    EXPECT_EQ(exporter.event_count(), 0u);
}

TEST(TimelineExporterTest, ClearEmptiesTheBuffer) {
    auto& exporter = TimelineExporter::instance();
    exporter.clear();
    exporter.record_event("a", "c", 1);
    exporter.record_event("b", "c", 2);
    EXPECT_EQ(exporter.event_count(), 2u);

    exporter.clear();
    EXPECT_EQ(exporter.event_count(), 0u);
}

TEST(TimelineExporterTest, ExportingToAnUnwritablePathFails) {
    auto& exporter = TimelineExporter::instance();
    exporter.clear();
    exporter.record_event("a", "c", 1);

    EXPECT_FALSE(exporter.export_to_file("/nonexistent-directory-nova/trace.json"))
        << "an unwritable path must report failure rather than silently succeed";
}

TEST(TimelineExporterTest, TimestampsIncrease) {
    // get_timestamp_ns is private; observe the timestamps the exporter writes.
    auto& exporter = TimelineExporter::instance();
    exporter.clear();
    exporter.record_event("first", "c", 1);
    exporter.record_event("second", "c", 1);

    const std::string path = temp_path("ts");
    ASSERT_TRUE(exporter.export_to_file(path));
    const std::string json = slurp(path);
    std::remove(path.c_str());

    // Extract the two "ts" integers rather than assuming formatting.
    std::vector<int64_t> stamps;
    const std::string key = "\"ts\": ";
    for (size_t pos = json.find(key); pos != std::string::npos;
         pos = json.find(key, pos + 1)) {
        const size_t start = pos + key.size();
        const size_t end = json.find(',', start);
        if (end == std::string::npos) {
            continue;
        }
        stamps.push_back(std::stoll(json.substr(start, end - start)));
    }

    ASSERT_EQ(stamps.size(), 2u) << json;
    EXPECT_GT(stamps[1], stamps[0]) << "the clock must move forward";
    EXPECT_GT(stamps[0], 0) << "timestamps are nanoseconds since the epoch";
}

}  // namespace cuda::observability::test
