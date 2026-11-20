/**
 * @file logger_test.cpp
 * @brief Contracts for the structured logging helpers
 *
 * include/cuda/observability/logger.hpp is the structured logging entry point
 * every other module uses. It had no tests, and three defects lived in it:
 *
 * - The documented usage at the top of the file is
 *       NOVA_LOG_INFO("operation=allocate", "bytes=" << size);
 *   but log() took a `const char*`, so the macro passed the streamed expression
 *   to that parameter and the header's own example did not compile ("invalid
 *   operands of types 'const char[7]' and 'std::size_t' to binary operator<<").
 *
 * - format_timestamp() called std::localtime, which returns a pointer to a
 *   shared static buffer. Two threads logging concurrently raced on it
 *   (ThreadSanitizer reports a data race inside format_timestamp).
 *
 * - It appended the millisecond count with std::to_string, so a timestamp at
 *   +7ms read ".7" instead of ".007". The field was not fixed width, which is
 *   both wrong for ISO 8601 and breaks ordering a log by string.
 */

#include <cuda/observability/logger.hpp>

#include <gtest/gtest.h>

#include "test_temp_paths.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace nova::logging::test {

    namespace {

        // A fixed filename collided when the suite ran under `ctest -j`: every process
        // pointed freopen() at the same path, so one test read another's output or read
        // the file after a peer had truncated it. The name now carries the pid and a
        // per-call counter, and the file is uniquely owned by this call.
        std::string capture_log(const std::function<void()>& emit) {
            static std::atomic<unsigned> counter {0};

            const auto path = nova::test::temp_dir() / ("nova_logger_test." + std::to_string(nova::test::process_id()) + "." + std::to_string(counter.fetch_add(1)) + ".log");

            std::fflush(stderr);
            FILE* saved = stderr;
            FILE* redirected = std::freopen(path.string().c_str(), "w", stderr);
            if (redirected == nullptr) {
                stderr = saved;
                return {};
            }

            emit();

            std::fflush(stderr);
            // freopen replaced the C stream backing stderr; point it back at the
            // original before reading the captured file.
            stderr = saved;

            std::ifstream input(path, std::ios::binary);
            const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            input.close();
            std::filesystem::remove(path);
            return contents;
        }

    }  // namespace

    // Regression: this exact form is the header's documented example and did not
    // compile before the stream overload existed.
    TEST(LoggerTest, DocumentedStreamingFormCompilesAndPrints) {
        const std::size_t size = 64;
        const std::string output = capture_log([&size] { NOVA_LOG_INFO("operation=allocate", "bytes=" << size); });

        EXPECT_NE(output.find("bytes=64"), std::string::npos) << output;
        EXPECT_NE(output.find("operation=allocate"), std::string::npos) << output;
        EXPECT_NE(output.find("[INFO]"), std::string::npos) << output;
    }

    TEST(LoggerTest, LevelControlGatesOutput) {
        // Default level is INFO, so DEBUG and TRACE are filtered out.
        if (get_log_level() == LogLevel::INFO) {
            const std::string output = capture_log([] {
                NOVA_LOG_DEBUG("ctx", "debug-message");
                NOVA_LOG_INFO("ctx", "info-message");
            });
            EXPECT_EQ(output.find("debug-message"), std::string::npos) << "DEBUG must be filtered at INFO level";
            EXPECT_NE(output.find("info-message"), std::string::npos) << output;
        }
    }

    TEST(LoggerTest, SeverityOrderingIsErrorFirst) {
        EXPECT_LT(static_cast<int>(LogLevel::ERROR), static_cast<int>(LogLevel::WARN));
        EXPECT_LT(static_cast<int>(LogLevel::WARN), static_cast<int>(LogLevel::INFO));
        EXPECT_LT(static_cast<int>(LogLevel::INFO), static_cast<int>(LogLevel::DEBUG));
        EXPECT_LT(static_cast<int>(LogLevel::DEBUG), static_cast<int>(LogLevel::TRACE));

        // should_log keeps everything at or below the active level.
        EXPECT_TRUE(should_log(LogLevel::ERROR));
        EXPECT_EQ(should_log(LogLevel::TRACE), get_log_level() == LogLevel::TRACE);
    }

    TEST(LoggerTest, LevelNamesAreStable) {
        EXPECT_STREQ(level_name(LogLevel::ERROR), "ERROR");
        EXPECT_STREQ(level_name(LogLevel::WARN), "WARN");
        EXPECT_STREQ(level_name(LogLevel::INFO), "INFO");
        EXPECT_STREQ(level_name(LogLevel::DEBUG), "DEBUG");
        EXPECT_STREQ(level_name(LogLevel::TRACE), "TRACE");
    }

    // Regression: the fraction was printed with to_string, so it was 1-3 digits wide.
    TEST(LoggerTest, TimestampMillisecondsAreZeroPaddedToThreeDigits) {
        // Polling for a wall clock that happens to land on a millisecond below 100
        // makes this depend on scheduling. Assert the invariant directly instead:
        // EVERY timestamp is exactly three fractional digits, and the value must
        // equal the system clock's millisecond field, which is not padded by
        // std::to_string.
        for (int attempt = 0; attempt < 2000; ++attempt) {
            const auto now = std::chrono::system_clock::now();
            const auto expected_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            const std::string timestamp = format_timestamp();

            const auto dot = timestamp.rfind('.');
            ASSERT_NE(dot, std::string::npos) << timestamp;
            const std::string fraction = timestamp.substr(dot + 1);
            ASSERT_EQ(fraction.size(), 3u) << "milliseconds must be fixed width: " << timestamp;

            // Allow the clock to tick between the two reads; the width is the
            // property under test, and it held above.
            const int printed = std::stoi(fraction);
            const int actual = static_cast<int>(expected_ms.count());
            EXPECT_LE(std::abs(printed - actual), 5) << "printed " << printed << " ms, clock says " << actual;
            if (fraction[0] == '0') {
                return;  // observed a zero-padded value; the padding is real
            }
        }
    }

    TEST(LoggerTest, TimestampHasAnIso8601Shape) {
        const std::string timestamp = format_timestamp();

        // YYYY-MM-DDTHH:MM:SS.mmm
        EXPECT_EQ(timestamp.size(), 23u) << timestamp;
        EXPECT_EQ(timestamp[4], '-');
        EXPECT_EQ(timestamp[7], '-');
        EXPECT_EQ(timestamp[10], 'T');
        EXPECT_EQ(timestamp[13], ':');
        EXPECT_EQ(timestamp[16], ':');
        EXPECT_EQ(timestamp[19], '.');
    }

    TEST(LoggerTest, TimestampsAreNonDecreasing) {
        const std::string first = format_timestamp();
        const std::string second = format_timestamp();

        // Fixed width plus zero padding makes lexicographic order match time order.
        EXPECT_LE(first, second) << "a fixed-width timestamp must sort chronologically: " << first << " > " << second;
    }

    // Regression: std::localtime shares one static buffer, so concurrent logging
    // raced on it. This cannot assert the absence of a race by itself, but it does
    // prove concurrent formatting produces well-formed, independent results — run
    // under ThreadSanitizer the old code reported a data race here.
    TEST(LoggerTest, ConcurrentFormattingProducesWellFormedTimestamps) {
        constexpr int kThreads = 8;
        constexpr int kIterations = 100;
        std::vector<std::thread> workers;
        std::atomic<int> malformed {0};

        for (int i = 0; i < kThreads; ++i) {
            workers.emplace_back([&malformed] {
                for (int iteration = 0; iteration < kIterations; ++iteration) {
                    const std::string timestamp = format_timestamp();
                    if (timestamp.size() != 23u || timestamp[10] != 'T' || timestamp[19] != '.') {
                        malformed.fetch_add(1);
                    }
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }

        EXPECT_EQ(malformed.load(), 0) << "concurrent formatting produced a bad timestamp";
    }

    TEST(LoggerTest, NullContextIsPrintedAsEmpty) {
        const std::string output = capture_log([] { NOVA_LOG_INFO(nullptr, "message-with-null-context"); });

        EXPECT_NE(output.find("message-with-null-context"), std::string::npos) << output;
        EXPECT_NE(output.find("[]"), std::string::npos) << output;
    }

}  // namespace nova::logging::test
