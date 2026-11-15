/**
 * @file logger.hpp
 * @brief Structured logging infrastructure
 * @defgroup logger Structured Logging
 * @ingroup observability
 *
 * Provides structured logging with multiple severity levels:
 * - ERROR: Failures preventing operation
 * - WARN: Recoverable issues, degraded performance
 * - INFO: Significant milestones
 * - DEBUG: Detailed trace information
 * - TRACE: Per-iteration logging (disabled by default)
 *
 * Example usage:
 * @code
 * NOVA_LOG_INFO("operation=allocate", "bytes=" << size);
 * NOVA_LOG_WARN("device=" << device_id, "reason=low_memory");
 * @endcode
 *
 * Compile-time filtering via NOVA_LOG_LEVEL
 */

#ifndef NOVA_CUDA_LOGGER_HPP
#define NOVA_CUDA_LOGGER_HPP

#include <cstdio>
#include <string>
#include <sstream>
#include <chrono>
#include <ctime>
#include <utility>

namespace nova {
namespace logging {

/**
 * @brief Log severity levels
 * @enum LogLevel
 * @ingroup logger
 */
enum class LogLevel {
    ERROR = 0,
    WARN = 1,
    INFO = 2,
    DEBUG = 3,
    TRACE = 4
};

/**
 * @brief Get current log level
 * @return Active log level
 * @note Can be overridden at compile time via NOVA_LOG_LEVEL
 */
inline LogLevel get_log_level() {
#if defined(NOVA_LOG_LEVEL_TRACE)
    return LogLevel::TRACE;
#elif defined(NOVA_LOG_LEVEL_DEBUG)
    return LogLevel::DEBUG;
#elif defined(NOVA_LOG_LEVEL_INFO)
    return LogLevel::INFO;
#elif defined(NOVA_LOG_LEVEL_WARN)
    return LogLevel::WARN;
#elif defined(NOVA_LOG_LEVEL_ERROR)
    return LogLevel::ERROR;
#else
    return LogLevel::INFO;
#endif
}

/**
 * @brief Check if a level should be logged
 * @param level Level to check
 * @return true if level should be output
 */
inline bool should_log(LogLevel level) {
    return static_cast<int>(level) <= static_cast<int>(get_log_level());
}

/**
 * @brief Format timestamp for log output
 * @return ISO 8601 formatted timestamp
 */
inline std::string format_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    // std::localtime returns a pointer to a shared static buffer, so two threads
    // logging at once raced on it (ThreadSanitizer flags format_timestamp
    // directly). localtime_r writes into caller-owned storage.
    std::tm broken_down{};
#if defined(_WIN32)
    localtime_s(&broken_down, &time);
#else
    localtime_r(&time, &broken_down);
#endif

    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &broken_down);

    // Milliseconds are zero-padded to three digits. Without this a timestamp at
    // +7ms read ".7", which is both wrong for ISO 8601 and breaks the
    // lexicographic ordering of a fixed-width log line.
    char millis[8];
    std::snprintf(millis, sizeof(millis), "%03lld",
                  static_cast<long long>(ms.count()));
    return std::string(buf) + "." + millis;
}

/**
 * @brief Log a message at specified level
 * @param level Log level
 * @param context Key=value context pairs
 * @param message Log message
 */
inline const char* level_name(LogLevel level) {
    switch (level) {
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::TRACE: return "TRACE";
    }
    return "UNKNOWN";
}

/**
 * @brief Log a preformatted message at the specified level
 * @param level Log level
 * @param context Key=value context pairs
 * @param message Log message
 */
inline void write_log_line(LogLevel level, const char* context,
                           const std::string& message) {
    std::fprintf(stderr, "[%s] [%s] [%s] %s\n",
                 format_timestamp().c_str(),
                 level_name(level),
                 context ? context : "",
                 message.c_str());
}

/** @brief Log an already-formatted message */
inline void log(LogLevel level, const char* context, const std::string& message) {
    if (!should_log(level)) return;
    write_log_line(level, context, message);
}

/**
 * @brief Log a message built from a streamed expression
 *
 * The macros accept `"key=" << value` (see the example at the top of this
 * file), so something has to turn that into a string. Without this overload the
 * documented form did not compile at all: the macro passed the streamed
 * expression to a `const char*` parameter.
 *
 * Only the selected stream operator runs when the level is filtered out, so an
 * expensive value is not formatted for a message that is then discarded.
 */
template <typename... Args>
void log(LogLevel level, const char* context, Args&&... args) {
    if (!should_log(level)) return;

    std::ostringstream stream;
    (stream << ... << std::forward<Args>(args));
    // Calls write_log_line, not log(): an unqualified log(...) with a
    // std::string argument selects this template again (a forwarding reference
    // binds better than const&) and recurses until the stack is gone.
    write_log_line(level, context, stream.str());
}

}  // namespace logging
}  // namespace nova

/**
 * @def NOVA_LOG_ERROR(context, message)
 * Log ERROR level message
 * @ingroup logger
 */
#define NOVA_LOG_ERROR(context, ...) \
    do { \
        if (nova::logging::should_log(nova::logging::LogLevel::ERROR)) { \
            std::ostringstream nova_log_stream_; \
            nova_log_stream_ << __VA_ARGS__; \
            nova::logging::log(nova::logging::LogLevel::ERROR, context, \
                               nova_log_stream_.str()); \
        } \
    } while (false)

/**
 * @def NOVA_LOG_WARN(context, message)
 * Log WARN level message
 * @ingroup logger
 */
#define NOVA_LOG_WARN(context, ...) \
    do { \
        if (nova::logging::should_log(nova::logging::LogLevel::WARN)) { \
            std::ostringstream nova_log_stream_; \
            nova_log_stream_ << __VA_ARGS__; \
            nova::logging::log(nova::logging::LogLevel::WARN, context, \
                               nova_log_stream_.str()); \
        } \
    } while (false)

/**
 * @def NOVA_LOG_INFO(context, message)
 * Log INFO level message
 * @ingroup logger
 */
#define NOVA_LOG_INFO(context, ...) \
    do { \
        if (nova::logging::should_log(nova::logging::LogLevel::INFO)) { \
            std::ostringstream nova_log_stream_; \
            nova_log_stream_ << __VA_ARGS__; \
            nova::logging::log(nova::logging::LogLevel::INFO, context, \
                               nova_log_stream_.str()); \
        } \
    } while (false)

/**
 * @def NOVA_LOG_DEBUG(context, message)
 * Log DEBUG level message
 * @ingroup logger
 */
#define NOVA_LOG_DEBUG(context, ...) \
    do { \
        if (nova::logging::should_log(nova::logging::LogLevel::DEBUG)) { \
            std::ostringstream nova_log_stream_; \
            nova_log_stream_ << __VA_ARGS__; \
            nova::logging::log(nova::logging::LogLevel::DEBUG, context, \
                               nova_log_stream_.str()); \
        } \
    } while (false)

/**
 * @def NOVA_LOG_TRACE(context, message)
 * Log TRACE level message
 * @ingroup logger
 */
#define NOVA_LOG_TRACE(context, ...) \
    do { \
        if (nova::logging::should_log(nova::logging::LogLevel::TRACE)) { \
            std::ostringstream nova_log_stream_; \
            nova_log_stream_ << __VA_ARGS__; \
            nova::logging::log(nova::logging::LogLevel::TRACE, context, \
                               nova_log_stream_.str()); \
        } \
    } while (false)

#endif  // NOVA_CUDA_LOGGER_HPP
