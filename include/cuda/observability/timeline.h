#pragma once

#include <chrono>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

// getpid() lives in <unistd.h> on POSIX and in <process.h> as _getpid() on
// Windows. Both the constructor and clear() need it, and both are in a header
// that users of the library include, so the platform split has to be here
// rather than behind a source file. The two-argument localtime case in
// logger.hpp uses the same #if defined(_WIN32) shape.
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace cuda::observability {

    namespace detail {

        /// Process id as int. Windows spells it _getpid() and returns int, POSIX
        /// getpid() returns pid_t; both fit the int that TimelineEvent stores.
        inline int current_process_id() {
#if defined(_WIN32)
            return static_cast<int>(::_getpid());
#else
            return static_cast<int>(::getpid());
#endif
        }

    }  // namespace detail

    struct TimelineEvent {
        std::string name;
        std::string category;
        int64_t timestamp_ns;
        int64_t duration_ns;
        int pid;
        int tid;
    };

    struct ChromeTraceEvent {
        std::string name;
        std::string cat;
        int64_t ts;
        int64_t dur;
        int pid;
        int tid;
        std::string ph;
        bool args_empty = true;
    };

    /// Tag type so the singleton can seed pid_ from the running process without the
    /// public default constructor doing I/O.
    struct ProcessIdTag {};

    class TimelineExporter {
    public:
        static TimelineExporter& instance();

        void begin_event(const char* name, const char* category);
        void end_event(const char* name, const char* category);
        void record_event(const char* name, const char* category, int64_t duration_ns);

        void set_process_id(int pid) { pid_ = pid; }
        void set_thread_id(int tid) { tid_ = tid; }

        bool export_to_file(const std::string& filepath);

        void clear() {
            std::lock_guard<std::mutex> lock(mutex_);
            events_.clear();
            // pid_ is writable through set_process_id() and a test that changes it has no
            // obvious reason to change it back; when it does not, every later reader in the
            // same binary attributes its events to the wrong process. Clearing restores the
            // value this exporter would have reported on its own, so clear() means "back to
            // the state I started in" rather than "drop the events and keep the overrides".
            pid_ = detail::current_process_id();
        }

        size_t event_count() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return events_.size();
        }

    private:
        TimelineExporter() = default;

        /// Seeds pid_ from detail::current_process_id(). The exporter previously
        /// left pid_ at 0 and
        /// nothing in production ever called set_process_id, so every trace
        /// advertised itself as process 0 — a value Chrome's trace viewer cannot use
        /// to tell processes apart.
        explicit TimelineExporter(ProcessIdTag);
        ~TimelineExporter() = default;
        TimelineExporter(const TimelineExporter&) = delete;
        TimelineExporter& operator=(const TimelineExporter&) = delete;

        int64_t get_timestamp_ns();
        std::string escape_json_string(const std::string& s);

        mutable std::mutex mutex_;
        std::vector<ChromeTraceEvent> events_;
        std::vector<int64_t> event_stack_;
        int pid_ = 0;
        int tid_ = 1;
    };

    class ScopedTimelineEvent {
    public:
        ScopedTimelineEvent(const char* name, const char* category)
            : name_(name),
              category_(category) {
            TimelineExporter::instance().begin_event(name_, category_);
            start_time_ = std::chrono::high_resolution_clock::now();
        }

        ~ScopedTimelineEvent() {
            // start_time_ is recorded so the class documents its own lifetime, but the
            // duration is measured inside end_event() from its own timestamp -- computing
            // it here as well produced a value that was then discarded.
            TimelineExporter::instance().end_event(name_, category_);
        }

    private:
        const char* name_;
        const char* category_;
        std::chrono::high_resolution_clock::time_point start_time_;
    };

#ifdef __COUNTER__
#define NOVA_TIMELINE_SCOPED(name, category) cuda::observability::ScopedTimelineEvent NOVA_CONCAT(_nova_timeline_event_, __COUNTER__)(name, category)
#define NOVA_CONCAT(a, b) NOVA_CONCAT_IMPL(a, b)
#define NOVA_CONCAT_IMPL(a, b) a##b
#else
#define NOVA_TIMELINE_SCOPED(name, category) cuda::observability::ScopedTimelineEvent _nova_timeline_event_(name, category)
#endif

#define NOVA_TIMELINE_BEGIN(name, category) cuda::observability::TimelineExporter::instance().begin_event(name, category)

#define NOVA_TIMELINE_END(name, category) cuda::observability::TimelineExporter::instance().end_event(name, category)

#define NOVA_TIMELINE_RECORD(name, category, duration_ns) cuda::observability::TimelineExporter::instance().record_event(name, category, duration_ns)

#define NOVA_TIMELINE_EXPORT(filepath) cuda::observability::TimelineExporter::instance().export_to_file(filepath)

}  // namespace cuda::observability
