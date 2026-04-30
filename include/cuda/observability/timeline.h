#pragma once

#include <chrono>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace cuda::observability {

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
    }

    size_t event_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.size();
    }

private:
    TimelineExporter() = default;
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
        : name_(name), category_(category) {
        TimelineExporter::instance().begin_event(name_, category_);
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    ~ScopedTimelineEvent() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time_).count();
        TimelineExporter::instance().end_event(name_, category_);
    }

private:
    const char* name_;
    const char* category_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

#define NOVA_TIMELINE_SCOPED(name, category) \
    cuda::observability::ScopedTimelineEvent _nova_timeline_event_(name, category)

#define NOVA_TIMELINE_BEGIN(name, category) \
    cuda::observability::TimelineExporter::instance().begin_event(name, category)

#define NOVA_TIMELINE_END(name, category) \
    cuda::observability::TimelineExporter::instance().end_event(name, category)

#define NOVA_TIMELINE_RECORD(name, category, duration_ns) \
    cuda::observability::TimelineExporter::instance().record_event(name, category, duration_ns)

#define NOVA_TIMELINE_EXPORT(filepath) \
    cuda::observability::TimelineExporter::instance().export_to_file(filepath)

}  // namespace cuda::observability
