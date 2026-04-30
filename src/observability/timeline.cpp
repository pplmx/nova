#include "cuda/observability/timeline.h"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace cuda::observability {

TimelineExporter& TimelineExporter::instance() {
    static TimelineExporter instance;
    return instance;
}

int64_t TimelineExporter::get_timestamp_ns() {
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();
}

std::string TimelineExporter::escape_json_string(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b";  break;
            case '\f': result += "\\f";  break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    std::ostringstream oss;
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                    result += oss.str();
                } else {
                    result += c;
                }
        }
    }
    return result;
}

void TimelineExporter::begin_event(const char* name, const char* category) {
    std::lock_guard<std::mutex> lock(mutex_);
    ChromeTraceEvent event;
    event.name = name;
    event.cat = category;
    event.ts = get_timestamp_ns();
    event.dur = 0;
    event.pid = pid_;
    event.tid = tid_;
    event.ph = 'B';
    event.args_empty = true;
    events_.push_back(event);
    event_stack_.push_back(events_.size() - 1);
}

void TimelineExporter::end_event(const char* name, const char* category) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!event_stack_.empty()) {
        size_t idx = event_stack_.back();
        event_stack_.pop_back();
        events_[idx].dur = get_timestamp_ns() - events_[idx].ts;
        events_[idx].ph = 'E';
    }
}

void TimelineExporter::record_event(const char* name, const char* category, int64_t duration_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    ChromeTraceEvent event;
    event.name = name;
    event.cat = category;
    event.ts = get_timestamp_ns();
    event.dur = duration_ns;
    event.pid = pid_;
    event.tid = tid_;
    event.ph = 'X';
    event.args_empty = true;
    events_.push_back(event);
}

bool TimelineExporter::export_to_file(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    file << "{\n\"traceEvents\": [\n";

    for (size_t i = 0; i < events_.size(); ++i) {
        const auto& e = events_[i];
        file << "  {\n";
        file << "    \"name\": \"" << escape_json_string(e.name) << "\",\n";
        file << "    \"cat\": \"" << escape_json_string(e.cat) << "\",\n";
        file << "    \"ts\": " << e.ts << ",\n";
        if (e.dur > 0) {
            file << "    \"dur\": " << e.dur << ",\n";
        }
        file << "    \"pid\": " << e.pid << ",\n";
        file << "    \"tid\": " << e.tid << ",\n";
        file << "    \"ph\": \"" << e.ph << "\"\n";
        file << "  }";
        if (i < events_.size() - 1) {
            file << ",";
        }
        file << "\n";
    }

    file << "],\n";
    file << "\"metadata\": {\n";
    file << "    \" exporter\": \"Nova Timeline Exporter\",\n";
    file << "    \"version\": \"1.0\"\n";
    file << "  }\n";
    file << "}\n";

    file.close();
    return file.good();
}

}  // namespace cuda::observability
