#include <cuda/performance/dashboard/flame_graph.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace cuda::performance::dashboard {

    namespace {

        /// Pull the string value of `"key"` out of one JSON object fragment, or "" when absent.
        std::string json_string(const std::string& object, const std::string& key) {
            const std::string needle = "\"" + key + "\"";
            const auto at = object.find(needle);
            if (at == std::string::npos) {
                return {};
            }
            auto colon = object.find(':', at + needle.size());
            if (colon == std::string::npos) {
                return {};
            }
            auto open = object.find('"', colon);
            if (open == std::string::npos) {
                return {};
            }
            // Walk to the closing quote, honouring backslash escapes so an escaped quote inside a
            // kernel name does not terminate the value early.
            std::string value;
            for (size_t i = open + 1; i < object.size(); ++i) {
                if (object[i] == '\\' && i + 1 < object.size()) {
                    value.push_back(object[++i]);
                    continue;
                }
                if (object[i] == '"') {
                    return value;
                }
                value.push_back(object[i]);
            }
            return {};
        }

        /// Pull the numeric value of `"key"` out of one JSON object fragment, or fallback.
        uint64_t json_number(const std::string& object, const std::string& key, uint64_t fallback) {
            const std::string needle = "\"" + key + "\"";
            const auto at = object.find(needle);
            if (at == std::string::npos) {
                return fallback;
            }
            auto colon = object.find(':', at + needle.size());
            if (colon == std::string::npos) {
                return fallback;
            }
            size_t i = colon + 1;
            while (i < object.size() && (object[i] == ' ' || object[i] == '\t')) {
                ++i;
            }
            const size_t start = i;
            if (i < object.size() && object[i] == '-') {
                ++i;
            }
            while (i < object.size() && std::isdigit(static_cast<unsigned char>(object[i]))) {
                ++i;
            }
            if (i == start) {
                return fallback;
            }
            try {
                return static_cast<uint64_t>(std::stoll(object.substr(start, i - start)));
            } catch (const std::exception&) {
                return fallback;
            }
        }

        /// Every balanced {...} object in the document, in order.
        std::vector<std::string> json_objects(const std::string& json) {
            std::vector<std::string> objects;
            int depth = 0;
            size_t start = std::string::npos;
            bool in_string = false;
            for (size_t i = 0; i < json.size(); ++i) {
                const char c = json[i];
                if (in_string) {
                    if (c == '\\') {
                        ++i;
                        continue;
                    }
                    if (c == '"') {
                        in_string = false;
                    }
                    continue;
                }
                if (c == '"') {
                    in_string = true;
                } else if (c == '{') {
                    if (depth == 0) {
                        start = i;
                    }
                    ++depth;
                } else if (c == '}') {
                    if (depth > 0 && --depth == 0 && start != std::string::npos) {
                        objects.push_back(json.substr(start, i - start + 1));
                        start = std::string::npos;
                    }
                }
            }
            return objects;
        }

        std::vector<ChromeTraceEvent> parse_events(const std::string& json) {
            std::vector<ChromeTraceEvent> events;
            for (const auto& object : json_objects(json)) {
                ChromeTraceEvent e;
                e.name = json_string(object, "name");
                // An object with no name is not an event; the array may carry metadata records.
                if (e.name.empty()) {
                    continue;
                }
                e.category = json_string(object, "cat");
                e.ph = json_string(object, "ph");
                e.ts = json_number(object, "ts", 0);
                e.dur = json_number(object, "dur", 0);
                e.pid = static_cast<int>(json_number(object, "pid", 0));
                e.tid = static_cast<int>(json_number(object, "tid", 0));
                events.push_back(e);
            }
            return events;
        }

    }  // namespace

    std::string FlameGraphNode::to_json(int indent) const {
        std::ostringstream oss;
        std::string pad(indent, ' ');

        oss << "{\n";
        oss << pad << "  \"name\": \"" << name << "\",\n";
        oss << pad << "  \"value\": " << value << ",\n";
        oss << pad << "  \"self\": " << self_value << ",\n";

        if (!children.empty()) {
            oss << pad << "  \"children\": [\n";
            for (size_t i = 0; i < children.size(); ++i) {
                if (i > 0) {
                    oss << ",\n";
                }
                oss << pad << "    " << children[i].to_json(indent + 4);
            }
            oss << "\n" << pad << "  ]\n";
        } else {
            oss << pad << "  \"children\": []\n";
        }

        oss << pad << "}";
        return oss.str();
    }

    FlameGraphGenerator::FlameGraphGenerator() {}

    void FlameGraphGenerator::add_event(const ChromeTraceEvent& event) {
        events_.push_back(event);
    }

    void FlameGraphGenerator::add_events(const std::vector<ChromeTraceEvent>& events) {
        events_.insert(events_.end(), events.begin(), events.end());
    }

    void FlameGraphGenerator::clear() {
        events_.clear();
        aggregated_.clear();
    }

    size_t FlameGraphGenerator::event_count() const {
        return events_.size();
    }

    void FlameGraphGenerator::aggregate_events() {
        aggregated_.clear();
        for (const auto& event : events_) {
            aggregated_[event.name] += event.dur;
        }
    }

    FlameGraphNode FlameGraphGenerator::build_flame_graph() const {
        FlameGraphNode root;
        root.name = "root";
        root.value = 0;

        std::map<std::string, std::vector<ChromeTraceEvent>> by_category;
        for (const auto& event : events_) {
            if (event.ph == "X") {
                by_category[event.category].push_back(event);
                root.value += event.dur;
            }
        }

        for (const auto& [category, category_events] : by_category) {
            FlameGraphNode category_node;
            category_node.name = category;
            category_node.value = 0;

            std::map<std::string, uint64_t> by_name;
            for (const auto& event : category_events) {
                by_name[event.name] += event.dur;
                category_node.value += event.dur;
            }

            for (const auto& [name, duration] : by_name) {
                FlameGraphNode name_node;
                name_node.name = name;
                name_node.value = duration;
                name_node.self_value = duration;
                category_node.children.push_back(name_node);
            }

            root.children.push_back(category_node);
        }

        return root;
    }

    std::string FlameGraphGenerator::to_json() const {
        auto flame_graph = build_flame_graph();
        return flame_graph.to_json();
    }

    std::string FlameGraphGenerator::to_chrome_trace() const {
        std::ostringstream oss;
        oss << "[\n";

        for (size_t i = 0; i < events_.size(); ++i) {
            const auto& e = events_[i];
            if (i > 0) {
                oss << ",\n";
            }
            oss << "    {\n";
            oss << "      \"name\": \"" << e.name << "\",\n";
            oss << "      \"cat\": \"" << e.category << "\",\n";
            oss << "      \"ph\": \"" << e.ph << "\",\n";
            oss << "      \"ts\": " << e.ts << ",\n";
            oss << "      \"dur\": " << e.dur << ",\n";
            oss << "      \"pid\": " << e.pid << ",\n";
            oss << "      \"tid\": " << e.tid << "\n";
            oss << "    }";
        }

        oss << "\n  ]\n";
        return oss.str();
    }

    FlameGraphGenerator FlameGraphGenerator::from_chrome_trace(const std::string& trace_json) {
        // This used to return a default-constructed generator and drop the argument, so the inverse
        // of to_chrome_trace() silently produced an empty graph. Measured: two events written by
        // to_chrome_trace() (277 bytes of JSON) came back as event_count() == 0, which is
        // indistinguishable from a trace that legitimately held nothing.
        FlameGraphGenerator generator;
        generator.add_events(parse_events(trace_json));
        return generator;
    }

    NVTraceParser::NVTraceParser() {}

    std::vector<ChromeTraceEvent> NVTraceParser::parse_file(const std::string& path) const {
        // A missing or unreadable file yields an empty vector, which is the same answer this always
        // gave -- but it is now an honest empty rather than a stub that never looked at the file.
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return {};
        }
        std::ostringstream buffer;
        buffer << in.rdbuf();
        return parse_events(buffer.str());
    }

    std::vector<ChromeTraceEvent> NVTraceParser::parse_json(const std::string& json) const {
        return parse_events(json);
    }

    std::string NVTraceParser::get_nvtx_domain(const std::string& category) {
        if (category.find("memory") != std::string::npos) {
            return "nova.memory";
        }
        if (category.find("device") != std::string::npos) {
            return "nova.device";
        }
        if (category.find("algo") != std::string::npos) {
            return "nova.algo";
        }
        if (category.find("nvblox") != std::string::npos) {
            return "nova.performance.nvblox";
        }
        if (category.find("fusion") != std::string::npos) {
            return "nova.performance.fusion";
        }
        if (category.find("bandwidth") != std::string::npos) {
            return "nova.performance.bandwidth";
        }
        if (category.find("performance") != std::string::npos) {
            return "nova.performance";
        }
        return "nova.api";
    }

}  // namespace cuda::performance::dashboard
