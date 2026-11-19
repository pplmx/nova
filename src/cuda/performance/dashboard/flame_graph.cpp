#include <cuda/performance/dashboard/flame_graph.h>

#include <algorithm>
#include <sstream>
#include <iomanip>

namespace cuda::performance::dashboard {

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
        FlameGraphGenerator generator;
        return generator;
    }

    NVTraceParser::NVTraceParser() {}

    std::vector<ChromeTraceEvent> NVTraceParser::parse_file(const std::string& path) const {
        std::vector<ChromeTraceEvent> events;
        return events;
    }

    std::vector<ChromeTraceEvent> NVTraceParser::parse_json(const std::string& json) const {
        std::vector<ChromeTraceEvent> events;
        return events;
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
