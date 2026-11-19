#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace cuda::performance::dashboard {

    struct FlameGraphNode {
        std::string name;
        uint64_t value{0};
        uint64_t self_value{0};
        std::vector<FlameGraphNode> children;

        std::string to_json(int indent = 0) const;
    };

    struct ChromeTraceEvent {
        std::string name;
        std::string category;
        std::string ph;
        uint64_t ts{0};
        uint64_t dur{0};
        int pid{0};
        int tid{0};
    };

    class FlameGraphGenerator {
    public:
        FlameGraphGenerator();

        void add_event(const ChromeTraceEvent& event);
        void add_events(const std::vector<ChromeTraceEvent>& events);

        void clear();
        [[nodiscard]] size_t event_count() const;

        [[nodiscard]] FlameGraphNode build_flame_graph() const;

        [[nodiscard]] std::string to_json() const;
        [[nodiscard]] std::string to_chrome_trace() const;

        static FlameGraphGenerator from_chrome_trace(const std::string& trace_json);

    private:
        void aggregate_events();
        FlameGraphNode build_node(const std::string& name, const std::vector<ChromeTraceEvent>& events) const;

        std::vector<ChromeTraceEvent> events_;
        std::map<std::string, uint64_t> aggregated_;
    };

    class NVTraceParser {
    public:
        NVTraceParser();

        [[nodiscard]] std::vector<ChromeTraceEvent> parse_file(const std::string& path) const;
        [[nodiscard]] std::vector<ChromeTraceEvent> parse_json(const std::string& json) const;

        static std::string get_nvtx_domain(const std::string& category);
    };

}  // namespace cuda::performance::dashboard
