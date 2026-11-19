#pragma once

#include <functional>
#include <vector>
#include <string>

namespace cuda::testing {

    struct IntegrationTestCase {
        std::string name;
        std::string description;
        std::function<bool()> test_fn;
        bool enabled = true;
    };

    class IntegrationTestRunner {
    public:
        IntegrationTestRunner() = default;

        void add_test(const IntegrationTestCase& test);
        void remove_test(const std::string& name);

        struct TestResult {
            std::string name;
            bool passed;
            std::string error_message;
            double execution_time_ms;
        };

        std::vector<TestResult> run_all();
        std::vector<TestResult> run_enabled();

        size_t total_tests() const { return tests_.size(); }
        size_t enabled_tests() const;

    private:
        std::vector<IntegrationTestCase> tests_;
    };

    struct E2ERobustnessProfileResult {
        bool robust;
        size_t tests_passed;
        size_t tests_failed;
        double total_time_ms;
        std::vector<std::string> failed_tests;
    };

    E2ERobustnessProfileResult run_e2e_robustness_with_profiling();

    struct MemorySafetyValidationResult {
        bool all_safe;
        size_t algorithms_validated;
        size_t issues_found;
        std::vector<std::string> unsafe_algorithms;
    };

    MemorySafetyValidationResult validate_all_algorithm_memory_safety();

}  // namespace cuda::testing
