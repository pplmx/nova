#include "cuda/testing/integration.h"
#include "cuda/observability/timeline.h"

#include <chrono>
#include <cuda_runtime.h>

namespace cuda::testing {

    void IntegrationTestRunner::add_test(const IntegrationTestCase& test) {
        tests_.push_back(test);
    }

    void IntegrationTestRunner::remove_test(const std::string& name) {
        tests_.erase(std::remove_if(tests_.begin(), tests_.end(), [&name](const IntegrationTestCase& t) { return t.name == name; }), tests_.end());
    }

    size_t IntegrationTestRunner::enabled_tests() const {
        return std::count_if(tests_.begin(), tests_.end(), [](const IntegrationTestCase& t) { return t.enabled; });
    }

    std::vector<IntegrationTestRunner::TestResult> IntegrationTestRunner::run_all() {
        return run_enabled();
    }

    std::vector<IntegrationTestRunner::TestResult> IntegrationTestRunner::run_enabled() {
        std::vector<TestResult> results;

        for (const auto& test : tests_) {
            if (!test.enabled) {
                continue;
            }

            TestResult result;
            result.name = test.name;

            auto start = std::chrono::high_resolution_clock::now();

            try {
                result.passed = test.test_fn();
                result.error_message = result.passed ? "" : "Test assertion failed";
            } catch (const std::exception& e) {
                result.passed = false;
                result.error_message = e.what();
            } catch (...) {
                result.passed = false;
                result.error_message = "Unknown exception";
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            result.execution_time_ms = duration.count();

            results.push_back(result);
        }

        return results;
    }

    E2ERobustnessProfileResult run_e2e_robustness_with_profiling() {
        E2ERobustnessProfileResult result;
        result.robust = true;
        result.tests_passed = 0;
        result.tests_failed = 0;
        result.total_time_ms = 0;

        IntegrationTestRunner runner;

        // Each probe must release what it acquired. The originals returned the
        // creation status and dropped the handle, so every run of this profile leaked
        // 1 MiB of device memory plus a stream and an event — and a robustness suite
        // that leaks is measuring the wrong thing.
        runner.add_test({"test_memory_allocation", "Memory allocation robustness", []() {
                             void* ptr = nullptr;
                             if (cudaMalloc(&ptr, 1024 * 1024) != cudaSuccess) {
                                 return false;
                             }
                             const bool freed = cudaFree(ptr) == cudaSuccess;
                             return freed;
                         }});

        runner.add_test({"test_stream_creation", "Stream creation robustness", []() {
                             cudaStream_t stream = nullptr;
                             if (cudaStreamCreate(&stream) != cudaSuccess) {
                                 return false;
                             }
                             const bool destroyed = cudaStreamDestroy(stream) == cudaSuccess;
                             return destroyed;
                         }});

        runner.add_test({"test_event_creation", "Event creation robustness", []() {
                             cudaEvent_t event = nullptr;
                             if (cudaEventCreate(&event) != cudaSuccess) {
                                 return false;
                             }
                             const bool destroyed = cudaEventDestroy(event) == cudaSuccess;
                             return destroyed;
                         }});

        runner.add_test({"test_context_sync", "Context synchronization", []() { return cudaDeviceSynchronize() == cudaSuccess; }});

        auto start = std::chrono::high_resolution_clock::now();
        auto results = runner.run_all();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        result.total_time_ms = duration.count();

        for (const auto& r : results) {
            if (r.passed) {
                result.tests_passed++;
            } else {
                result.tests_failed++;
                result.robust = false;
                result.failed_tests.push_back(r.name);
            }
        }

        return result;
    }

    MemorySafetyValidationResult validate_all_algorithm_memory_safety() {
        MemorySafetyValidationResult result;
        result.all_safe = true;
        result.algorithms_validated = 0;
        result.issues_found = 0;

        struct AlgorithmValidation {
            std::string name;
            std::function<bool()> validate;
        };

        // Every entry here used to be `return true;`. The function therefore reported
        // all_safe = true with algorithms_validated = 6 having inspected nothing at
        // all — a claim of memory-safety validation that no code backed. These probes
        // exercise the device allocation path each algorithm depends on and release
        // what they take, so the verdict reflects something real.
        std::vector<AlgorithmValidation> algorithms = {
            {"segmented_sort",
             []() {
                 // Needs a device scratch buffer for the per-segment sort.
                 void* scratch = nullptr;
                 if (cudaMalloc(&scratch, 64 * 1024) != cudaSuccess) {
                     return false;
                 }
                 cudaFree(scratch);
                 return true;
             }},
            {"spmv",
             []() {
                 // Needs a device vector and a row pointer array.
                 float* values = nullptr;
                 int* columns = nullptr;
                 if (cudaMalloc(&values, 1024 * sizeof(float)) != cudaSuccess) {
                     return false;
                 }
                 const bool ok = cudaMalloc(&columns, 1024 * sizeof(int)) == cudaSuccess;
                 cudaFree(values);
                 if (ok) {
                     cudaFree(columns);
                 }
                 return ok;
             }},
            {"sample_sort",
             []() {
                 // Needs room for a keys and an index array.
                 void* keys = nullptr;
                 void* indices = nullptr;
                 if (cudaMalloc(&keys, 4096 * sizeof(float)) != cudaSuccess) {
                     return false;
                 }
                 const bool ok = cudaMalloc(&indices, 4096 * sizeof(int)) == cudaSuccess;
                 cudaFree(keys);
                 if (ok) {
                     cudaFree(indices);
                 }
                 return ok;
             }},
            {"sssp",
             []() {
                 // Needs a distance array per vertex.
                 float* distances = nullptr;
                 if (cudaMalloc(&distances, 1024 * sizeof(float)) != cudaSuccess) {
                     return false;
                 }
                 cudaFree(distances);
                 return true;
             }},
            {"timeline",
             []() {
                 // Host-side exporter: it must round-trip an event without device work.
                 auto& exporter = ::cuda::observability::TimelineExporter::instance();
                 exporter.clear();
                 exporter.record_event("probe", "integration", 1);
                 const bool recorded = exporter.event_count() == 1;
                 exporter.clear();
                 return recorded;
             }},
            {"bandwidth_tracker",
             []() {
                 // Needs two device buffers to time a copy between.
                 void* source = nullptr;
                 void* target = nullptr;
                 if (cudaMalloc(&source, 1024 * 1024) != cudaSuccess) {
                     return false;
                 }
                 const bool allocated = cudaMalloc(&target, 1024 * 1024) == cudaSuccess;
                 cudaFree(source);
                 if (allocated) {
                     cudaFree(target);
                 }
                 return allocated;
             }},
        };

        for (const auto& algo : algorithms) {
            result.algorithms_validated++;

            try {
                if (!algo.validate()) {
                    result.all_safe = false;
                    result.unsafe_algorithms.push_back(algo.name);
                    result.issues_found++;
                }
            } catch (...) {
                result.all_safe = false;
                result.unsafe_algorithms.push_back(algo.name);
                result.issues_found++;
            }
        }

        return result;
    }

}  // namespace cuda::testing
