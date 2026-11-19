#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <functional>
#include <fstream>
#include <mutex>

namespace cuda::performance {

    struct AutotuneConfig {
        int device_id = 0;
        std::string kernel_name;
        std::vector<int> block_sizes {64, 128, 256, 512, 1024};
        std::vector<int> grid_sizes {1, 2, 4, 8, 16, 32, 64, 128, 256};
        int warmup_iterations = 10;
        int measure_iterations = 100;
        float max_time_ms = 1000.0f;
        std::string config_path = "autotune_config.json";
    };

    struct AutotuneResult {
        int optimal_block_size = 256;
        int optimal_grid_size = 64;
        float best_time_ms = 0.0f;
        float speedup_vs_default = 1.0f;
    };

    class Autotuner {
    public:
        explicit Autotuner(const AutotuneConfig& config);
        ~Autotuner();

        template <typename Func>
        AutotuneResult tune(Func kernel_func) {
            std::lock_guard<std::mutex> lock(mutex_);
            return tune_internal(kernel_func);
        }

        void set_block_sizes(const std::vector<int>& sizes);
        void set_grid_sizes(const std::vector<int>& sizes);
        void set_warmup_iterations(int iterations);
        void set_measure_iterations(int iterations);

        std::optional<AutotuneResult> load_cached_result(const std::string& kernel_name);
        void save_result(const std::string& kernel_name, const AutotuneResult& result);
        void save_all_results();

        static std::string get_default_config_path();

    private:
        template <typename Func>
        AutotuneResult tune_internal(Func kernel_func) {
            AutotuneResult best_result;

            // A search needs both dimensions to have candidates and at least one
            // measurement per candidate. Without these guards the loop body never ran
            // and the FLT_MAX sentinel below was returned as the measured time, or
            // measure_iterations == 0 produced 0/0 = NaN, which compares false against
            // the sentinel and left it in place too. Both reported 3.40282e+38 ms as
            // the best time — a value no timing can produce.
            if (config_.block_sizes.empty() || config_.grid_sizes.empty()) {
                throw std::invalid_argument(
                    "Autotuner: the search space needs at least one block size and one "
                    "grid size");
            }
            if (config_.measure_iterations <= 0) {
                throw std::invalid_argument("Autotuner: measure_iterations must be positive, got " + std::to_string(config_.measure_iterations));
            }

            best_result.best_time_ms = std::numeric_limits<float>::max();

            for (int block_size : config_.block_sizes) {
                for (int grid_size : config_.grid_sizes) {
                    float total_time = 0.0f;

                    for (int i = 0; i < config_.warmup_iterations; ++i) {
                        kernel_func(block_size, grid_size);
                    }

                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);

                    for (int i = 0; i < config_.measure_iterations; ++i) {
                        cudaEventRecord(start);
                        kernel_func(block_size, grid_size);
                        cudaEventRecord(stop);
                        cudaEventSynchronize(stop);

                        float elapsed;
                        cudaEventElapsedTime(&elapsed, start, stop);
                        total_time += elapsed;
                    }

                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);

                    float avg_time = total_time / config_.measure_iterations;

                    if (avg_time < best_result.best_time_ms) {
                        best_result.best_time_ms = avg_time;
                        best_result.optimal_block_size = block_size;
                        best_result.optimal_grid_size = grid_size;
                    }
                }
            }

            return best_result;
        }

        std::string get_cache_key(const std::string& kernel_name);
        void load_cache();
        void persist_cache();

        AutotuneConfig config_;
        std::unordered_map<std::string, AutotuneResult> cached_results_;
        std::mutex mutex_;
        bool cache_loaded_ = false;
    };

    template <typename T>
    struct TunedParameters {
        T block_size;
        T grid_size;
        T shared_mem_bytes;
    };

    class AutotuneRegistry {
    public:
        static AutotuneRegistry& instance();

        void register_result(const std::string& kernel_name, int device_id, const AutotuneResult& result);
        std::optional<AutotuneResult> get_result(const std::string& kernel_name, int device_id);

        void save_to_file(const std::string& filepath);
        void load_from_file(const std::string& filepath);

        void clear();

    private:
        std::string make_key(const std::string& kernel_name, int device_id);

        std::unordered_map<std::string, AutotuneResult> results_;
        std::mutex mutex_;
    };

#define AUTOTUNE(kernel_name, device_id, func) cuda::performance::Autotuner::instance().tune(kernel_name, device_id, func)

}  // namespace cuda::performance
