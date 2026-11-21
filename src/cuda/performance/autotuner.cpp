#include "cuda/performance/autotuner.h"

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <chrono>
#include <cmath>
#include <optional>
#include <type_traits>

namespace cuda::performance {

    namespace {

        /// Parse one scalar out of a cache file field.
        ///
        /// Returns nullopt rather than throwing, and rejects trailing junk: the cache is
        /// written by a previous run and read back as untrusted input, so a corrupted field
        /// must cost its own value, not the constructor that happens to be doing the reading.
        template <typename T>
        std::optional<T> parse_scalar(const std::string& text) {
            if (text.empty()) {
                return std::nullopt;
            }
            try {
                size_t consumed = 0;
                if constexpr (std::is_integral_v<T>) {
                    const long long value = std::stoll(text, &consumed);
                    if (value < static_cast<long long>(std::numeric_limits<T>::min()) || value > static_cast<long long>(std::numeric_limits<T>::max())) {
                        return std::nullopt;
                    }
                    if (consumed != text.size()) {
                        return std::nullopt;
                    }
                    return static_cast<T>(value);
                } else {
                    const long double value = std::stold(text, &consumed);
                    if (consumed != text.size()) {
                        return std::nullopt;
                    }
                    return static_cast<T>(value);
                }
            } catch (const std::exception&) {
                return std::nullopt;
            }
        }

        /// Read the four fields of an AutotuneResult out of "k=v,k=v,...".
        ///
        /// Shared by Autotuner::load_cache() and AutotuneRegistry::load_from_file(), which had
        /// identical copies of the parsing loop and identical defects in it. A field that
        /// cannot be parsed keeps its default; a non-finite timing is not a measurement.
        AutotuneResult parse_result_fields(const std::string& value_str) {
            AutotuneResult result;
            std::istringstream vss(value_str);
            std::string pair;

            while (std::getline(vss, pair, ',')) {
                const size_t eq = pair.find('=');
                if (eq == std::string::npos) {
                    continue;
                }

                const std::string k = pair.substr(0, eq);
                const std::string v = pair.substr(eq + 1);

                if (k == "block_size") {
                    if (const auto parsed = parse_scalar<int>(v)) {
                        result.optimal_block_size = *parsed;
                    }
                } else if (k == "grid_size") {
                    if (const auto parsed = parse_scalar<int>(v)) {
                        result.optimal_grid_size = *parsed;
                    }
                } else if (k == "time_ms") {
                    if (const auto parsed = parse_scalar<float>(v); parsed && std::isfinite(*parsed)) {
                        result.best_time_ms = *parsed;
                    }
                } else if (k == "speedup") {
                    if (const auto parsed = parse_scalar<float>(v); parsed && std::isfinite(*parsed)) {
                        result.speedup_vs_default = *parsed;
                    }
                }
            }
            return result;
        }

    }  // namespace

    Autotuner::Autotuner(const AutotuneConfig& config)
        : config_(config) {
        load_cache();
    }

    Autotuner::~Autotuner() {
        save_all_results();
    }

    void Autotuner::set_block_sizes(const std::vector<int>& sizes) {
        config_.block_sizes = sizes;
    }

    void Autotuner::set_grid_sizes(const std::vector<int>& sizes) {
        config_.grid_sizes = sizes;
    }

    void Autotuner::set_warmup_iterations(int iterations) {
        config_.warmup_iterations = iterations;
    }

    void Autotuner::set_measure_iterations(int iterations) {
        config_.measure_iterations = iterations;
    }

    std::optional<AutotuneResult> Autotuner::load_cached_result(const std::string& kernel_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cached_results_.find(get_cache_key(kernel_name));
        if (it != cached_results_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void Autotuner::save_result(const std::string& kernel_name, const AutotuneResult& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        cached_results_[get_cache_key(kernel_name)] = result;
    }

    void Autotuner::save_all_results() {
        persist_cache();
    }

    std::string Autotuner::get_default_config_path() {
        return "autotune_config.json";
    }

    std::string Autotuner::get_cache_key(const std::string& kernel_name) {
        return kernel_name + "_d" + std::to_string(config_.device_id);
    }

    void Autotuner::load_cache() {
        if (cache_loaded_) {
            return;
        }
        cache_loaded_ = true;

        std::ifstream file(config_.config_path);
        if (!file.is_open()) {
            return;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();

        std::string line;
        while (std::getline(buffer, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }

            size_t colon = line.find(':');
            if (colon == std::string::npos) {
                continue;
            }

            std::string key = line.substr(0, colon);
            // A cache file is untrusted input: written by a previous run, possibly truncated by
            // a crash or edited by hand. It used to be parsed with std::stoi/std::stof directly,
            // so one damaged field threw out of the CONSTRUCTOR and made the autotuner
            // unusable. Measured:
            //
            //     block_size=abc                       -> std::invalid_argument: stoi
            //     block_size=99999999999999999999      -> std::out_of_range: stoi
            //     block_size=                           -> std::invalid_argument: stoi
            //     time_ms=nan                           -> accepted, timing = nan
            //
            // parse_result_fields() skips a field it cannot parse, so one bad line costs one
            // cached result rather than the whole cache.
            cached_results_[key] = parse_result_fields(line.substr(colon + 1));
        }
    }

    void Autotuner::persist_cache() {
        std::ofstream file(config_.config_path);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not save autotune config to " << config_.config_path << std::endl;
            return;
        }

        file << "# Nova Autotune Configuration\n";
        file << "# Auto-generated - do not edit manually\n\n";

        for (const auto& [key, result] : cached_results_) {
            file << key << ":" << "block_size=" << result.optimal_block_size << "," << "grid_size=" << result.optimal_grid_size << "," << "time_ms=" << result.best_time_ms << ","
                 << "speedup=" << result.speedup_vs_default << "\n";
        }
    }

    AutotuneRegistry& AutotuneRegistry::instance() {
        static AutotuneRegistry instance;
        return instance;
    }

    void AutotuneRegistry::register_result(const std::string& kernel_name, int device_id, const AutotuneResult& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        results_[make_key(kernel_name, device_id)] = result;
    }

    std::optional<AutotuneResult> AutotuneRegistry::get_result(const std::string& kernel_name, int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = results_.find(make_key(kernel_name, device_id));
        if (it != results_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void AutotuneRegistry::save_to_file(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not save autotune registry to " << filepath << std::endl;
            return;
        }

        file << "# Nova Autotune Registry\n";
        file << "# Auto-generated - do not edit manually\n\n";

        for (const auto& [key, result] : results_) {
            file << key << ":" << "block_size=" << result.optimal_block_size << "," << "grid_size=" << result.optimal_grid_size << "," << "time_ms=" << result.best_time_ms << ","
                 << "speedup=" << result.speedup_vs_default << "\n";
        }
    }

    void AutotuneRegistry::load_from_file(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ifstream file(filepath);
        if (!file.is_open()) {
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }

            size_t colon = line.find(':');
            if (colon == std::string::npos) {
                continue;
            }

            // Same untrusted-input problem as Autotuner::load_cache(), through a copy of the
            // same loop: a damaged field threw out of load_from_file() instead of being
            // skipped.
            results_[line.substr(0, colon)] = parse_result_fields(line.substr(colon + 1));
        }
    }

    void AutotuneRegistry::clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        results_.clear();
    }

    std::string AutotuneRegistry::make_key(const std::string& kernel_name, int device_id) {
        return kernel_name + "_d" + std::to_string(device_id);
    }

}  // namespace cuda::performance
