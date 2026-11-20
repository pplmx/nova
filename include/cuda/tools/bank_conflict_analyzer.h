#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>

namespace cuda::tools {

    struct BankConflictConfig {
        int shared_mem_size = 0;
        int block_size = 256;
        int num_threads = 0;
        bool check_padding = true;
    };

    struct BankConflictResult {
        int potential_conflicts = 0;
        int suggested_padding = 0;
        std::string analysis;
    };

    BankConflictResult analyze_bank_conflicts(const void* shared_mem_ptr, size_t data_size, const BankConflictConfig& config);

    int detect_bank_conflicts(const void* shared_mem_ptr, int num_elements, int stride);

    class SharedMemoryAnalyzer {
    public:
        static SharedMemoryAnalyzer& instance();

        void set_config(const BankConflictConfig& config);
        BankConflictConfig get_config() const;

        BankConflictResult analyze(const void* ptr, size_t size);

        int suggest_padding(int data_type_size, int num_elements);

        /// Turn padding suggestions on or off.
        ///
        /// When disabled, suggest_padding() returns 0 rather than computing a figure the
        /// caller asked not to receive.
        void enable_padding_hints(bool enable);

        [[nodiscard]] bool is_padding_hints_enabled() const { return padding_hints_enabled_; }

    private:
        SharedMemoryAnalyzer() = default;

        BankConflictConfig config_;
        bool padding_hints_enabled_ = true;
    };

}  // namespace cuda::tools
