#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace cuda::testing {

enum class MemorySafetyTool {
    ComputeSanitizer,
    CudaMallocValidate,
    PoisonPattern,
};

class MemorySafetyValidator {
public:
    MemorySafetyValidator() = default;

    static MemorySafetyValidator& instance();

    bool validate_allocation(const void* ptr, size_t size);
    bool validate_access(const void* ptr, size_t size);
    bool check_uninitialized(const void* ptr, size_t size);
    bool check_double_free(const void* ptr);

    void enable();
    void disable();
    bool is_enabled() const { return enabled_; }

    void set_tool(MemorySafetyTool tool);
    MemorySafetyTool tool() const { return tool_; }

    struct ValidationResult {
        bool valid;
        std::string error_message;
        size_t bytes_checked;
    };

    ValidationResult validate();

    void reset() {
        validation_count_ = 0;
        error_count_ = 0;
    }

    size_t validation_count() const { return validation_count_; }
    size_t error_count() const { return error_count_; }

private:
    MemorySafetyValidator(const MemorySafetyValidator&) = delete;
    MemorySafetyValidator& operator=(const MemorySafetyValidator&) = delete;

    bool enabled_ = true;
    MemorySafetyTool tool_ = MemorySafetyTool::PoisonPattern;
    size_t validation_count_ = 0;
    size_t error_count_ = 0;
};

class MemoryPoisonGuard {
public:
    explicit MemoryPoisonGuard(const void* ptr, size_t size);
    ~MemoryPoisonGuard();

    MemoryPoisonGuard(const MemoryPoisonGuard&) = delete;
    MemoryPoisonGuard& operator=(const MemoryPoisonGuard&) = delete;

    bool is_poisoned() const;
    void mark_valid();
    void mark_invalid();

private:
    const void* ptr_;
    size_t size_;
    bool is_poisoned_;
};

void run_memory_safety_checks();

}  // namespace cuda::testing
