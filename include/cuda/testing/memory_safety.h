#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <set>
#include <vector>

namespace cuda::testing {

    bool is_memory_aligned(const void* ptr);

    enum class MemorySafetyTool {
        ComputeSanitizer,
        CudaMallocValidate,
        PoisonPattern,
    };

    class MemorySafetyValidator {
    public:
        MemorySafetyValidator() = default;

        static MemorySafetyValidator& instance();

        /**
         * @brief Check that @p ptr is a device pointer the runtime recognises
         *
         * @warning @p size is IGNORED, and cannot be otherwise. cudaPointerGetAttributes() returns
         * type, device and the device/host aliases; it carries no length, and CUDA exposes no way
         * to ask how large an allocation is. Measured: a 1 KiB cudaMalloc validated successfully
         * when presented as 1 GiB and when presented as 0 bytes, because only the pointer is
         * examined.
         *
         * So this answers "is this a live device pointer", not "is this range in bounds". An
         * out-of-bounds access on a valid pointer passes. Use compute-sanitizer for bounds; the
         * parameter is kept because removing it is an interface change and callers pass a real
         * size, but do not read a true result as a bounds check.
         */
        bool validate_allocation(const void* ptr, size_t size);
        bool validate_access(const void* ptr, size_t size);
        bool check_uninitialized(const void* ptr, size_t size);

        /**
         * @brief Report whether this address has already been presented to this validator
         *
         * Records @p ptr and returns false the second and every later time the same address is
         * seen. It cannot actually tell a double free from a reuse: CUDA exposes no "was this
         * released" query, and an address handed back by a later cudaMalloc looks identical. Treat
         * a false result as worth investigating, not as proof.
         *
         * Was `return true;` unconditionally — it never reported anything.
         */
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
            seen_pointers_.clear();
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
        std::set<const void*> seen_pointers_;
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
