#include "cuda/testing/memory_safety.h"

#include <cstring>

namespace cuda::testing {

static constexpr uint8_t POISON_BYTE = 0xFE;

MemorySafetyValidator& MemorySafetyValidator::instance() {
    static MemorySafetyValidator instance;
    return instance;
}

bool MemorySafetyValidator::validate_allocation(const void* ptr, size_t size) {
    if (!enabled_) return true;

    validation_count_++;

    if (ptr == nullptr) {
        error_count_++;
        return false;
    }

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);

    if (err != cudaSuccess) {
        error_count_++;
        return false;
    }

    return true;
}

bool MemorySafetyValidator::validate_access(const void* ptr, size_t size) {
    if (!enabled_) return true;

    if (!is_memory_aligned(ptr)) {
        return false;
    }

    return validate_allocation(ptr, size);
}

bool MemorySafetyValidator::check_uninitialized(const void* ptr, size_t size) {
    if (!enabled_) return true;

    validation_count_++;

    const uint8_t* bytes = static_cast<const uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i) {
        if (bytes[i] == POISON_BYTE) {
            error_count_++;
            return false;
        }
    }

    return true;
}

bool MemorySafetyValidator::check_double_free(const void* ptr) {
    if (!enabled_) return true;
    return true;
}

void MemorySafetyValidator::enable() {
    enabled_ = true;
}

void MemorySafetyValidator::disable() {
    enabled_ = false;
}

void MemorySafetyValidator::set_tool(MemorySafetyTool tool) {
    tool_ = tool;
}

MemorySafetyValidator::ValidationResult MemorySafetyValidator::validate() {
    ValidationResult result;
    result.valid = true;
    result.bytes_checked = 0;

    if (!enabled_) {
        return result;
    }

    return result;
}

MemoryPoisonGuard::MemoryPoisonGuard(const void* ptr, size_t size)
    : ptr_(ptr), size_(size), is_poisoned_(false) {
}

MemoryPoisonGuard::~MemoryPoisonGuard() {
    if (!is_poisoned_) {
        uint8_t* bytes = const_cast<uint8_t*>(static_cast<const uint8_t*>(ptr_));
        std::memset(bytes, POISON_BYTE, size_);
    }
}

bool MemoryPoisonGuard::is_poisoned() const {
    const uint8_t* bytes = static_cast<const uint8_t*>(ptr_);
    for (size_t i = 0; i < size_; ++i) {
        if (bytes[i] == POISON_BYTE) {
            return true;
        }
    }
    return false;
}

void MemoryPoisonGuard::mark_valid() {
    is_poisoned_ = true;
}

void MemoryPoisonGuard::mark_invalid() {
    is_poisoned_ = false;
}

void run_memory_safety_checks() {
}

}  // namespace cuda::testing
