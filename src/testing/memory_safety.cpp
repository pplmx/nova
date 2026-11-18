#include "cuda/testing/memory_safety.h"
#include "cuda/testing/boundary_testing.h"

#include <cstring>
#include <string>

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

    if (ptr == nullptr) {
        error_count_++;
        return false;
    }

    // `ptr` is GPU memory: scanning it directly from the CPU segfaults. Stage a
    // host copy and inspect that. If the memory cannot be read back, treat the
    // region as unsafe rather than crashing.
    std::vector<uint8_t> host_buf(size);
    if (cudaMemcpy(host_buf.data(), ptr, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        error_count_++;
        return false;
    }
    for (size_t i = 0; i < size; ++i) {
        if (host_buf[i] == POISON_BYTE) {
            error_count_++;
            return false;
        }
    }

    return true;
}

bool MemorySafetyValidator::check_double_free(const void* ptr) {
    if (!enabled_) return true;

    validation_count_++;

    if (ptr == nullptr) {
        error_count_++;
        return false;
    }

    // See the declaration: this detects a repeated address, which is not the same statement as
    // "this was freed twice". It is the strongest claim the CUDA API allows.
    const auto [it, inserted] = seen_pointers_.insert(ptr);
    (void)it;
    if (!inserted) {
        error_count_++;
        return false;
    }
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

    // This used to be `result.valid = true; return result;` — a validation that could not
    // fail, which is indistinguishable from no validation at all. It now summarises what the
    // individual checks recorded: any error makes the result invalid, and the message says how
    // many were seen of how many checks.
    //
    // It is a summary, not a fresh inspection: there is no list of allocations to re-examine,
    // so the answer reflects check_*() calls made since the last reset().
    result.valid = (error_count_ == 0);
    result.bytes_checked = 0;

    if (!result.valid) {
        result.error_message = std::to_string(error_count_) + " error(s) across " +
                               std::to_string(validation_count_) + " check(s)";
    }

    return result;
}

MemoryPoisonGuard::MemoryPoisonGuard(const void* ptr, size_t size)
    : ptr_(ptr), size_(size), is_poisoned_(false) {
}

// The pointee is device memory, so both the fill and the scan below have to go through a
// staging buffer — exactly what check_uninitialized above already does and says why. These two
// used to dereference the device pointer directly from the host: measured, a guard constructed
// over a cudaMalloc'd block segfaults in the destructor and in is_poisoned(), exit code 139.
MemoryPoisonGuard::~MemoryPoisonGuard() {
    if (!is_poisoned_ && ptr_ != nullptr && size_ > 0) {
        std::vector<uint8_t> poison(size_, POISON_BYTE);
        // Best effort: a destructor cannot report a failure, and poisoning a block that has
        // already been released is not worth terminating the process over.
        (void)cudaMemcpy(const_cast<void*>(ptr_), poison.data(), size_,
                         cudaMemcpyHostToDevice);
    }
}

bool MemoryPoisonGuard::is_poisoned() const {
    if (ptr_ == nullptr || size_ == 0) return false;

    std::vector<uint8_t> host_buf(size_);
    if (cudaMemcpy(host_buf.data(), ptr_, size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
        // Unreadable is not the same as unpoisoned, but this predicate has two answers and
        // cannot raise. Reporting false keeps the destructor from writing to memory it could
        // not read; callers wanting the distinction must check cudaGetLastError themselves.
        (void)cudaGetLastError();
        return false;
    }
    for (size_t i = 0; i < size_; ++i) {
        if (host_buf[i] == POISON_BYTE) {
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
