#include "cuda_alloc_injection.h"

#include <cuda_runtime.h>

#include <atomic>

// Provided by the linker for each --wrap=cudaMalloc that a library links against.
extern "C" cudaError_t __real_cudaMalloc(void** ptr, size_t size);

namespace {

    /// -1 means "forward everything"; otherwise the index at which to start failing.
    thread_local int g_fail_from = -1;
    thread_local int g_attempts = 0;

}  // namespace

extern "C" cudaError_t __wrap_cudaMalloc(void** ptr, size_t size) {
    const int attempt = g_attempts++;
    if (g_fail_from >= 0 && attempt >= g_fail_from) {
        if (ptr != nullptr) {
            *ptr = nullptr;
        }
        return cudaErrorMemoryAllocation;
    }
    return __real_cudaMalloc(ptr, size);
}

namespace cuda_test {

    void fail_allocation_after(int n) {
        g_fail_from = n < 0 ? 0 : n;
        g_attempts = 0;
    }

    void stop_failing_allocations() {
        g_fail_from = -1;
        g_attempts = 0;
    }

    int allocation_attempts() {
        return g_attempts;
    }

}  // namespace cuda_test
