#include "cuda_alloc_injection.h"

#include <cuda_runtime.h>

#include <atomic>
#include <new>
#include <cstdlib>

// Provided by the linker for each --wrap=cudaMalloc that a library links against.
extern "C" cudaError_t __real_cudaMalloc(void** ptr, size_t size);
extern "C" cudaError_t __real_cudaEventCreate(cudaEvent_t* event);
extern "C" cudaError_t __real_cudaEventDestroy(cudaEvent_t event);
extern "C" cudaError_t __real_cudaFree(void* ptr);

namespace {

    /// -1 means "forward everything"; otherwise the index at which to start failing.
    thread_local int g_fail_from = -1;
    thread_local int g_attempts = 0;

    /// The same arm, for cudaEventCreate. Kept separate from the allocation counter so that failing
    /// an event does not depend on -- or disturb -- the sequence of cudaMalloc calls.
    thread_local int g_event_fail_from = -1;
    thread_local int g_event_attempts = 0;

    /// Events created minus destroyed, per thread. Only counts what went through the wrapper, which
    /// is every call in the test binary.
    thread_local int g_live_events = 0;

    /// Allocations from the wrapper that have not been released.
    thread_local int g_live_allocations = 0;

}  // namespace

extern "C" cudaError_t __wrap_cudaMalloc(void** ptr, size_t size) {
    const int attempt = g_attempts++;
    if (g_fail_from >= 0 && attempt >= g_fail_from) {
        if (ptr != nullptr) {
            *ptr = nullptr;
        }
        return cudaErrorMemoryAllocation;
    }
    const cudaError_t status = __real_cudaMalloc(ptr, size);
    if (status == cudaSuccess) {
        ++g_live_allocations;
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaFree(void* ptr) {
    const cudaError_t status = __real_cudaFree(ptr);
    if (status == cudaSuccess && ptr != nullptr) {
        --g_live_allocations;
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaEventCreate(cudaEvent_t* event) {
    // Event failures have their own arm rather than sharing the allocation counter: a case that
    // wants "the second cudaMalloc fails" must not also depend on how many events some unrelated
    // call in between happens to create. See fail_event_after().
    const int attempt = g_event_attempts++;
    if (g_event_fail_from >= 0 && attempt >= g_event_fail_from) {
        if (event != nullptr) {
            *event = nullptr;
        }
        return cudaErrorInitializationError;
    }
    const cudaError_t status = __real_cudaEventCreate(event);
    if (status == cudaSuccess) {
        ++g_live_events;
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaEventDestroy(cudaEvent_t event) {
    const cudaError_t status = __real_cudaEventDestroy(event);
    if (status == cudaSuccess) {
        --g_live_events;
    }
    return status;
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

    void fail_event_after(int n) {
        g_event_fail_from = n < 0 ? 0 : n;
        g_event_attempts = 0;
    }

    void stop_failing_events() {
        g_event_fail_from = -1;
        g_event_attempts = 0;
    }

    int event_attempts() {
        return g_event_attempts;
    }

    int live_event_count() {
        return g_live_events;
    }

    void reset_event_count() {
        g_live_events = 0;
    }

    int live_allocation_count() {
        return g_live_allocations;
    }

    void reset_allocation_count() {
        g_live_allocations = 0;
    }

}  // namespace cuda_test

// ---------------------------------------------------------------------------------------------
// Host allocation injection. Replacing operator new is standard C++ and affects every allocation in
// the binary, so the replacement forwards to malloc unless a case has armed it; the suite allocates
// through it constantly and passes.
// ---------------------------------------------------------------------------------------------

namespace {

    thread_local int g_host_fail_from = -1;
    thread_local int g_host_attempts = 0;
    thread_local int g_live_host = 0;

}  // namespace

void* operator new(std::size_t size) {
    const int attempt = g_host_attempts++;
    if (g_host_fail_from >= 0 && attempt >= g_host_fail_from) {
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(size)) {
        ++g_live_host;
        return p;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* p) noexcept {
    if (p != nullptr) {
        --g_live_host;
    }
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    ::operator delete(p);
}

void operator delete[](void* p) noexcept {
    ::operator delete(p);
}

void operator delete[](void* p, std::size_t) noexcept {
    ::operator delete(p);
}

namespace cuda_test {

    void fail_host_allocation_after(int n) {
        g_host_fail_from = n < 0 ? 0 : n;
        g_host_attempts = 0;
    }

    void stop_failing_host_allocations() {
        g_host_fail_from = -1;
        g_host_attempts = 0;
    }

    int host_allocation_attempts() {
        return g_host_attempts;
    }

    int live_host_allocation_count() {
        return g_live_host;
    }

    void reset_live_host_allocation_count() {
        g_live_host = 0;
    }

}  // namespace cuda_test
