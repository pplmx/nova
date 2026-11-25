#pragma once

/**
 * @file cuda_alloc_injection.h
 * @brief Make the next matching CUDA allocation fail, from a test.
 *
 * Some of the library's error paths only run when an allocation fails -- a cudaMalloc that returns
 * cudaErrorMemoryAllocation inside a constructor, a reallocation half-way through a resize, a guard
 * that releases what it took before rethrowing. Those paths could not be tested before this: the
 * failure cannot be requested through any public API, and exhausting device memory on purpose is both
 * unreliable and destructive to the rest of the suite.
 *
 * The mechanism is the linker's --wrap. tests/CMakeLists.txt passes --wrap=cudaMalloc (and the other
 * allocation entry points) when linking nova-tests, so a call to cudaMalloc from anywhere in the
 * library resolves to this file's __wrap_cudaMalloc, which either fails on request or forwards to
 * __real_cudaMalloc. Production code is untouched and the wrapper is not part of any shipped library.
 *
 * Usage:
 *
 *     cuda_test::fail_allocation_after(0);   // the next cudaMalloc fails
 *     EXPECT_THROW(pool.allocate(1024), std::runtime_error);
 *     cuda_test::stop_failing_allocations();
 *
 * The counter is per-thread and counts only allocations that did NOT fail, so
 * fail_allocation_after(1) lets one through and fails the second.
 */
namespace cuda_test {

    /// Fails every allocation from the nth onward. 0 fails the next one. The counter resets whenever
    /// this is called, so successive calls do not accumulate.
    void fail_allocation_after(int n);

    /// Returns to forwarding every allocation. Call this even when a case fails, so one broken case
    /// does not poison the ones after it.
    void stop_failing_allocations();

    /// How many allocations have been attempted since the last fail_allocation_after/stop.
    int allocation_attempts();

    /// CUDA events created minus destroyed since the last reset. Driver objects are not heap
    /// allocations, so ASan says nothing about an event that is never destroyed and CUDA offers no
    /// handle-count query -- counting them here is the only way a case can tell.
    ///
    /// Only useful for callers that live in a compiled translation unit. It does NOT see calls made
    /// from a header-only template instantiated in the test itself: --wrap redirects unresolved
    /// symbols at link time, and a template body compiled into the caller resolves cudaEventCreate
    /// directly. Verified by trying to use it on Benchmark::run(), which is a template in
    /// include/cuda/benchmark/benchmark.h -- the count stayed 0 through a mutation that created two
    /// events and never destroyed them, so the case written for it was removed rather than left
    /// green and meaningless.
    int live_event_count();

    /// Zeroes the event counter. Call before the code under test, then check live_event_count() after.
    void reset_event_count();

    /// Device allocations that went through the wrapper and have not been freed yet. Unlike
    /// live_event_count() this does work for code compiled into the library, because the calls it
    /// counts are in .cpp/.cu files rather than in a header-only template.
    int live_allocation_count();

    /// Zeroes the allocation counter.
    void reset_allocation_count();

    /// Makes the next host allocation (operator new) fail, throwing std::bad_alloc.
    ///
    /// Separate from fail_allocation_after(), which covers cudaMalloc: a constructor that takes host
    /// arrays with new[] and a constructor that takes device buffers with cudaMalloc fail through
    /// different functions, and the guards for them need different hooks. Replacing operator new is
    /// ordinary C++; the replacement in cuda_alloc_injection.cpp forwards to malloc, so it changes
    /// nothing while idle -- the rest of the suite allocates through it and passes.
    void fail_host_allocation_after(int n);

    /// Returns operator new to forwarding every request.
    void stop_failing_host_allocations();

    /// Host allocations attempted since the last fail_host_allocation_after/stop.
    int host_allocation_attempts();

    /// Host allocations that have been made and not yet freed. Lets a case assert that an object
    /// which threw out of its constructor released what it took, instead of only that it threw.
    int live_host_allocation_count();

    /// Zeroes the live host allocation count. Call right before the code under test.
    void reset_live_host_allocation_count();

}  // namespace cuda_test
