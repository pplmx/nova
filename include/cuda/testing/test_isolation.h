#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <memory>

namespace cuda::testing {

class TestIsolationContext {
public:
    TestIsolationContext() = default;

    static std::unique_ptr<TestIsolationContext> create();
    static void reset_all();

    virtual ~TestIsolationContext() = default;

    virtual void reset_cuda_context() = 0;
    virtual void reset_singletons() = 0;
    virtual void synchronize() = 0;

    virtual bool is_isolated() const = 0;

    void execute_isolated(std::function<void()> fn);
};

class ScopedTestIsolation {
public:
    explicit ScopedTestIsolation();
    ~ScopedTestIsolation();

    ScopedTestIsolation(const ScopedTestIsolation&) = delete;
    ScopedTestIsolation& operator=(const ScopedTestIsolation&) = delete;

    static ScopedTestIsolation& current();

private:
    std::unique_ptr<TestIsolationContext> context_;
};

void reset_all_cuda_state();

/**
 * Registers a reset callback for a process-wide singleton.
 *
 * `reset_all_singletons()` walks the registrations and calls each one; a singleton that
 * does not register here is not reset by it. That distinction was invisible before this
 * comment existed, and the registry was empty in the shipped code — the walk iterated
 * nothing, so `reset_all_singletons()` silently did less than its name says.
 *
 * A singleton that holds mutable state between tests should register from its own
 * translation unit, next to the definition of its reset method:
 *
 * @code
 * namespace {
 * struct Registrar {
 *     Registrar() {
 *         register_singleton_resetter("Profiler", [] {
 *             Profiler::instance().reset();
 *         });
 *     }
 * } g_registrar;
 * }
 * @endcode
 *
 * Names are used for diagnostics only; registering the same name twice replaces the
 * previous callback rather than adding a second one.
 */
void register_singleton_resetter(const char* name, std::function<void()> resetter);

/// Calls every registered resetter. Does nothing for a singleton that has not registered —
/// see register_singleton_resetter() for what that means and why it matters.
void reset_all_singletons();

/// How many resetter registrations exist. Exposed so a test can assert the registry was
/// populated rather than trusting that it was.
size_t registered_singleton_count();

class CUDAContextGuard {
public:
    CUDAContextGuard();
    ~CUDAContextGuard();

    CUDAContextGuard(const CUDAContextGuard&) = delete;
    CUDAContextGuard& operator=(const CUDAContextGuard&) = delete;

    void reset();
    cudaError_t sync() const;

private:
    int device_;
    cudaSharedMemConfig shmem_config_;
};

}  // namespace cuda::testing
