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

void register_singleton_resetter(const char* name, std::function<void()> resetter);
void reset_all_singletons();

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
