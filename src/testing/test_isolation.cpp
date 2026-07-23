#include "cuda/testing/test_isolation.h"
#include "cuda/device/error.h"

#include <cuda_runtime.h>
#include <map>
#include <mutex>

namespace cuda::testing {

static std::map<std::string, std::function<void()>> g_singletons;
static std::mutex g_mutex;

class CUDATestIsolationContext : public TestIsolationContext {
public:
    void reset_cuda_context() override {
        // cudaDeviceReset is intentionally unchecked: it can fail if
        // the context is already invalid, but we want to proceed anyway
        cudaDeviceReset();
        CUDA_CHECK(cudaSetDevice(0));
    }

    void reset_singletons() override {
        reset_all_singletons();
    }

    void synchronize() override {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    bool is_isolated() const override {
        return true;
    }
};

std::unique_ptr<TestIsolationContext> TestIsolationContext::create() {
    return std::make_unique<CUDATestIsolationContext>();
}

void TestIsolationContext::reset_all() {
    cudaDeviceReset();
    CUDA_CHECK(cudaSetDevice(0));
}

void TestIsolationContext::execute_isolated(std::function<void()> fn) {
    auto context = create();
    context->reset_cuda_context();
    context->reset_singletons();

    fn();

    context->synchronize();
}

void register_singleton_resetter(const char* name, std::function<void()> resetter) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_singletons[name] = resetter;
}

void reset_all_singletons() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto& pair : g_singletons) {
        if (pair.second) {
            pair.second();
        }
    }
}

void reset_all_cuda_state() {
    // Intentionally unchecked: used during test teardown when the context
    // may already be invalid. Silent failure is acceptable here.
    cudaDeviceReset();
}

CUDAContextGuard::CUDAContextGuard()
    : device_(0), shmem_config_(cudaSharedMemBankSizeDefault) {
    CUDA_CHECK(cudaGetDevice(&device_));
    CUDA_CHECK(cudaDeviceGetSharedMemConfig(&shmem_config_));
}

CUDAContextGuard::~CUDAContextGuard() {
    // Destructor: cannot throw (noexcept). These calls restore the
    // previous device/shared memory config during test teardown.
    cudaSetDevice(device_);
    cudaDeviceSetSharedMemConfig(shmem_config_);
}

void CUDAContextGuard::reset() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

cudaError_t CUDAContextGuard::sync() const {
    // Returns error code for the caller to handle
    return cudaDeviceSynchronize();
}

}  // namespace cuda::testing
