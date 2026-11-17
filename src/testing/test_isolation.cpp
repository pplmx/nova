#include "cuda/testing/test_isolation.h"
#include "cuda/performance/profiler.h"
#include "cuda/performance/kernel_profiler.h"
#include "cuda/performance/autotuner.h"
#include "cuda/performance/nvblox_metrics.h"
#include "cuda/tools/timeline_visualizer.h"
#include "cuda/observability/timeline.h"
#include "cuda/error/timeout.hpp"
#include "cuda/error/degrade.hpp"
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

namespace {

/// Registers the singletons whose state must not survive between isolated tests. This was
/// an empty registry: reset_all_singletons() walked nothing, so execute_isolated() reset the
/// CUDA context and left every singleton holding whatever the previous test put there. The
/// entry below is the first; further ones belong in their own translation units next to the
/// definition of their reset method.
struct SingletonRegistrations {
    SingletonRegistrations() {
        register_singleton_resetter("Profiler", [] {
            cuda::performance::Profiler::instance().reset();
        });
        register_singleton_resetter("KernelProfiler", [] {
            cuda::performance::KernelProfiler::instance().reset();
        });
        register_singleton_resetter("AutotuneRegistry", [] {
            cuda::performance::AutotuneRegistry::instance().clear();
        });
        register_singleton_resetter("NVBloxMetricsCollector", [] {
            cuda::performance::NVBloxMetricsCollector::instance().reset();
        });
        register_singleton_resetter("BandwidthAnalyzer", [] {
            cuda::tools::BandwidthAnalyzer::instance().clear();
        });
        register_singleton_resetter("TimelineVisualizer", [] {
            cuda::tools::TimelineVisualizer::instance().clear();
        });
        register_singleton_resetter("TimelineExporter", [] {
            cuda::observability::TimelineExporter::instance().clear();
        });
        register_singleton_resetter("timeout_manager", [] {
            nova::error::timeout_manager::instance().reset();
        });
        register_singleton_resetter("degradation_manager", [] {
            nova::error::degradation_manager::instance().reset();
        });
    }
} g_singleton_registrations;

}  // namespace

void reset_all_singletons() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto& pair : g_singletons) {
        if (pair.second) {
            pair.second();
        }
    }
}

size_t registered_singleton_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_singletons.size();
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
