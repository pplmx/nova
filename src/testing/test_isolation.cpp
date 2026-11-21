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
#include <cuda/distributed/common.h>
#include <map>
#include <vector>
#include <mutex>

namespace cuda::testing {

    namespace {

        /// The registry and its mutex are reached through accessors, not file-scope
        /// objects.
        ///
        /// register_singleton_resetter() is called from the constructors of static
        /// registration objects in OTHER translation units (the convention documented in
        /// the header is to register from the singleton's own TU). Those constructors can
        /// run before this one does, and a std::map that has not been constructed yet is
        /// not an empty map -- inserting into it dereferences an uninitialised tree.
        /// Measured: registering one cache from src/cuda/distributed/matmul.cu crashed
        /// every nova-tests invocation inside std::_Rb_tree_decrement before main() ran,
        /// which also broke gtest test discovery and so the build itself.
        ///
        /// Function-local statics are constructed on first use, so whichever order the
        /// translation units initialise in, the registry exists before it is touched.
        std::map<std::string, std::function<void()>>& singleton_registry() {
            static auto* registry = new std::map<std::string, std::function<void()>>();
            return *registry;
        }

        std::mutex& singleton_registry_mutex() {
            static auto* mutex = new std::mutex();
            return *mutex;
        }

    }  // namespace

    class CUDATestIsolationContext : public TestIsolationContext {
    public:
        void reset_cuda_context() override {
            // cudaDeviceReset is intentionally unchecked: it can fail if
            // the context is already invalid, but we want to proceed anyway
            cudaDeviceReset();
            CUDA_CHECK(cudaSetDevice(0));
        }

        void reset_singletons() override { reset_all_singletons(); }

        void synchronize() override { CUDA_CHECK(cudaDeviceSynchronize()); }

        bool is_isolated() const override { return true; }
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
        // Singletons FIRST. Their resetters release CUDA handles those singletons own
        // (streams, events, cuBLAS/cuSPARSE handles), and cudaDeviceReset() destroys the
        // context those handles live on. In the old order the resetters ran afterwards,
        // on dead handles: libcuda dereferences them rather than returning an error, so
        // the first singleton with a cached event crashed inside cuEventDestroy --
        // measured as an AddressSanitizer SEGV in MeshStreams::reset() reached from this
        // function. Every singleton that owns CUDA state has to be registered here for
        // the order to help; MeshStreams was not, and is now.
        context->reset_singletons();
        context->reset_cuda_context();

        fn();

        context->synchronize();
    }

    void register_singleton_resetter(const char* name, std::function<void()> resetter) {
        std::lock_guard<std::mutex> lock(singleton_registry_mutex());
        singleton_registry()[name] = std::move(resetter);
    }

    namespace {

        /// Registers the singletons whose state must not survive between isolated tests. This was
        /// an empty registry: reset_all_singletons() walked nothing, so execute_isolated() reset the
        /// CUDA context and left every singleton holding whatever the previous test put there. The
        /// entry below is the first; further ones belong in their own translation units next to the
        /// definition of their reset method.
        struct SingletonRegistrations {
            SingletonRegistrations() {
                register_singleton_resetter("Profiler", [] { cuda::performance::Profiler::instance().reset(); });
                register_singleton_resetter("KernelProfiler", [] { cuda::performance::KernelProfiler::instance().reset(); });
                register_singleton_resetter("AutotuneRegistry", [] { cuda::performance::AutotuneRegistry::instance().clear(); });
                register_singleton_resetter("NVBloxMetricsCollector", [] { cuda::performance::NVBloxMetricsCollector::instance().reset(); });
                register_singleton_resetter("BandwidthAnalyzer", [] { cuda::tools::BandwidthAnalyzer::instance().clear(); });
                register_singleton_resetter("TimelineVisualizer", [] { cuda::tools::TimelineVisualizer::instance().clear(); });
                register_singleton_resetter("TimelineExporter", [] { cuda::observability::TimelineExporter::instance().clear(); });
                register_singleton_resetter("timeout_manager", [] { nova::error::timeout_manager::instance().reset(); });
                register_singleton_resetter("degradation_manager", [] { nova::error::degradation_manager::instance().reset(); });
                // Owns a CUDA stream and event per device. Without an entry its handles were
                // released after cudaDeviceReset() had already destroyed the context, which
                // segfaults inside cuEventDestroy.
                register_singleton_resetter("MeshStreams", [] { cuda::distributed::MeshStreams::instance().reset(); });
            }
        } g_singleton_registrations;

    }  // namespace

    void reset_all_singletons() {
        // Copy the callbacks out and release the lock before running any of them.
        //
        // Calling them under the registry mutex meant a resetter that reaches back into the registry —
        // registering another singleton, or anything that does — would re-enter a non-recursive
        // mutex. Running user code while holding a lock that user code may need is the same
        // mistake the profiler's stop_timer made with its own mutex.
        std::vector<std::function<void()>> callbacks;
        {
            auto& registry = singleton_registry();
            std::lock_guard<std::mutex> lock(singleton_registry_mutex());
            callbacks.reserve(registry.size());
            for (auto& pair : registry) {
                if (pair.second) {
                    callbacks.push_back(pair.second);
                }
            }
        }

        for (auto& callback : callbacks) {
            callback();
        }
    }

    void unregister_singleton_resetter(const char* name) {
        std::lock_guard<std::mutex> lock(singleton_registry_mutex());
        singleton_registry().erase(name);
    }

    size_t registered_singleton_count() {
        std::lock_guard<std::mutex> lock(singleton_registry_mutex());
        return singleton_registry().size();
    }

    void reset_all_cuda_state() {
        // Intentionally unchecked: used during test teardown when the context
        // may already be invalid. Silent failure is acceptable here.
        cudaDeviceReset();
    }

    CUDAContextGuard::CUDAContextGuard()
        : device_(0),
          shmem_config_(cudaSharedMemBankSizeDefault) {
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
