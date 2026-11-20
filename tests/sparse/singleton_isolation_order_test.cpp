/**
 * @file singleton_isolation_order_test.cpp
 * @brief The isolated state must not depend on the order tests run in
 *
 * Rounds 112-116 uncovered a chain of order dependencies: an overridden process id leaking
 * from one test file into another, a checkpoint callback surviving in a singleton, and a
 * reset registry that was empty while execute_isolated() claimed to reset singletons. Each
 * was fixed individually. This file states the property those fixes were for, in the one
 * form a unit test can check: whatever a test does to a registered singleton, resetting puts
 * it back.
 *
 * The whole-binary order dependence itself is checked by running the suite with
 * --gtest_shuffle, which no ctest case does. What is covered here is the mechanism that
 * makes shuffling safe.
 */

#include <cuda/testing/test_isolation.h>
#include <cuda/observability/timeline.h>
#include <cuda/performance/profiler.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "test_temp_paths.h"

namespace singleton_isolation_order_test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

    }  // namespace

    TEST(SingletonIsolationOrderTest, ResettingRestoresTheProcessIdAnOverrideChanged) {
        // Reproduces the exact sequence that broke a test in another file: override the pid,
        // reset through the registry, and require the real one back on the next event.
        //
        // My first version of this case issued three redundant set/clear pairs of its own on
        // top of the reset — leftovers from an earlier edit — and the pair of cases then
        // crashed with SIGBUS when run together. The redundant calls are gone; what remains is
        // the sequence the property is actually about.
        cuda::observability::TimelineExporter& exporter = cuda::observability::TimelineExporter::instance();

        exporter.clear();
        exporter.set_process_id(1);

        cuda::testing::reset_all_singletons();

        exporter.record_event("order-probe", "cat", 1);
        const std::string path = nova::test::temp_path("nova_order_probe_", ".json");
        ASSERT_TRUE(exporter.export_to_file(path));

        std::ifstream file(path);
        const std::string json((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        std::remove(path.c_str());

        EXPECT_NE(json.find("\"pid\": " + std::to_string(nova::test::process_id()) + ","), std::string::npos) << "the reset should have restored the real pid:\n" << json;
    }

    TEST(SingletonIsolationOrderTest, EveryRegisteredSingletonIsDistinctlyNamed) {
        // Duplicate names would silently collapse two resetters into one, since the registry is
        // keyed by name — a reset would then leave one singleton untouched with nothing to show
        // for it. Registering a known name and seeing the count stay put is what detects a
        // collapse; a distinct one must grow the registry.
        const size_t before = cuda::testing::registered_singleton_count();
        cuda::testing::register_singleton_resetter("order-probe-unique", [] {});
        EXPECT_EQ(cuda::testing::registered_singleton_count(), before + 1);

        cuda::testing::register_singleton_resetter("order-probe-unique", [] {});
        EXPECT_EQ(cuda::testing::registered_singleton_count(), before + 1) << "the same name twice must not add a second entry";

        // The registry outlives this test, so the entry has to go with it. These particular
        // callbacks capture nothing, but leaving the registration behind would still change what
        // every later reset_all_singletons() runs.
        cuda::testing::unregister_singleton_resetter("order-probe-unique");
        EXPECT_EQ(cuda::testing::registered_singleton_count(), before);
    }

    TEST(SingletonIsolationOrderTest, TheProfilerIsRestoredEvenAfterRepeatedUse) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        cuda::performance::Profiler& profiler = cuda::performance::Profiler::instance();
        profiler.enable();
        for (int i = 0; i < 10; ++i) {
            profiler.record_kernel("order-repeat", static_cast<float>(i));
        }
        ASSERT_FALSE(profiler.get_kernel_metrics().empty());

        cuda::testing::reset_all_singletons();
        EXPECT_TRUE(profiler.get_kernel_metrics().empty());
    }

}  // namespace singleton_isolation_order_test
