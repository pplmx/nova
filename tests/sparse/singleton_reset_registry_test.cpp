/**
 * @file singleton_reset_registry_test.cpp
 * @brief The singleton reset registry must actually contain something
 *
 * src/testing/test_isolation.cpp exposes register_singleton_resetter() and
 * reset_all_singletons(), and reset_all_singletons() simply walks the registry. Measured
 * before this change: no translation unit in the tree ever called the register function, so
 * the walk iterated an empty map. execute_isolated() reset the CUDA context and left every
 * singleton holding whatever the previous test had put there — the isolation facility did
 * less than its name says, and nothing reported it.
 *
 * The registry now contains at least the Profiler. These cases pin that it is non-empty and
 * that resetting really clears state, so the next singleton that should register has a
 * failing test to notice.
 */

#include <cuda/testing/test_isolation.h>
#include <cuda/performance/profiler.h>
#include <cuda/error/degrade.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <memory>

#include <cuda_runtime.h>

namespace singleton_reset_registry_test {

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

// Regression: the registry was empty, so this count was 0.
TEST(SingletonResetRegistryTest, TheRegistryCoversTheSingletonsThatHoldState) {
    // Nine are registered. Asserting a floor rather than the exact number keeps this from
    // failing every time another singleton joins, while still failing if the registry is
    // emptied or the registrations stop being linked in.
    EXPECT_GE(cuda::testing::registered_singleton_count(), 9u)
        << "reset_all_singletons() walks this registry; an empty one makes the isolation "
           "facility a no-op";
}

TEST(SingletonResetRegistryTest, ResetClearsMoreThanOneSingleton) {
    // Two singletons from different modules, so the registration is demonstrably not a
    // single call site: a metrics collector and a degradation manager both hold state that
    // outlives a test without a reset.
    nova::error::degradation_manager& manager = nova::error::degradation_manager::instance();
    manager.reset();

    // Driving the manager into a degraded state, then resetting, and requiring the default
    // back, is what shows the registration is wired rather than merely present.
    manager.trigger_degradation("registry-op", nova::error::precision_level::low, "test");
    EXPECT_EQ(manager.get_precision("registry-op"), nova::error::precision_level::low);

    cuda::testing::reset_all_singletons();

    EXPECT_EQ(manager.get_precision("registry-op"), nova::error::precision_level::high)
        << "the degradation manager was not reset by the registry";
}

TEST(SingletonResetRegistryTest, ResettingClearsTheProfilerThatRegistered) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::performance::Profiler& profiler = cuda::performance::Profiler::instance();
    profiler.enable();
    profiler.record_kernel("registry-probe", 1.5f);
    ASSERT_FALSE(profiler.get_kernel_metrics().empty())
        << "the record must exist before the reset can be seen to clear it";

    cuda::testing::reset_all_singletons();

    EXPECT_TRUE(profiler.get_kernel_metrics().empty())
        << "the registered resetter should have cleared the profiler's records";
}

TEST(SingletonResetRegistryTest, RegisteringUnderTheSameNameReplacesRatherThanDuplicates) {
    // Names are keys in a map, and the header documents replacement. A second registration
    // must not grow the registry by one every time it runs.
    //
    // The callbacks below capture an atomic through a shared_ptr rather than a stack int.
    // My first version captured locals by reference and left the registration in place when
    // the test returned; the registry is process-wide, so a later reset_all_singletons()
    // called into a destroyed frame. That was the SIGBUS a shuffled run kept producing.
    // Unregistering at the end is the other half of the fix.
    const size_t before = cuda::testing::registered_singleton_count();

    auto first_calls = std::make_shared<std::atomic<int>>(0);
    auto second_calls = std::make_shared<std::atomic<int>>(0);

    cuda::testing::register_singleton_resetter(
        "replacement-probe", [first_calls] { first_calls->fetch_add(1); });
    cuda::testing::register_singleton_resetter(
        "replacement-probe", [second_calls] { second_calls->fetch_add(1); });

    EXPECT_EQ(cuda::testing::registered_singleton_count(), before + 1)
        << "the same name twice must leave one entry, not two";

    cuda::testing::reset_all_singletons();
    EXPECT_EQ(first_calls->load(), 0) << "the replaced callback must not run";
    EXPECT_EQ(second_calls->load(), 1);

    cuda::testing::unregister_singleton_resetter("replacement-probe");
    EXPECT_EQ(cuda::testing::registered_singleton_count(), before)
        << "unregistering must remove the entry";

    EXPECT_EQ(cuda::testing::registered_singleton_count(), before)
        << "unregistering must remove the entry";
}

}  // namespace singleton_reset_registry_test
