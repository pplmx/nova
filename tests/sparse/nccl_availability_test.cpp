/**
 * @file nccl_availability_test.cpp
 * @brief has_nccl() must answer whether a collective can actually run
 *
 * NcclContext::has_nccl() is the readiness gate every caller uses before entering a
 * collective, so it has to be false whenever one cannot run. It tested
 * `!communicators_.empty()`, which is a different question: create_streams_and_comms()
 * resizes that vector to device_count_ before either branch runs, and only the NCCL branch
 * fills it. In a NOVA_ENABLE_NCCL=OFF build the vector therefore holds device_count_ null
 * entries and the gate answered true.
 *
 * Measured on this host before the fix, with a small programme linked against the no-NCCL
 * library:
 *
 *     NOVA_NCCL_ENABLED=0 has_nccl=1 device_count=2
 *
 * so callers were told NCCL was ready and then posted a collective that returned "NCCL not
 * enabled". The build without NCCL is the configuration where this answer matters most, and
 * it was the one that was wrong.
 *
 * The assertions are written so they hold in both configurations: with NCCL the answer is
 * true, and without it the answer must be false.
 */

#include <cuda/nccl/nccl_context.h>

#include <gtest/gtest.h>

#include <string>

#ifndef NOVA_NCCL_ENABLED
#error "NOVA_NCCL_ENABLED must be defined by cuda_nccl's interface; see CMakeLists.txt"
#endif

namespace cuda::nccl::test {

    // A default-constructed context has not been initialised, so nothing can run on it
    // regardless of how the build was configured. This was already correct and stays here to
    // pin the "not initialised" half of the condition.
    TEST(NcclAvailabilityTest, AnUninitialisedContextIsNotReady) {
        NcclContext context;
        EXPECT_FALSE(context.has_nccl());
    }

    // The regression. Without NCCL compiled in, no arrangement of the context can make a
    // collective work, so the gate must never report ready.
    TEST(NcclAvailabilityTest, AContextIsNeverReadyWhenNcclIsNotCompiledIn) {
#if NOVA_NCCL_ENABLED
        GTEST_SKIP() << "this build links NCCL; the no-NCCL answer is covered by the sibling case";
#else
        NcclContext context;
        try {
            context.initialize();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "no device to initialise on: " << e.what();
        }
        EXPECT_FALSE(context.has_nccl()) << "has_nccl() said a collective could run in a build that compiles NCCL out";
#endif
    }

    // The mirror case: with NCCL linked, an initialised context on a real multi-GPU host must
    // report ready, so the fix above cannot be satisfied by always answering false.
    TEST(NcclAvailabilityTest, AnInitialisedContextWithNcclIsReady) {
#if !NOVA_NCCL_ENABLED
        GTEST_SKIP() << "this build compiles NCCL out; that answer is covered by the sibling case";
#else
        auto& context = NcclContext::instance();
        try {
            context.initialize();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "NCCL is not usable on this host: " << e.what();
        }
        EXPECT_TRUE(context.has_nccl()) << "an initialised context with communicators must report ready";
#endif
    }

    // The singleton and a fresh context must agree about readiness being a property of the
    // configuration rather than of which instance was asked.
    TEST(NcclAvailabilityTest, TheAnswerDoesNotDependOnWhichInstanceIsAsked) {
#if NOVA_NCCL_ENABLED
        GTEST_SKIP() << "covered with NCCL compiled out, where the answer is unambiguously false";
#else
        NcclContext fresh;
        try {
            fresh.initialize();
        } catch (const std::exception&) {
            GTEST_SKIP() << "no device to initialise on";
        }
        auto& singleton = NcclContext::instance();
        try {
            singleton.initialize();
        } catch (const std::exception&) {
            GTEST_SKIP() << "no device to initialise on";
        }
        EXPECT_EQ(fresh.has_nccl(), singleton.has_nccl());
        EXPECT_FALSE(fresh.has_nccl());
#endif
    }

}  // namespace cuda::nccl::test
