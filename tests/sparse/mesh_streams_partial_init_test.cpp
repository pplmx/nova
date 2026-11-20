/**
 * @file mesh_streams_partial_init_test.cpp
 * @brief MeshStreams::initialize must not leak the streams it already created
 *
 * initialize() creates a stream and an event per device, each behind a CUDA_CHECK that throws:
 *
 *     CUDA_CHECK(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
 *     CUDA_CHECK(cudaEventCreate(&events_[i]));
 *
 * A failure on device i used to leave devices 0..i-1's streams and events unreferenced — the
 * vectors still held the handles, but the destructor only runs on a fully constructed object,
 * so nothing released them. The partial set is now destroyed on the way out.
 *
 * The failing path is hard to provoke on demand (it needs a driver-level failure at a
 * particular iteration), so these cases cover the contract that is observable: a successful
 * initialisation still produces one stream and one event per device, and re-initialising an
 * already-initialised context is refused rather than leaking a second set.
 */

#include <cuda/distributed/common.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>

namespace mesh_streams_partial_init_test {

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

    TEST(MeshStreamsPartialInitTest, InitialisingGivesOneStreamAndEventPerDevice) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        auto& streams = cuda::distributed::MeshStreams::instance();
        const int device_count = 1;  // one device is enough to check the per-device contract
        streams.initialize(device_count);

        for (int i = 0; i < device_count; ++i) {
            EXPECT_NE(streams.get_stream(i), nullptr) << "stream " << i;
            EXPECT_NE(streams.get_event(i), nullptr) << "event " << i;
        }
    }

    TEST(MeshStreamsPartialInitTest, ASecondInitializeIsIgnoredRatherThanReallocating) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        auto& streams = cuda::distributed::MeshStreams::instance();
        streams.initialize(1);

        const cudaStream_t first = streams.get_stream(0);

        // initialize() returns early when already initialised, before it looks at its argument at
        // all — so a second call neither throws nor replaces the streams it already made. My first
        // version of this case asserted a throw and passed alone while failing after its sibling
        // had initialised the singleton. The early return is the real contract, and it is what
        // keeps a repeat call from leaking a second set.
        EXPECT_NO_THROW(streams.initialize(1));
        EXPECT_EQ(streams.get_stream(0), first) << "the existing stream must be reused";
    }

    // reset() returns the one-shot singleton to its uninitialised state. Without it the
    // cost of initialize() being one-shot is that nothing can observe a second build --
    // which is why the class's threading claim could not be tested at all.
    TEST(MeshStreamsPartialInitTest, ResetReturnsToTheUninitialisedState) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        auto& streams = cuda::distributed::MeshStreams::instance();
        streams.initialize(1);
        ASSERT_TRUE(streams.initialized());
        const cudaStream_t first = streams.get_stream(0);
        ASSERT_NE(first, nullptr);

        streams.reset();
        EXPECT_FALSE(streams.initialized());
        EXPECT_EQ(streams.device_count(), 0);
        // Reading through the torn-down state is refused rather than returning a
        // destroyed handle.
        EXPECT_THROW(streams.get_stream(0), std::runtime_error);

        // And it can be brought up again, which is the whole point.
        streams.initialize(1);
        EXPECT_TRUE(streams.initialized());
        EXPECT_NE(streams.get_stream(0), nullptr);
    }

}  // namespace mesh_streams_partial_init_test
