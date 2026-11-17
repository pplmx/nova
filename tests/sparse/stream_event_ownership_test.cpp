/**
 * @file stream_event_ownership_test.cpp
 * @brief Stream and Event must move without leaving two owners of one handle
 *
 * include/cuda/stream/stream.h and event.h carried no documentation, though they are the
 * types every other module builds on. Their ownership behaviour is correct as written; these
 * cases make it explicit, because the two mistakes a reader could make — assuming they are
 * copyable, or assuming a moved-from object still works — are both silent when wrong.
 *
 * The header now states the rules; this file is what would notice if they changed.
 */

#include <cuda/stream/event.h>
#include <cuda/stream/stream.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <utility>

namespace stream_event_ownership_test {

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

TEST(StreamEventOwnershipTest, AMovedFromStreamReportsNoHandle) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::stream::Stream source;
    cudaStream_t original = source.get();
    ASSERT_NE(original, nullptr);

    cuda::stream::Stream destination(std::move(source));

    EXPECT_EQ(destination.get(), original) << "the handle moves unchanged";
    EXPECT_EQ(source.get(), nullptr)
        << "the moved-from stream must be null, or both destructors destroy one handle";
}

TEST(StreamEventOwnershipTest, MoveAssignmentReleasesTheTargetsOldHandle) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::stream::Stream target;
    cuda::stream::Stream source;
    cudaStream_t source_handle = source.get();
    ASSERT_NE(target.get(), nullptr);
    ASSERT_NE(source_handle, nullptr);
    ASSERT_NE(target.get(), source_handle);

    target = std::move(source);

    // The target's original handle was destroyed by the move assignment; the only thing
    // observable here is that it now holds the source's handle and the source is empty.
    EXPECT_EQ(target.get(), source_handle);
    EXPECT_EQ(source.get(), nullptr);
}

TEST(StreamEventOwnershipTest, StreamsAreNotCopyable) {
    // A compile-time property; asserting it here documents the intent next to the run-time
    // cases. If either were made copyable, this would stop compiling.
    static_assert(!std::is_copy_constructible_v<cuda::stream::Stream>);
    static_assert(!std::is_copy_assignable_v<cuda::stream::Stream>);
    static_assert(!std::is_copy_constructible_v<cuda::stream::Event>);
    static_assert(!std::is_copy_assignable_v<cuda::stream::Event>);
    SUCCEED();
}

TEST(StreamEventOwnershipTest, AMovedFromEventReportsNoHandle) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::stream::Event source;
    cudaEvent_t original = source.get();
    ASSERT_NE(original, nullptr);

    cuda::stream::Event destination(std::move(source));

    EXPECT_EQ(destination.get(), original);
    EXPECT_EQ(source.get(), nullptr);
}

TEST(StreamEventOwnershipTest, AnEventRecordedOnAStreamCanBeJoinedByTheHost) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::stream::Stream stream;
    cuda::stream::Event event;

    // A tiny async operation, then the event after it.
    int* device = nullptr;
    ASSERT_EQ(cudaMalloc(&device, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(device, 0, sizeof(int), stream.get()), cudaSuccess);
    EXPECT_NO_THROW(event.record(stream));

    // Waiting on the event is what makes the recorded position meaningful; reaching the
    // assertion below means the event was joinable.
    EXPECT_EQ(cudaEventSynchronize(event.get()), cudaSuccess);

    (void)cudaFree(device);
}

}  // namespace stream_event_ownership_test
