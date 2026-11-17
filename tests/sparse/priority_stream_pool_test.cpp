/**
 * @file priority_stream_pool_test.cpp
 * @brief Releasing a stream twice must not corrupt the pool
 *
 * PriorityStreamPool::release() pushed the handle into the pool's vector without checking
 * whether it was already there. Releasing the same PriorityStream twice therefore stored two
 * copies, and cleanup() later called cudaStreamDestroy on both — measured: the second release
 * of an acquired stream dumped core under timeout. Nothing in the header documented acquire /
 * release as a pairing either; it had no comment at all.
 *
 * release() now returns early when the pool already holds the handle. The cases below cover
 * that, the ordinary acquire/release cycle, and cleanup being callable from outside the
 * destructor.
 */

#include <cuda/production/priority_stream.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace priority_stream_pool_test {

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

TEST(PriorityStreamPoolTest, ReleasingTheSameStreamTwiceDoesNotDuplicateIt) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::production::PriorityStreamPool pool(2);
    const auto stream = pool.acquire(cuda::production::StreamPriority::Normal);
    ASSERT_TRUE(static_cast<bool>(stream));

    const size_t before = pool.total_available();
    pool.release(stream);
    const size_t after_first = pool.total_available();
    EXPECT_EQ(after_first, before + 1) << "one release should return one stream";

    // Regression: this call used to put a second copy of the handle in the pool.
    EXPECT_NO_THROW(pool.release(stream));
    EXPECT_EQ(pool.total_available(), after_first)
        << "a second release of the same handle must not add another entry";
}

TEST(PriorityStreamPoolTest, AcquireAndReleaseRoundTrips) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::production::PriorityStreamPool pool(2);

    auto first = pool.acquire(cuda::production::StreamPriority::High);
    ASSERT_TRUE(static_cast<bool>(first));
    EXPECT_EQ(first.priority(), cuda::production::StreamPriority::High);

    auto second = pool.acquire(cuda::production::StreamPriority::High);
    ASSERT_TRUE(static_cast<bool>(second));
    EXPECT_NE(first.get(), second.get()) << "two acquisitions must not hand out one stream";

    pool.release(first);
    pool.release(second);
    EXPECT_EQ(pool.available_count(cuda::production::StreamPriority::High), 2u);
}

TEST(PriorityStreamPoolTest, ReleasingADefaultConstructedStreamIsIgnored) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::production::PriorityStreamPool pool(2);
    const size_t before = pool.total_available();

    // A default-constructed PriorityStream holds nullptr; releasing it must be a no-op
    // rather than storing a null handle the destructor would then try to destroy.
    EXPECT_NO_THROW(pool.release(cuda::production::PriorityStream{}));
    EXPECT_EQ(pool.total_available(), before);
}

TEST(PriorityStreamPoolTest, CleanupIsSafeToCallTwice) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // cleanup() is public and the destructor calls it too; a second call must find empty
    // pools rather than destroy the same handles again.
    auto pool = std::make_unique<cuda::production::PriorityStreamPool>(2);
    (void)pool->acquire(cuda::production::StreamPriority::Low);
    EXPECT_NO_THROW(pool->cleanup());
    EXPECT_NO_THROW(pool->cleanup());
}

}  // namespace priority_stream_pool_test
