/**
 * @file cusparse_thread_safety_test.cpp
 * @brief The cuSPARSE context is a process-wide singleton and is not thread-safe
 *
 * include/cuda/sparse/cusparse_context.hpp claimed "Thread-safe handle management". It is
 * not: CusparseContext::get() returns one object for the whole process, and set_stream()
 * writes stream_ and calls cusparseSetStream on that shared handle with no lock. sparse_ops.cu
 * calls set_stream() before every SpMV, so two threads running SpMV concurrently race — one
 * can issue its operation on the other's stream.
 *
 * The header now says so. These cases document the part of the contract that remains true —
 * one process-wide handle, one stream at a time — and do not try to provoke the race, because
 * a data race that happens to pass proves nothing.
 */

#include <cuda/sparse/cusparse_context.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace cusparse_thread_safety_test {

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

TEST(CusparseThreadSafetyTest, GetReturnsTheSameHandleEveryTime) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The single shared object is what makes set_stream racy. If get() ever returned a
    // per-thread context, this case would fail and the warning could be removed.
    auto& first = nova::sparse::detail::CusparseContext::get();
    auto& second = nova::sparse::detail::CusparseContext::get();

    EXPECT_EQ(&first, &second) << "the context is a process-wide singleton";
    EXPECT_NE(first.handle(), nullptr);
}

TEST(CusparseThreadSafetyTest, SetStreamChangesWhatTheNextReaderSees) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    auto& ctx = nova::sparse::detail::CusparseContext::get();
    cudaStream_t stream_a = nullptr;
    cudaStream_t stream_b = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream_a), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream_b), cudaSuccess);

    // Sequential calls are consistent; only concurrent ones interfere. This is the positive
    // half of the same property, and the second assertion is the shape of the race.
    ctx.set_stream(stream_a);
    EXPECT_EQ(ctx.stream(), stream_a);
    ctx.set_stream(stream_b);
    EXPECT_EQ(ctx.stream(), stream_b)
        << "a second set_stream overrides the first, which is what two threads would do";

    ctx.set_stream(nullptr);
    (void)cudaStreamDestroy(stream_a);
    (void)cudaStreamDestroy(stream_b);
}

}  // namespace cusparse_thread_safety_test
