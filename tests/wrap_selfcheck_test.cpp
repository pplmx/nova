// Self-check: the new stream and async wrappers must actually intercept.
#include "cuda_alloc_injection.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>

TEST(WrapSelfCheck, StreamWrapperIntercepts) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { (void)cudaGetLastError(); GTEST_SKIP() << "no device"; }
    (void)cudaGetLastError();

    cuda_test::reset_stream_count();
    cuda_test::fail_stream_after(1);
    cudaStream_t s1 = nullptr, s2 = nullptr;
    EXPECT_EQ(cudaStreamCreate(&s1), cudaSuccess);
    EXPECT_NE(cudaStreamCreate(&s2), cudaSuccess);
    cuda_test::stop_failing_streams();
    EXPECT_EQ(cuda_test::live_stream_count(), 1) << "wrapper did not see the creations";
    cudaStreamDestroy(s1);
    EXPECT_EQ(cuda_test::live_stream_count(), 0);
}

TEST(WrapSelfCheck, AsyncWrapperIntercepts) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { (void)cudaGetLastError(); GTEST_SKIP() << "no device"; }
    (void)cudaGetLastError();

    cuda_test::reset_async_allocation_count();
    cuda_test::fail_async_allocation_after(1);
    void* p1 = nullptr; void* p2 = nullptr;
    const cudaError_t a = cudaMallocAsync(&p1, 64, nullptr);
    const cudaError_t b = cudaMallocAsync(&p2, 64, nullptr);
    cuda_test::stop_failing_async_allocations();
    EXPECT_EQ(a, cudaSuccess);
    EXPECT_NE(b, cudaSuccess) << "wrapper did not intercept the second allocation";
    EXPECT_EQ(cuda_test::live_async_allocation_count(), 1);
    if (a == cudaSuccess) cudaFreeAsync(p1, nullptr);
    (void)cudaDeviceSynchronize();
}
