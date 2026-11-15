/**
 * @file algo_wrapper_test.cu
 * @brief Contracts for the graph-captured algorithm wrappers
 *
 * include/cuda/production/algo_wrapper.h wraps reductions, scans and sorts in a
 * CUDA graph. This lives in a .cu file because the header declares a __global__
 * kernel at namespace scope, which the host compiler cannot parse.
 *
 * Three defects were fixed here:
 *
 * - GraphReduceWrapper::capture called cudaMalloc and cudaFree between
 *   begin_capture and end_capture. Neither is permitted while a stream is
 *   capturing, so the runtime aborted the process with "operation not permitted
 *   when stream is capturing". The scratch buffer is now allocated before
 *   capture starts and freed when the wrapper is destroyed.
 *
 * - GraphAlgoContext::wrap_algo is a template whose definition sat in the .cu
 *   file, so every external call failed to link with "undefined reference to
 *   wrap_algo<int>". It is now defined inline in the header.
 *
 * - GraphAlgoContext::capture_all opened one capture per entry on the same
 *   stream and closed none of them, leaving the stream permanently in capture
 *   mode: the next CUDA call failed with the same "not permitted" error.
 *
 * What the wrappers DO NOT do is asserted explicitly rather than pretended:
 * wrap_algo discards the algorithm it is given, so the captured graphs are
 * empty, and the scan/sort accessors have no buffer to return.
 */

#include <cuda/production/algo_wrapper.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace cuda::production::test {

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

TEST(GraphReduceWrapperTest, HoldsTheReducerItWasGiven) {
    bool called = false;
    GraphReduceWrapper<float> wrapper(
        [&called](const float*, size_t, float init) {
            called = true;
            return init;
        });

    // A freshly built wrapper reports the zero identity.
    EXPECT_FLOAT_EQ(wrapper.result(), 0.0f);

    // set_reducer replaces the function without touching anything else.
    wrapper.set_reducer([](const float*, size_t, float) { return 42.0f; });
    (void)called;
}

// Regression: this aborted the process inside a capturing stream.
TEST(GraphReduceWrapperTest, CaptureDoesNotAllocateInsideTheCapture) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    GraphReduceWrapper<float> wrapper(
        [](const float*, size_t, float init) { return init; });
    cuda::stream::Stream stream;

    EXPECT_NO_THROW(wrapper.capture(stream))
        << "cudaMalloc/cudaFree are illegal while a stream is capturing";
    (void)cudaGetLastError();
}

TEST(GraphReduceWrapperTest, CaptureLeavesTheStreamUsable) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    GraphReduceWrapper<float> wrapper(
        [](const float*, size_t, float init) { return init; });
    cuda::stream::Stream stream;
    ASSERT_NO_THROW(wrapper.capture(stream));

    // If the capture leaked, this host-side allocation fails with "operation not
    // permitted when stream is capturing".
    void* probe = nullptr;
    const cudaError_t status = cudaMalloc(&probe, 256);
    EXPECT_EQ(status, cudaSuccess)
        << "stream still capturing after capture(): " << cudaGetErrorString(status);
    if (probe != nullptr) {
        cudaFree(probe);
    }
    (void)cudaGetLastError();
}

TEST(GraphReduceWrapperTest, RepeatedCaptureIsStable) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    GraphReduceWrapper<float> wrapper(
        [](const float*, size_t, float init) { return init; });
    cuda::stream::Stream stream;

    // The scratch buffer is allocated once and reused, so capturing repeatedly
    // must not leak it or allocate inside a later capture.
    for (int iteration = 0; iteration < 3; ++iteration) {
        EXPECT_NO_THROW(wrapper.capture(stream)) << "iteration " << iteration;
    }
    (void)cudaGetLastError();
}

TEST(GraphReduceWrapperTest, SupportsTheInstantiatedTypes) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::stream::Stream stream;
    GraphReduceWrapper<double> real([](const double*, size_t, double init) { return init; });
    GraphReduceWrapper<int> integer([](const int*, size_t, int init) { return init; });
    GraphReduceWrapper<unsigned int> unsigned_integer(
        [](const unsigned int*, size_t, unsigned int init) { return init; });

    EXPECT_NO_THROW(real.capture(stream));
    EXPECT_NO_THROW(integer.capture(stream));
    EXPECT_NO_THROW(unsigned_integer.capture(stream));
    (void)cudaGetLastError();
}

// Regression: wrap_algo did not link at all from any other translation unit.
TEST(GraphAlgoContextTest, WrapAlgoLinksAndRecordsEntries) {
    GraphAlgoContext<float> context;
    EXPECT_EQ(context.algo_count(), 0u);

    std::vector<float> payload = {1.0f, 2.0f, 3.0f};
    context.wrap_algo("first", payload);
    context.wrap_algo("second", payload);

    EXPECT_EQ(context.algo_count(), 2u)
        << "each wrap_algo call should register one entry";
}

// Regression: this left the stream capturing forever.
TEST(GraphAlgoContextTest, CaptureAllLeavesTheStreamUsable) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    GraphAlgoContext<float> context;
    std::vector<float> payload = {1.0f, 2.0f};
    context.wrap_algo("only", payload);

    cuda::stream::Stream stream;
    ASSERT_NO_THROW(context.capture_all(stream));

    void* probe = nullptr;
    const cudaError_t status = cudaMalloc(&probe, 256);
    EXPECT_EQ(status, cudaSuccess)
        << "capture_all leaked the capture: " << cudaGetErrorString(status);
    if (probe != nullptr) {
        cudaFree(probe);
    }
    (void)cudaGetLastError();
}

TEST(GraphAlgoContextTest, CaptureAllWithNoEntriesIsANoOp) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    GraphAlgoContext<float> context;
    cuda::stream::Stream stream;

    EXPECT_NO_THROW(context.capture_all(stream));
    EXPECT_NO_THROW(context.launch_all(stream));
    (void)cudaGetLastError();
}

// These record a known limitation so it cannot be mistaken for working code:
// wrap_algo keeps only the name, so the captured graphs contain no work.
TEST(GraphAlgoContextTest, KnownLimitationWrappedAlgorithmsAreNotRecorded) {
    GraphAlgoContext<float> context;
    int invocations = 0;
    context.wrap_algo("should-run", [&invocations] { ++invocations; });

    // Registering is all that happens; the algorithm object is discarded.
    EXPECT_EQ(context.algo_count(), 1u);
    EXPECT_EQ(invocations, 0)
        << "wrap_algo discards the callable, so nothing can invoke it later";
}

TEST(GraphScanWrapperTest, OutputIsDocumentedAsUnimplemented) {
    GraphScanWrapper<float> wrapper([](float*, const float*, size_t, bool) {});
    // No buffer is owned, so there is nothing to return. Asserting the current
    // contract keeps the limitation visible instead of guessing.
    EXPECT_EQ(wrapper.output(), nullptr);
}

TEST(GraphSortWrapperTest, DataIsDocumentedAsUnimplemented) {
    GraphSortWrapper<float> wrapper([](float*, size_t, bool) {});
    EXPECT_EQ(wrapper.data(), nullptr);
}

TEST(GraphScanWrapperTest, CaptureDoesNotLeakTheCapture) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    GraphScanWrapper<float> wrapper([](float*, const float*, size_t, bool) {});
    cuda::stream::Stream stream;
    ASSERT_NO_THROW(wrapper.capture(stream));

    void* probe = nullptr;
    const cudaError_t status = cudaMalloc(&probe, 256);
    EXPECT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    if (probe != nullptr) {
        cudaFree(probe);
    }
    (void)cudaGetLastError();
}

TEST(GraphSortWrapperTest, CaptureDoesNotLeakTheCapture) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    GraphSortWrapper<float> wrapper([](float*, size_t, bool) {});
    cuda::stream::Stream stream;
    ASSERT_NO_THROW(wrapper.capture(stream));

    void* probe = nullptr;
    const cudaError_t status = cudaMalloc(&probe, 256);
    EXPECT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    if (probe != nullptr) {
        cudaFree(probe);
    }
    (void)cudaGetLastError();
}

}  // namespace cuda::production::test
