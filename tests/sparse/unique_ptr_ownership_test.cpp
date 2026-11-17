/**
 * @file unique_ptr_ownership_test.cpp
 * @brief cuda::memory::unique_ptr must free what it owns and hand back what it releases
 *
 * include/cuda/memory/unique_ptr.h had no documentation at all. What a reader most needs to
 * know is the difference between get() and release(): the first keeps ownership, the second
 * gives it away, and an ignored release() leaks the allocation because cudaFree is the only
 * way back. A test in this tree discarded a release() result and leaked 4 MB per run.
 *
 * The leak itself cannot be measured here — cudaMemGetInfo is device-wide, and the allocator
 * pools freed blocks so a small per-run leak need not move it at all (an injected leak was
 * measured passing such a check). These cases pin the semantics instead: what the pointer is
 * before and after, and that a released pointer is still usable, which is only true if it was
 * not freed.
 */

#include <cuda/memory/unique_ptr.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace unique_ptr_ownership_test {

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

TEST(UniquePtrOwnershipTest, ReleaseHandsOverAPointerThatIsStillValid) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::memory::unique_ptr<float> owner(16);
    ASSERT_NE(owner.get(), nullptr);

    float* raw = owner.release();

    // The pointer must survive release: it was handed over, not freed.
    ASSERT_NE(raw, nullptr);
    EXPECT_EQ(owner.get(), nullptr) << "the owner must forget what it released";

    // Touching it proves it was not freed. A memset is enough and does not read garbage.
    EXPECT_EQ(cudaMemset(raw, 0, 16 * sizeof(float)), cudaSuccess);

    // The caller now owns it. This is the step a forgotten release() result skips.
    EXPECT_EQ(cudaFree(raw), cudaSuccess);
}

TEST(UniquePtrOwnershipTest, ResetFreesTheOldBufferAndTakesTheNewOne) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::memory::unique_ptr<float> owner(8);
    float* first = owner.get();
    ASSERT_NE(first, nullptr);

    owner.reset();
    EXPECT_EQ(owner.get(), nullptr);

    float* replacement = nullptr;
    ASSERT_EQ(cudaMalloc(&replacement, 8 * sizeof(float)), cudaSuccess);
    owner.reset(replacement);
    EXPECT_EQ(owner.get(), replacement) << "reset must adopt the pointer it is given";
    // The destructor frees it; nothing else to do here.
}

TEST(UniquePtrOwnershipTest, MovingTransfersOwnershipWithoutFreeing) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::memory::unique_ptr<float> source(8);
    float* original = source.get();
    ASSERT_NE(original, nullptr);

    cuda::memory::unique_ptr<float> destination(std::move(source));

    EXPECT_EQ(destination.get(), original) << "the pointer moves unchanged";
    EXPECT_EQ(source.get(), nullptr) << "the source must not keep a pointer it gave away";
}

TEST(UniquePtrOwnershipTest, AZeroCountAllocationIsEmptyRatherThanFailing) {
    cuda::memory::unique_ptr<float> owner(0);
    EXPECT_EQ(owner.get(), nullptr)
        << "cudaMalloc(0) would return null anyway; the constructor skips it";
}

}  // namespace unique_ptr_ownership_test
