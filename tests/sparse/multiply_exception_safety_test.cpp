/**
 * @file multiply_exception_safety_test.cpp
 * @brief A failing multiplication must not leak what it already allocated
 *
 * multiplyMatricesOnGPU allocated three device buffers with raw cudaMalloc and freed them at
 * the end of the function. Every allocation sat behind CUDA_CHECK, which throws. So if the
 * second allocation failed — the ordinary way this function fails on a problem too large for
 * the GPU — the first leaked, and the host-to-device copies could leak all three. CUDA's
 * allocator keeps freed blocks in its pool but has no way to recover a block nobody holds a
 * pointer to.
 *
 * The buffers are now cuda::memory::unique_ptr, so unwinding releases them.
 *
 * Detecting the leak is the hard part. cudaMemGetInfo reports device-wide free memory, which
 * other tests on the same device disturb, and the allocator's pooling means a small
 * per-call leak need not move the figure at all — a leak test built on it here was measured
 * passing with a deliberate leak injected, so it was removed rather than trusted. What is
 * checked instead is the property that makes the leak impossible: the function completes
 * correctly on the ordinary path, and the allocator still holds the same amount after
 * repeated failing calls as after successful ones is NOT asserted, because the measurement
 * cannot support it.
 */

#include <matrix/mult.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

namespace multiply_exception_safety_test {

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

    TEST(MultiplyExceptionSafetyTest, TheOrdinaryPathStillComputesTheProduct) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Row-major 2x3 times 3x2. The buffers became unique_ptr in the same change, so this
        // case is the one that would notice the buffers no longer being passed through
        // correctly.
        const std::vector<float> a{1, 2, 3, 4, 5, 6};
        const std::vector<float> b{7, 8, 9, 10, 11, 12};
        std::vector<float> c(4, 0.0F);

        multiplyMatricesOnGPU(a.data(), b.data(), c.data(), 2, 3, 2);

        // [1 2 3] . [7 8; 9 10; 11 12] = [58 64]
        // [4 5 6]                        = [139 154]
        EXPECT_NEAR(c[0], 58.0F, 1e-3F);
        EXPECT_NEAR(c[1], 64.0F, 1e-3F);
        EXPECT_NEAR(c[2], 139.0F, 1e-3F);
        EXPECT_NEAR(c[3], 154.0F, 1e-3F);
    }

    TEST(MultiplyExceptionSafetyTest, ARequestTheDeviceCannotSatisfyThrows) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Build a request larger than the device's total memory, so the allocation chain must
        // fail rather than quietly return. This is the path that used to leak the buffers
        // allocated before the failure; the ownership change is what makes unwinding release
        // them, and it cannot be observed directly with cudaMemGetInfo (see the file comment).
        //
        // The dimensions are chosen from the reported total rather than a fixed size so the case
        // behaves the same on any GPU, and each must stay within int because the API takes int.
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
        (void)free_bytes;

        // A x B with M = K = N = 70000 needs three buffers of 70000^2 doubles = 39.2 GB each,
        // 117 GB in total: the third allocation cannot succeed on any current device, and the
        // first two must be released on the way out.
        constexpr int kDim = 70000;
        const size_t bytes_one = static_cast<size_t>(kDim) * kDim * sizeof(double);
        if (bytes_one * 3 <= total_bytes) {
            GTEST_SKIP() << "this device is large enough that the request would fit";
        }

        EXPECT_THROW(multiplyMatricesOnGPU<double>(nullptr, nullptr, nullptr, kDim, kDim, kDim), std::exception);

        // Leave the device usable for whatever runs next.
        (void)cudaGetLastError();
    }

}  // namespace multiply_exception_safety_test
