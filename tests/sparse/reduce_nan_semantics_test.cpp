/**
 * @file reduce_nan_semantics_test.cpp
 * @brief The reduce family disagrees about NaN, and the header now says so
 *
 * Asked what an all-NaN input reduces to, the three functions answer differently:
 *
 *   reduce_sum  nan    because 0 + NaN is NaN
 *   reduce_max  -inf   because it combines with max(), which returns the non-NaN operand, so
 *                      every NaN is dropped and the identity survives
 *   reduce_min  +inf   the same
 *
 * Each follows IEEE 754; the inconsistency is between them, not within any one. include/
 * cuda/algo/reduce.h said nothing about NaN at all, which matters because the max and min
 * answers look like measurements.
 *
 * The behaviour is pinned rather than changed: fmax/fmin ignoring NaN is the standard's
 * choice, and switching to NaN-propagating comparisons would be a different semantic, not a
 * bug fix. A caller who needs to detect NaN must check for it separately.
 */

#include <cuda/algo/reduce.h>
#include <cuda/memory/buffer.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <vector>

namespace reduce_nan_semantics_test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

cuda::memory::Buffer<float> stage(const std::vector<float>& host) {
    cuda::memory::Buffer<float> device(host.size());
    if (!host.empty()) {
        device.copy_from(host.data(), host.size());
    }
    return device;
}

}  // namespace

TEST(ReduceNanSemanticsTest, AnAllNanInputSumsToNan) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const auto device = stage({nan_value, nan_value});

    EXPECT_TRUE(std::isnan(cuda::algo::reduce_sum(device.data(), 2)))
        << "sum propagates NaN because 0 + NaN is NaN";
}

TEST(ReduceNanSemanticsTest, AnAllNanInputMaximisesToNegativeInfinity) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const auto device = stage({nan_value, nan_value});

    const float result = cuda::algo::reduce_max(device.data(), 2);
    // Not NaN: max() drops it. Not a real maximum either: it is the identity, which the
    // header now warns about.
    EXPECT_TRUE(std::isinf(result));
    EXPECT_LT(result, 0.0F) << "the documented identity for max is negative infinity";
}

TEST(ReduceNanSemanticsTest, AnAllNanInputMinimisesToPositiveInfinity) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const auto device = stage({nan_value, nan_value});

    const float result = cuda::algo::reduce_min(device.data(), 2);
    EXPECT_TRUE(std::isinf(result));
    EXPECT_GT(result, 0.0F) << "the documented identity for min is positive infinity";
}

TEST(ReduceNanSemanticsTest, OneNanAmongRealValuesStillSumsToNanButIsIgnoredByMax) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const auto device = stage({1.0F, nan_value, 3.0F, 2.0F});

    // The two treatments side by side, which is what the header's note describes.
    EXPECT_TRUE(std::isnan(cuda::algo::reduce_sum(device.data(), 4)));
    EXPECT_FLOAT_EQ(cuda::algo::reduce_max(device.data(), 4), 3.0F)
        << "max skips the NaN and finds the real maximum";
    EXPECT_FLOAT_EQ(cuda::algo::reduce_min(device.data(), 4), 1.0F);
}

TEST(ReduceNanSemanticsTest, AnEmptyBufferIsTheIdentityForAllThree) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::memory::Buffer<float> empty(0);

    // Distinct from the NaN cases above: here there really is no data, and the identity is
    // the right answer for every operation.
    EXPECT_FLOAT_EQ(cuda::algo::reduce_sum(empty.data(), 0), 0.0F);
    EXPECT_FLOAT_EQ(cuda::algo::reduce_max(empty.data(), 0), 0.0F);
    EXPECT_FLOAT_EQ(cuda::algo::reduce_min(empty.data(), 0), 0.0F);
}

}  // namespace reduce_nan_semantics_test
