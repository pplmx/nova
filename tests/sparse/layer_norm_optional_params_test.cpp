/**
 * @file layer_norm_optional_params_test.cpp
 * @brief Two of layer_norm's pointers may be null and two may not
 *
 * The signature takes input, gamma, beta, output, mean, variance. Two of those are optional
 * and two are not, and neither the header nor the kernel made that visible:
 *
 *   gamma / beta      guarded by `if (gamma && beta)` in the kernel, so null is fine — but
 *                     only together. Passing one without the other silently skips the
 *                     affine step instead of applying half of it.
 *   mean / variance   written unconditionally, so null faults inside the kernel. Measured:
 *                     passing nullptr raises "an illegal memory access was encountered".
 *
 * The header now says so at each parameter. These cases pin the working half of that
 * contract; the null-buffer half would fault the process rather than fail an assertion, so it
 * is described in the header and not executed here.
 */

#include <cuda/neural/layer_norm.h>
#include <cuda/memory/buffer.h>
#include <cuda/stream/stream.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace layer_norm_optional_params_test {

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

    TEST(LayerNormOptionalParamsTest, OmittingGammaAndBetaSkipsTheAffineStep) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        constexpr int kRows = 2;
        constexpr int kCols = 8;
        const std::vector<float> host {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};
        cuda::memory::Buffer<float> input(kRows * kCols);
        cuda::memory::Buffer<float> output(kRows * kCols);
        cuda::memory::Buffer<float> mean(kRows);
        cuda::memory::Buffer<float> variance(kRows);
        input.copy_from(host.data(), host.size());
        cuda::stream::Stream stream;

        // gamma and beta omitted, mean and variance supplied — the combination the header now
        // documents as valid.
        ASSERT_NO_THROW(cuda::neural::layer_norm(input.data(), nullptr, nullptr, output.data(), mean.data(), variance.data(), kRows, kCols, 1e-5F, stream.get()));

        std::vector<float> result(kRows * kCols, -999.0F);
        output.copy_to(result.data(), result.size());

        // Without an affine step, each row normalises to zero mean and unit variance.
        for (int row = 0; row < kRows; ++row) {
            float sum = 0.0F;
            for (int col = 0; col < kCols; ++col) {
                sum += result[row * kCols + col];
            }
            EXPECT_NEAR(sum, 0.0F, 1e-3F) << "row " << row << " should be zero-mean";
        }
    }

    TEST(LayerNormOptionalParamsTest, AConstantRowNormalisesToZeros) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Zero variance: every element equals the mean, so each normalised value is 0 / sqrt(eps).
        // This is the boundary where a naive implementation divides by zero and produces NaN.
        constexpr int kCols = 16;
        const std::vector<float> host(kCols, 5.0F);
        cuda::memory::Buffer<float> input(kCols);
        cuda::memory::Buffer<float> output(kCols);
        cuda::memory::Buffer<float> mean(1);
        cuda::memory::Buffer<float> variance(1);
        input.copy_from(host.data(), host.size());
        cuda::stream::Stream stream;

        cuda::neural::layer_norm(input.data(), nullptr, nullptr, output.data(), mean.data(), variance.data(), 1, kCols, 1e-5F, stream.get());

        std::vector<float> result(kCols, -999.0F);
        output.copy_to(result.data(), result.size());
        for (int i = 0; i < kCols; ++i) {
            EXPECT_FALSE(std::isnan(result[i])) << "index " << i << ": eps should prevent NaN";
            EXPECT_NEAR(result[i], 0.0F, 1e-3F) << "index " << i;
        }
    }

    TEST(LayerNormOptionalParamsTest, TheMeanAndVarianceBuffersAreWritten) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // They are required rather than optional precisely because the kernel writes them; this
        // checks they contain the row statistics and not the sentinel they started with.
        constexpr int kCols = 4;
        const std::vector<float> host {2.0F, 4.0F, 6.0F, 8.0F};  // mean 5, variance 5
        cuda::memory::Buffer<float> input(kCols);
        cuda::memory::Buffer<float> output(kCols);
        cuda::memory::Buffer<float> mean(1);
        cuda::memory::Buffer<float> variance(1);
        input.copy_from(host.data(), host.size());
        cuda::stream::Stream stream;

        cuda::neural::layer_norm(input.data(), nullptr, nullptr, output.data(), mean.data(), variance.data(), 1, kCols, 1e-5F, stream.get());

        float host_mean = -999.0F;
        float host_variance = -999.0F;
        mean.copy_to(&host_mean, 1);
        variance.copy_to(&host_variance, 1);

        EXPECT_NEAR(host_mean, 5.0F, 1e-3F) << "the row mean should have been written";
        EXPECT_NEAR(host_variance, 5.0F, 1e-2F) << "the row variance should have been written";
    }

}  // namespace layer_norm_optional_params_test
