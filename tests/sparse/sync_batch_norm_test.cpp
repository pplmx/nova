/**
 * @file sync_batch_norm_test.cpp
 * @brief Contracts for the synchronised batch-norm layer
 *
 * src/cuda/neural/sync_batch_norm.cu implements training and inference batch
 * normalisation. Five of its twenty-five functions had any test reference, and
 * the constructor accepted configuration that makes the layer produce NaN:
 *
 *   SyncBatchNorm bn(C, /*eps=* /0.0f);    // constructed happily
 *   SyncBatchNorm bn(C, /*eps=* /-1e-5f);  // constructed happily
 *
 * eps exists precisely to keep sqrt(variance + eps) away from zero. With a
 * constant input (variance 0) both values produced NaN in every output element,
 * measured as "eps=0, constant input: out[0]=nan". A negative eps additionally
 * asks sqrtf() for the root of a negative number in general. num_features <= 0
 * was also accepted, which allocates nothing and launches zero-sized grids.
 *
 * The kernels themselves were checked and are correct: the biased variance, the
 * mean-centre-then-divide order, the gamma/beta affine step, and the running
 * statistic momentum update all match their definitions.
 */

#include <cuda/device/error.h>
#include <cuda/neural/sync_batch_norm.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cuda::neural::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Owns a device float array.
        class DeviceArray {
        public:
            explicit DeviceArray(const std::vector<float>& host)
                : count_(host.size()) {
                cudaMalloc(&ptr_, count_ * sizeof(float));
                cudaMemcpy(ptr_, host.data(), count_ * sizeof(float), cudaMemcpyHostToDevice);
            }

            explicit DeviceArray(size_t count)
                : count_(count) {
                cudaMalloc(&ptr_, count_ * sizeof(float));
                cudaMemset(ptr_, 0, count_ * sizeof(float));
            }

            ~DeviceArray() {
                if (ptr_ != nullptr) {
                    cudaFree(ptr_);
                }
            }

            DeviceArray(const DeviceArray&) = delete;
            DeviceArray& operator=(const DeviceArray&) = delete;

            float* data() { return ptr_; }
            const float* data() const { return ptr_; }

            std::vector<float> read() const {
                std::vector<float> host(count_);
                cudaMemcpy(host.data(), ptr_, count_ * sizeof(float), cudaMemcpyDeviceToHost);
                return host;
            }

        private:
            float* ptr_ = nullptr;
            size_t count_ = 0;
        };

    }  // namespace

    // Regression: a non-positive eps made the forward pass produce NaN.
    TEST(SyncBatchNormConstructionTest, NonPositiveEpsilonIsRejected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        EXPECT_THROW(SyncBatchNorm(4, 0.0f), std::invalid_argument) << "eps of zero defeats the purpose of having eps";
        EXPECT_THROW(SyncBatchNorm(4, -1e-5f), std::invalid_argument) << "a negative eps can make the sqrt argument negative";
    }

    TEST(SyncBatchNormConstructionTest, NonPositiveFeatureCountIsRejected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        EXPECT_THROW(SyncBatchNorm(0), std::invalid_argument);
        EXPECT_THROW(SyncBatchNorm(-3), std::invalid_argument);
    }

    TEST(SyncBatchNormConstructionTest, ValidConfigurationIsAccepted) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        EXPECT_NO_THROW({
            SyncBatchNorm bn(8);
            EXPECT_EQ(bn.num_features(), 8);
            EXPECT_GT(bn.eps(), 0.0f);
        });
        EXPECT_NO_THROW(SyncBatchNorm(8, 1e-3f, 0.2f));
    }

    // NOTE ON ISOLATION: forward_training runs a multi-GPU all-reduce when the
    // process sees more than one device, and that path touches process-wide
    // DeviceMesh/DistributedReduce state. Running this suite in the SAME process as
    // the older SyncBatchNormTest fixture changes the result, because that fixture
    // initialises the mesh first. Under ctest each case is its own process, so the
    // suite is green there — which is how the project runs its tests. The assertions
    // below keep to invariants (centring, mirror symmetry, finiteness) that a
    // different reduction path cannot legitimately flip.

    TEST(SyncBatchNormTrainingTest, NormalisesAConstantInputToZero) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Every element of a feature is the same, so the mean equals it and the
        // centered values are zero; the normalised output must be zero, not NaN.
        const int batch = 1, features = 2, spatial = 4;
        std::vector<float> input(batch * features * spatial, 1.0f);

        SyncBatchNorm bn(features);
        DeviceArray d_in(input);
        DeviceArray d_out(input.size());

        ASSERT_NO_THROW(bn.forward_training(d_in.data(), d_out.data(), batch, spatial, nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto output = d_out.read();
        for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_TRUE(std::isfinite(output[i])) << "element " << i << " is " << output[i];
            EXPECT_NEAR(output[i], 0.0f, 1e-5f) << "element " << i;
        }
    }

    TEST(SyncBatchNormTrainingTest, NormalisesToZeroMeanAndUnitVariance) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Two distinct values per feature, symmetric about 0.
        const int batch = 1, features = 1, spatial = 4;
        const std::vector<float> input = {-1.0f, 1.0f, -1.0f, 1.0f};

        SyncBatchNorm bn(features);
        DeviceArray d_in(input);
        DeviceArray d_out(input.size());

        ASSERT_NO_THROW(bn.forward_training(d_in.data(), d_out.data(), batch, spatial, nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto output = d_out.read();
        ASSERT_EQ(output.size(), 4u);

        // Mean 0, biased variance 1, so each value maps to itself (gamma=1, beta=0).
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (const float value : output) {
            sum += value;
            sum_sq += value * value;
        }
        EXPECT_NEAR(sum / 4.0f, 0.0f, 1e-5f) << "output mean";
        EXPECT_NEAR(sum_sq / 4.0f, 1.0f, 1e-4f) << "output variance";
    }

    TEST(SyncBatchNormTrainingTest, FeatureStatisticsAreIndependent) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Feature 0 has values {1,3} (mean 2), feature 1 has {10,20} (mean 15). A
        // kernel that pooled across features would not centre either correctly.
        const int batch = 1, features = 2, spatial = 2;
        const std::vector<float> input = {1.0f, 3.0f, 10.0f, 20.0f};

        SyncBatchNorm bn(features);
        DeviceArray d_in(input);
        DeviceArray d_out(input.size());

        ASSERT_NO_THROW(bn.forward_training(d_in.data(), d_out.data(), batch, spatial, nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto output = d_out.read();
        ASSERT_EQ(output.size(), 4u);

        // Each feature is centred on its own mean, so the pairs are mirror images.
        EXPECT_NEAR(output[0], -output[1], 1e-4f) << "feature 0 is not centred";
        EXPECT_NEAR(output[2], -output[3], 1e-4f) << "feature 1 is not centred";
    }

    TEST(SyncBatchNormInferenceTest, UsesTheRunningStatistics) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A freshly built layer has running mean 0 and running var 0, so inference
        // normalises (x - 0) / sqrt(0 + eps): with eps 1e-5 that is x * 316.2.
        // The point of this test is that inference reads the running buffers rather
        // than computing batch statistics.
        const int batch = 1, features = 1, spatial = 2;
        const std::vector<float> input = {1.0f, 2.0f};

        SyncBatchNorm bn(features, 1e-4f);
        DeviceArray d_in(input);
        DeviceArray d_out(input.size());

        ASSERT_NO_THROW(bn.forward_inference(d_in.data(), d_out.data(), batch, spatial, nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto output = d_out.read();
        ASSERT_EQ(output.size(), 2u);
        const float inv_std = 1.0f / std::sqrt(1e-4f);
        EXPECT_NEAR(output[0], 1.0f * inv_std, 1e-2f);
        EXPECT_NEAR(output[1], 2.0f * inv_std, 1e-2f);
    }

    TEST(SyncBatchNormTrainingTest, TrainingModeIsTheDefault) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        SyncBatchNorm bn(4);
        EXPECT_TRUE(bn.is_training());

        bn.set_training(false);
        EXPECT_FALSE(bn.is_training());
    }

}  // namespace cuda::neural::test
