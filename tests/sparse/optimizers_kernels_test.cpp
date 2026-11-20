/**
 * @file optimizers_kernels_test.cpp
 * @brief Contracts for the device AdamW / LAMB update kernels
 *
 * src/cuda/neural/optimizers/optimizers_kernels.cu implements the fused device
 * update steps. Nine functions, of which only one was referenced from any test;
 * adamw_step_device, lamb_step_device, clip_device and gradient_norm_device had
 * no direct coverage.
 *
 * The math was verified against hand-computed values before writing these tests,
 * and it is correct — AdamW's bias-corrected moments and weight decay, LAMB's
 * trust ratio with both clamping and the param == 0 guard, the L2 and Inf norms,
 * and clipping. What was missing was any regression net, so a future change to
 * the update rule could break training silently. That is what this adds.
 */

#include <cuda/neural/optimizers/optimizers.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace cuda::neural::optimizers::test {

    namespace {

        using NormType = GradientClipConfig::NormType;

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Owns one device scalar for a test.
        class DeviceScalar {
        public:
            explicit DeviceScalar(float value) {
                cudaMalloc(&ptr_, sizeof(float));
                cudaMemcpy(ptr_, &value, sizeof(float), cudaMemcpyHostToDevice);
            }

            ~DeviceScalar() {
                if (ptr_ != nullptr) {
                    cudaFree(ptr_);
                }
            }

            DeviceScalar(const DeviceScalar&) = delete;
            DeviceScalar& operator=(const DeviceScalar&) = delete;

            float* data() { return ptr_; }

            float read() const {
                float value = 0.0f;
                cudaMemcpy(&value, ptr_, sizeof(float), cudaMemcpyDeviceToHost);
                return value;
            }

        private:
            float* ptr_ = nullptr;
        };

        class DeviceVector {
        public:
            explicit DeviceVector(const std::vector<float>& host)
                : count_(host.size()) {
                cudaMalloc(&ptr_, count_ * sizeof(float));
                cudaMemcpy(ptr_, host.data(), count_ * sizeof(float), cudaMemcpyHostToDevice);
            }

            ~DeviceVector() {
                if (ptr_ != nullptr) {
                    cudaFree(ptr_);
                }
            }

            DeviceVector(const DeviceVector&) = delete;
            DeviceVector& operator=(const DeviceVector&) = delete;

            float* data() { return ptr_; }

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

    // The reference values in this suite were computed by hand from the update
    // rule; they are not recordings of whatever the kernel happened to print.
    TEST(AdamWStepTest, SingleStepMatchesTheHandComputedUpdate) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, param=1, grad=0.5,
        // moments zero. After one step:
        //   m = 0.05, v = 2.5e-4, m_hat = 0.5, v_hat = 0.25
        //   update = 0.5 / (0.5 + 1e-8) + 0.01 * 1 = 1.01
        //   param  = 1 - 0.1 * 1.01 = 0.899
        DeviceScalar param(1.0f);
        DeviceScalar grad(0.5f);
        DeviceScalar m(0.0f);
        DeviceScalar v(0.0f);

        detail::adamw_step_device(param.data(), grad.data(), m.data(), v.data(), 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f, nullptr);
        (void)cudaDeviceSynchronize();

        EXPECT_NEAR(param.read(), 0.899f, 1e-6f);
        EXPECT_NEAR(m.read(), 0.05f, 1e-7f);
        EXPECT_NEAR(v.read(), 2.5e-4f, 1e-8f);
    }

    TEST(AdamWStepTest, WeightDecayPullsEvenWithZeroGradient) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // With grad = 0 the moment terms vanish and only wd * param remains, so the
        // parameter must shrink by lr * wd * param.
        DeviceScalar param(2.0f);
        DeviceScalar grad(0.0f);
        DeviceScalar m(0.0f);
        DeviceScalar v(0.0f);

        detail::adamw_step_device(param.data(), grad.data(), m.data(), v.data(), 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.5f, 0.9f, 0.999f, nullptr);
        (void)cudaDeviceSynchronize();

        // The gradient term vanishes (m_hat/(sqrt(v_hat)+eps) = 0), leaving
        // update = wd*param = 0.5*2 = 1, so param = 2 - 0.1*1 = 1.9.
        EXPECT_NEAR(param.read(), 1.9f, 1e-6f) << "decoupled weight decay must apply without a gradient";
    }

    TEST(AdamWStepTest, AppliesToEveryElementOfAFullBuffer) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // More elements than one block, with a constant gradient, so every element
        // must end up at the same place.
        const std::vector<float> params(1000, 1.0f);
        const std::vector<float> grads(1000, 0.5f);
        const std::vector<float> zeros(1000, 0.0f);

        DeviceVector d_param(params);
        DeviceVector d_grad(grads);
        DeviceVector d_m(zeros);
        DeviceVector d_v(zeros);

        detail::adamw_step_device(d_param.data(), d_grad.data(), d_m.data(), d_v.data(), params.size(), 0.1f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f, nullptr);
        (void)cudaDeviceSynchronize();

        const auto result = d_param.read();
        ASSERT_EQ(result.size(), params.size());
        for (size_t i = 0; i < result.size(); ++i) {
            ASSERT_NEAR(result[i], 0.899f, 1e-6f) << "element " << i << " was not updated";
        }
    }

    TEST(LambStepTest, SingleStepWithLayerAdaptationStaysFinite) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        DeviceScalar param(1.0f);
        DeviceScalar grad(0.5f);
        DeviceScalar m(0.0f);
        DeviceScalar v(0.0f);

        detail::lamb_step_device(param.data(), grad.data(), m.data(), v.data(), 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f, 1.0f, 10.0f, true, nullptr);
        (void)cudaDeviceSynchronize();

        EXPECT_TRUE(std::isfinite(param.read()));
        EXPECT_NEAR(m.read(), 0.05f, 1e-7f);
        EXPECT_NEAR(v.read(), 2.5e-4f, 1e-8f);
    }

    // The trust ratio is |param| / |update|. A zero update would divide by zero, so
    // the kernel must keep the ratio finite for every combination that reaches it.
    TEST(LambStepTest, ZeroUpdateAndZeroParamStaysFinite) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        DeviceScalar param(0.0f);
        DeviceScalar grad(0.0f);
        DeviceScalar m(0.0f);
        DeviceScalar v(0.0f);

        detail::lamb_step_device(param.data(), grad.data(), m.data(), v.data(), 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.9f, 0.999f, 1.0f, 10.0f, true, nullptr);
        (void)cudaDeviceSynchronize();

        EXPECT_TRUE(std::isfinite(param.read())) << "a zero parameter with a zero update must not produce inf or NaN";
        EXPECT_FLOAT_EQ(param.read(), 0.0f) << "nothing to update, nothing moves";
    }

    TEST(LambStepTest, ZeroUpdateWithNonzeroParamLeavesItAlone) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // grad = 0 and wd = 0 make the update exactly zero while the parameter is
        // non-zero, which is the other way to reach the 0/0 in the trust ratio.
        DeviceScalar param(1.0f);
        DeviceScalar grad(0.0f);
        DeviceScalar m(0.0f);
        DeviceScalar v(0.0f);

        detail::lamb_step_device(param.data(), grad.data(), m.data(), v.data(), 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.9f, 0.999f, 1.0f, 10.0f, true, nullptr);
        (void)cudaDeviceSynchronize();

        EXPECT_TRUE(std::isfinite(param.read()));
        EXPECT_FLOAT_EQ(param.read(), 1.0f) << "a zero update must not move the parameter regardless of the ratio";
    }

    TEST(LambStepTest, LayerAdaptationCanBeDisabled) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // With layer adaptation off the trust ratio is 1, so the step is lr * rtw.
        DeviceScalar param(1.0f);
        DeviceScalar grad(0.5f);
        DeviceScalar m(0.0f);
        DeviceScalar v(0.0f);

        detail::lamb_step_device(param.data(), grad.data(), m.data(), v.data(), 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f, 1.0f, 10.0f, false, nullptr);
        (void)cudaDeviceSynchronize();

        EXPECT_TRUE(std::isfinite(param.read()));
        EXPECT_LT(param.read(), 1.0f) << "the update should still be applied";
    }

    TEST(GradientNormTest, L2NormOfOnesIsTheSquareRootOfTheCount) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<float> grads(1000, 1.0f);
        DeviceVector d_grads(grads);

        const float norm = detail::gradient_norm_device(d_grads.data(), grads.size(), NormType::L2, nullptr);
        EXPECT_NEAR(norm, std::sqrt(1000.0f), 1e-3f);
    }

    TEST(GradientNormTest, InfNormIsTheLargestMagnitude) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // The largest magnitude is 5, appearing as a negative value; a norm that
        // forgot the absolute value would report 2 or -5.
        const std::vector<float> grads = {1.0f, -3.0f, 2.0f, -5.0f, 0.5f};
        DeviceVector d_grads(grads);

        const float norm = detail::gradient_norm_device(d_grads.data(), grads.size(), NormType::Inf, nullptr);
        EXPECT_FLOAT_EQ(norm, 5.0f);
    }

    TEST(GradientNormTest, EmptyInputIsZero) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        EXPECT_FLOAT_EQ(detail::gradient_norm_device(nullptr, 0, NormType::L2, nullptr), 0.0f);
        EXPECT_FLOAT_EQ(detail::gradient_norm_device(nullptr, 0, NormType::Inf, nullptr), 0.0f);
    }

    TEST(GradientNormTest, NormIsUnaffectedByABusyStream) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // The kernel launches on the caller's stream and the partial sums are copied
        // back with a blocking copy, so a busy stream must not change the answer.
        const std::vector<float> grads(1000, 1.0f);
        DeviceVector d_grads(grads);

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        const float norm = detail::gradient_norm_device(d_grads.data(), grads.size(), NormType::L2, stream);
        cudaStreamDestroy(stream);

        EXPECT_NEAR(norm, std::sqrt(1000.0f), 1e-3f);
    }

    TEST(ClipDeviceTest, ScalesEveryGradient) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<float> grads = {1.0f, -2.0f, 3.0f, -4.0f};
        DeviceVector d_grads(grads);

        detail::clip_device(d_grads.data(), grads.size(), 0.5f, nullptr);
        (void)cudaDeviceSynchronize();

        const auto result = d_grads.read();
        ASSERT_EQ(result.size(), grads.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_FLOAT_EQ(result[i], grads[i] * 0.5f) << "element " << i;
        }
    }

    TEST(ClipDeviceTest, AScaleOfOneIsANoOp) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<float> grads = {0.25f, -0.75f};
        DeviceVector d_grads(grads);

        detail::clip_device(d_grads.data(), grads.size(), 1.0f, nullptr);
        (void)cudaDeviceSynchronize();

        EXPECT_EQ(d_grads.read(), grads);
    }

}  // namespace cuda::neural::optimizers::test
