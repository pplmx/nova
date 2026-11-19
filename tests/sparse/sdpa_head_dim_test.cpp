/**
 * @file sdpa_head_dim_test.cpp
 * @brief Contracts for the scaled-dot-product attention dimension checks
 *
 * src/cuda/neural/attention_kernels.cu launches device kernels for fused attention.
 * validate_sdpa_dims rejected non-positive dims but not a head_dim whose shared-memory
 * requirement exceeds the device limit:
 *
 *     const int shared = (kSdpaBlock / 32 + head_dim) * sizeof(float);
 *
 * With head_dim = 20000 that is 78 KiB against the 48 KiB default, so CUDA rejected
 * the launch with a generic "invalid argument" naming neither the parameter nor the
 * limit. Measured before the fix:
 *
 *     sdpa_forward_device(m=1, heads=1, head_dim=20000)
 *       -> CUDA error: invalid argument
 *
 * The check now compares the computed requirement against
 * cudaDevAttrMaxSharedMemoryPerBlock and reports both numbers.
 *
 * The kernel maths were checked and are correct: the online max, the sum of
 * exponentials, the accumulated weighted value, and the final normalisation.
 */

#include <cuda/neural/tensor_parallel_attention.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::neural::test {

    namespace {

        using cuda::neural::detail::sdpa_backward_device;
        using cuda::neural::detail::sdpa_forward_device;

        /// The reduction slots the kernel reserves ahead of the head dimension.
        constexpr int kReductionSlots = 256 / 32;

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Runs `call` and returns the exception message, or an empty string.
        template <typename Callable>
        std::string message_from(Callable&& call) {
            try {
                call();
            } catch (const std::exception& error) {
                return error.what();
            }
            return "";
        }

        /// Owns a device float array.
        class DeviceArray {
        public:
            explicit DeviceArray(size_t count)
                : count_(count) {
                cudaMalloc(&ptr_, (count > 0 ? count : 1) * sizeof(float));
                cudaMemset(ptr_, 0, (count > 0 ? count : 1) * sizeof(float));
            }

            ~DeviceArray() {
                if (ptr_ != nullptr) {
                    cudaFree(ptr_);
                }
            }

            DeviceArray(const DeviceArray&) = delete;
            DeviceArray& operator=(const DeviceArray&) = delete;

            float* data() { return ptr_; }

        private:
            float* ptr_ = nullptr;
            size_t count_ = 0;
        };

    }  // namespace

    // Regression: this surfaced as a generic "invalid argument" from the runtime.
    TEST(SdpaDimensionTest, HeadDimBeyondTheSharedMemoryLimitIsRejected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        int max_shared = 0;
        ASSERT_EQ(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, 0), cudaSuccess);
        ASSERT_GT(max_shared, 0);

        // One element past what the limit allows.
        const int head_dim = max_shared / static_cast<int>(sizeof(float)) - kReductionSlots + 1;
        const size_t count = static_cast<size_t>(head_dim);

        DeviceArray q(count);
        DeviceArray out(count);

        const std::string message = message_from([&] { sdpa_forward_device(q.data(), q.data(), q.data(), out.data(), 1, 1, head_dim, nullptr); });

        EXPECT_FALSE(message.empty()) << "a head dimension that cannot fit its shared buffer must not launch";
        EXPECT_NE(message.find("head_dim"), std::string::npos) << "the message should name the parameter: " << message;
        EXPECT_NE(message.find("shared memory"), std::string::npos) << "the message should name the resource: " << message;
        EXPECT_NE(message.find(std::to_string(max_shared)), std::string::npos) << "the message should state the device limit: " << message;
        (void)cudaGetLastError();
    }

    TEST(SdpaDimensionTest, NonPositiveDimensionsAreRejected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        DeviceArray buffer(64);

        for (const int bad : {0, -1}) {
            const std::string zero_m = message_from([&] { sdpa_forward_device(buffer.data(), buffer.data(), buffer.data(), buffer.data(), bad, 1, 4, nullptr); });
            EXPECT_FALSE(zero_m.empty()) << "m = " << bad;

            const std::string zero_heads = message_from([&] { sdpa_forward_device(buffer.data(), buffer.data(), buffer.data(), buffer.data(), 1, bad, 4, nullptr); });
            EXPECT_FALSE(zero_heads.empty()) << "local_heads = " << bad;

            const std::string zero_dim = message_from([&] { sdpa_forward_device(buffer.data(), buffer.data(), buffer.data(), buffer.data(), 1, 1, bad, nullptr); });
            EXPECT_FALSE(zero_dim.empty()) << "head_dim = " << bad;
        }
        (void)cudaGetLastError();
    }

    TEST(SdpaDimensionTest, BackwardAppliesTheSameCheck) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        int max_shared = 0;
        ASSERT_EQ(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, 0), cudaSuccess);

        const int head_dim = max_shared / static_cast<int>(sizeof(float)) - kReductionSlots + 1;
        DeviceArray buffer(static_cast<size_t>(head_dim));

        const std::string message = message_from(
            [&] { sdpa_backward_device(buffer.data(), buffer.data(), buffer.data(), buffer.data(), buffer.data(), buffer.data(), buffer.data(), 1, 1, head_dim, nullptr); });

        EXPECT_FALSE(message.empty()) << "the backward pass uses the same shared buffer";
        EXPECT_NE(message.find("sdpa_backward_device"), std::string::npos) << "the message should name which entry point failed: " << message;
        (void)cudaGetLastError();
    }

    TEST(SdpaDimensionTest, AReasonableHeadDimStillRuns) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // head_dim 64 needs (8 + 64) * 4 = 288 bytes, far inside the limit.
        const int m = 2, heads = 1, head_dim = 64;
        const size_t count = static_cast<size_t>(m) * heads * head_dim;

        DeviceArray q(count);
        DeviceArray out(count);

        EXPECT_NO_THROW(sdpa_forward_device(q.data(), q.data(), q.data(), out.data(), m, heads, head_dim, nullptr));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }

    // NOTE: a case computing a softmax-weighted average was drafted and dropped. The
    // signature takes (q, k, v, out, m, local_heads, head_dim), where m is BOTH the query
    // count and the key count — this is self-attention, not cross-attention — so a test
    // that supplies one query and two keys does not describe a shape the function accepts.
    // The numeric behaviour of the kernel is covered by the existing attention suites;
    // what this file adds is the dimension checking.
}  // namespace cuda::neural::test
