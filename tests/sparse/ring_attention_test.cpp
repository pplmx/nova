/**
 * @file ring_attention_test.cpp
 * @brief Contracts for the ring-attention device entry points
 *
 * include/cuda/distributed/sequence_parallel.h declares three host entry points
 * implemented in src/cuda/distributed/ring_attention.cu: ring_attn_init,
 * ring_attn_block and ring_attn_finalize. They stream a block of keys into a
 * running online softmax, which is how ring attention avoids materialising the
 * full score matrix.
 *
 * Two edge cases were broken:
 *
 * - ring_attn_init launched a zero-sized grid for local_seq == 0. CUDA rejects
 *   that with "invalid configuration argument", so initialising an empty shard
 *   threw — while ring_attn_block and ring_attn_finalize both guard the same
 *   case and return quietly.
 *
 * - finalize divides by running_l, the sum of exp(score - max). With no keys
 *   visited that sum stays 0 and every output element became NaN. Measured:
 *   "after finalize: out[0]=nan". A row with no keys has no weights to apply.
 *
 * The online-softmax maths itself is correct: the rescale-on-new-max step, the
 * accumulation, and the final normalisation all reproduce a direct computation.
 */

#include <cuda/distributed/sequence_parallel.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace cuda::distributed::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Owns device storage for one float array.
        class DeviceArray {
        public:
            explicit DeviceArray(size_t count)
                : count_(count) {
                cudaMalloc(&ptr_, count_ * sizeof(float));
                cudaMemset(ptr_, 0, count_ * sizeof(float));
            }

            explicit DeviceArray(const std::vector<float>& host)
                : count_(host.size()) {
                cudaMalloc(&ptr_, count_ * sizeof(float));
                cudaMemcpy(ptr_, host.data(), count_ * sizeof(float), cudaMemcpyHostToDevice);
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

    TEST(RingAttentionTest, ProducesTheSoftmaxWeightedValueAverage) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // One query, two keys, hidden 2. q = [1,0]; k0 = [1,0], k1 = [0,1]; so the
        // scores are 1 and 0 and the softmax weights are e/(e+1) and 1/(e+1).
        // v0 = [10,0], v1 = [0,20], giving out = [7.3106, 5.3789].
        const int seq = 1, keys = 2, hidden = 2;
        DeviceArray q(std::vector<float>{1.0f, 0.0f});
        DeviceArray k(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f});
        DeviceArray v(std::vector<float>{10.0f, 0.0f, 0.0f, 20.0f});
        DeviceArray out(static_cast<size_t>(seq) * hidden);
        DeviceArray running_max(seq);
        DeviceArray running_l(seq);

        ring_attn_init(out.data(), running_max.data(), running_l.data(), seq, hidden, nullptr);
        ring_attn_block(q.data(), k.data(), v.data(), out.data(), running_max.data(), running_l.data(), seq, keys, hidden, 1.0f, nullptr);
        ring_attn_finalize(out.data(), running_l.data(), seq, hidden, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto result = out.read();
        ASSERT_EQ(result.size(), 2u);
        const float e = std::exp(1.0f);
        EXPECT_NEAR(result[0], 10.0f * e / (e + 1.0f), 1e-4f);
        EXPECT_NEAR(result[1], 20.0f / (e + 1.0f), 1e-4f);
    }

    TEST(RingAttentionTest, AccumulatesAcrossMultipleBlocks) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Feeding the same keys in two calls must give what one call with both
        // gives: the online softmax is incrementally correct.
        const int seq = 1, hidden = 2;

        DeviceArray q(std::vector<float>{1.0f, 0.0f});
        DeviceArray k(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f});
        DeviceArray v(std::vector<float>{10.0f, 0.0f, 0.0f, 20.0f});

        // Single call with both keys.
        DeviceArray out_once(static_cast<size_t>(seq) * hidden);
        DeviceArray max_once(seq);
        DeviceArray l_once(seq);
        ring_attn_init(out_once.data(), max_once.data(), l_once.data(), seq, hidden, nullptr);
        ring_attn_block(q.data(), k.data(), v.data(), out_once.data(), max_once.data(), l_once.data(), seq, 2, hidden, 1.0f, nullptr);
        ring_attn_finalize(out_once.data(), l_once.data(), seq, hidden, nullptr);

        // Two calls, one key each.
        DeviceArray out_split(static_cast<size_t>(seq) * hidden);
        DeviceArray max_split(seq);
        DeviceArray l_split(seq);
        ring_attn_init(out_split.data(), max_split.data(), l_split.data(), seq, hidden, nullptr);
        ring_attn_block(q.data(), k.data(), v.data(), out_split.data(), max_split.data(), l_split.data(), seq, 1, hidden, 1.0f, nullptr);
        // The second key starts at offset hidden.
        ring_attn_block(q.data(), k.data() + hidden, v.data() + hidden, out_split.data(), max_split.data(), l_split.data(), seq, 1, hidden, 1.0f, nullptr);
        ring_attn_finalize(out_split.data(), l_split.data(), seq, hidden, nullptr);

        CUDA_CHECK(cudaDeviceSynchronize());

        const auto once = out_once.read();
        const auto split = out_split.read();
        ASSERT_EQ(once.size(), split.size());
        for (size_t i = 0; i < once.size(); ++i) {
            EXPECT_NEAR(once[i], split[i], 1e-4f) << "element " << i << " differs between one block and two";
        }
    }

    // Regression: a zero-sized grid was rejected with "invalid configuration
    // argument", while the other two entry points guard the same case.
    TEST(RingAttentionTest, InitWithEmptySequenceIsANoOp) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        EXPECT_NO_THROW(ring_attn_init(nullptr, nullptr, nullptr, 0, 8, nullptr));
        EXPECT_NO_THROW(ring_attn_finalize(nullptr, nullptr, 0, 8, nullptr));
        (void)cudaGetLastError();
    }

    // Regression: this produced NaN because the normaliser was zero.
    TEST(RingAttentionTest, FinalizeWithoutAnyKeysDoesNotProduceNaN) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const int seq = 2, hidden = 4;

        DeviceArray q(static_cast<size_t>(seq) * hidden);
        DeviceArray out(static_cast<size_t>(seq) * hidden);
        DeviceArray running_max(seq);
        DeviceArray running_l(seq);

        ring_attn_init(out.data(), running_max.data(), running_l.data(), seq, hidden, nullptr);

        // No key block is streamed in, so running_l stays zero.
        ring_attn_block(q.data(), q.data(), q.data(), out.data(), running_max.data(), running_l.data(), seq, 0, hidden, 1.0f, nullptr);
        ring_attn_finalize(out.data(), running_l.data(), seq, hidden, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto result = out.read();
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_TRUE(std::isfinite(result[i])) << "element " << i << " is " << result[i] << "; an empty key set must not divide by zero";
            EXPECT_FLOAT_EQ(result[i], 0.0f);
        }
    }

    TEST(RingAttentionTest, SingleKeyReturnsThatValue) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // With one key the softmax weight is 1, so the output is v itself.
        const int seq = 1, hidden = 3;
        DeviceArray q(std::vector<float>{0.0f, 0.0f, 0.0f});
        DeviceArray k(std::vector<float>{0.0f, 0.0f, 0.0f});
        DeviceArray v(std::vector<float>{1.5f, -2.5f, 3.5f});
        DeviceArray out(static_cast<size_t>(seq) * hidden);
        DeviceArray running_max(seq);
        DeviceArray running_l(seq);

        ring_attn_init(out.data(), running_max.data(), running_l.data(), seq, hidden, nullptr);
        ring_attn_block(q.data(), k.data(), v.data(), out.data(), running_max.data(), running_l.data(), seq, 1, hidden, 1.0f, nullptr);
        ring_attn_finalize(out.data(), running_l.data(), seq, hidden, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto result = out.read();
        ASSERT_EQ(result.size(), 3u);
        EXPECT_NEAR(result[0], 1.5f, 1e-5f);
        EXPECT_NEAR(result[1], -2.5f, 1e-5f);
        EXPECT_NEAR(result[2], 3.5f, 1e-5f);
    }

    TEST(RingAttentionTest, LargeScoresDoNotOverflow) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Scores around 100 would make exp(100) overflow without the running-max
        // trick; the online formulation subtracts the max first.
        const int seq = 1, keys = 2, hidden = 1;
        DeviceArray q(std::vector<float>{1.0f});
        DeviceArray k(std::vector<float>{200.0f, 100.0f});
        DeviceArray v(std::vector<float>{1.0f, 2.0f});
        DeviceArray out(static_cast<size_t>(seq) * hidden);
        DeviceArray running_max(seq);
        DeviceArray running_l(seq);

        const float inv_sqrt = 2.0f;  // scores become 400 and 200
        ring_attn_init(out.data(), running_max.data(), running_l.data(), seq, hidden, nullptr);
        ring_attn_block(q.data(), k.data(), v.data(), out.data(), running_max.data(), running_l.data(), seq, keys, hidden, inv_sqrt, nullptr);
        ring_attn_finalize(out.data(), running_l.data(), seq, hidden, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto result = out.read();
        ASSERT_EQ(result.size(), 1u);
        EXPECT_TRUE(std::isfinite(result[0])) << "got " << result[0];
        // exp(400-400)=1 dominates exp(200-400)~0, so the output is v0 = 1.
        EXPECT_NEAR(result[0], 1.0f, 1e-4f);
    }

}  // namespace cuda::distributed::test
