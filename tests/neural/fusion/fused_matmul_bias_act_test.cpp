#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/neural/fusion/fused_matmul_bias_act.h>
#include <cuda/neural/fusion/kernel_fusion.h>
#include <cuda/memory/buffer.h>
#include <cuda/device/error.h>
#include <cmath>

namespace cuda::neural::fusion::test {

    class FusedMatmulBiasActTest : public ::testing::Test {
    protected:
        void SetUp() override {
            cudaGetDevice(&device_);
            cudaStreamCreate(&stream_);
        }

        void TearDown() override { cudaStreamDestroy(stream_); }

        int device_ = 0;
        cudaStream_t stream_ = nullptr;
    };

    TEST_F(FusedMatmulBiasActTest, BasicMatmulBiasRelu) {
        constexpr int m = 128;
        constexpr int n = 256;
        constexpr int k = 64;

        cuda::memory::Buffer<float> A(m * k);
        cuda::memory::Buffer<float> B(k * n);
        cuda::memory::Buffer<float> bias(n);
        cuda::memory::Buffer<float> C(m * n);

        A.fill(1.0f);
        B.fill(2.0f);
        bias.fill(0.5f);
        C.fill(0.0f);

        MatmulBiasActConfig config;
        config.activation = ActivationType::ReLU;
        FusedMatmulBiasAct fused(config);

        fused.forward(A.data(), B.data(), bias.data(), C.data(), m, n, k, stream_);

        cudaStreamSynchronize(stream_);

        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }

    TEST_F(FusedMatmulBiasActTest, MatmulBiasSigmoid) {
        constexpr int m = 64;
        constexpr int n = 128;
        constexpr int k = 32;

        cuda::memory::Buffer<float> A(m * k);
        cuda::memory::Buffer<float> B(k * n);
        cuda::memory::Buffer<float> bias(n);
        cuda::memory::Buffer<float> C(m * n);

        A.fill(0.5f);
        B.fill(0.5f);
        bias.fill(0.0f);
        C.fill(0.0f);

        MatmulBiasActConfig config;
        config.activation = ActivationType::Sigmoid;
        FusedMatmulBiasAct fused(config);

        fused.forward(A.data(), B.data(), bias.data(), C.data(), m, n, k, stream_);

        cudaStreamSynchronize(stream_);

        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }

    TEST_F(FusedMatmulBiasActTest, MatmulBiasGELU) {
        constexpr int m = 32;
        constexpr int n = 64;
        constexpr int k = 16;

        cuda::memory::Buffer<float> A(m * k);
        cuda::memory::Buffer<float> B(k * n);
        cuda::memory::Buffer<float> bias(n);
        cuda::memory::Buffer<float> C(m * n);

        A.fill(0.1f);
        B.fill(0.2f);
        bias.fill(0.0f);
        C.fill(0.0f);

        MatmulBiasActConfig config;
        config.activation = ActivationType::GELU;
        FusedMatmulBiasAct fused(config);

        fused.forward(A.data(), B.data(), bias.data(), C.data(), m, n, k, stream_);

        cudaStreamSynchronize(stream_);

        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }

    TEST_F(FusedMatmulBiasActTest, FusionPolicyManager) {
        auto& manager = FusionPolicyManager::instance();

        FusionPolicy policy;
        policy.fuse_matmul_bias = true;
        policy.fuse_matmul_bias_activation = true;
        policy.fuse_layernorm_softmax = false;
        manager.set_policy(policy);

        auto retrieved_policy = manager.get_policy();
        EXPECT_TRUE(retrieved_policy.fuse_matmul_bias);
        EXPECT_TRUE(retrieved_policy.fuse_matmul_bias_activation);
        EXPECT_FALSE(retrieved_policy.fuse_layernorm_softmax);

        EXPECT_TRUE(manager.should_fuse("matmul_bias", 1024));
        EXPECT_TRUE(manager.should_fuse("matmul_bias_activation", 1024));
        EXPECT_FALSE(manager.should_fuse("layernorm_softmax", 1024));

        EXPECT_FALSE(manager.should_fuse("matmul_bias", 32));
    }

    TEST_F(FusedMatmulBiasActTest, ActivationTypes) {
        MatmulBiasActConfig config;

        config.activation = ActivationType::None;
        EXPECT_EQ(config.activation, ActivationType::None);

        config.activation = ActivationType::ReLU;
        EXPECT_EQ(config.activation, ActivationType::ReLU);

        config.activation = ActivationType::Sigmoid;
        EXPECT_EQ(config.activation, ActivationType::Sigmoid);

        config.activation = ActivationType::Tanh;
        EXPECT_EQ(config.activation, ActivationType::Tanh);

        config.activation = ActivationType::GELU;
        EXPECT_EQ(config.activation, ActivationType::GELU);
    }

    TEST_F(FusedMatmulBiasActTest, CudaFusionToggle) {
        MatmulBiasActConfig config;
        config.use_cuda_fusion = false;

        FusedMatmulBiasAct fused(config);

        EXPECT_FALSE(fused.is_cuda_fusion_enabled());

        fused.disable_cuda_fusion();
        EXPECT_FALSE(fused.is_cuda_fusion_enabled());
    }

    // forward() used to branch on config_.use_cuda_fusion && cuda_fusion_available_, and both arms
    // called forward_fallback. Enabling fusion created a cublasLt handle that nothing used --
    // cublasLtMatmul is called nowhere in this file -- and produced identical results. Measured
    // before the change: C[0] = 4.500 with fusion on and with it off, the only difference being
    // is_cuda_fusion_enabled() reporting true.
    //
    // Scope, stated plainly: restoring the old two-arm branch leaves this case green, because both
    // arms produced identical results -- that is exactly the defect. What it does check is that the
    // product is right (4.5 = 1*2*2 + 0.5, ReLU no-op) with the flag in both positions, so a future
    // change that makes the flag matter in a WRONG way fails here. The removal of the branch itself
    // is not testable from outside; the comment on enable_cuda_fusion() carries that half.
    TEST_F(FusedMatmulBiasActTest, FusionFlagDoesNotChangeTheResult) {
        const int m = 4, n = 3, k = 2;

        std::vector<float> h_A(m * k, 1.0f);
        std::vector<float> h_B(k * n, 2.0f);
        std::vector<float> h_bias(n, 0.5f);

        cuda::memory::Buffer<float> d_A(m * k);
        cuda::memory::Buffer<float> d_B(k * n);
        cuda::memory::Buffer<float> d_bias(n);
        cuda::memory::Buffer<float> d_C(m * n);
        d_A.copy_from(h_A.data(), h_A.size());
        d_B.copy_from(h_B.data(), h_B.size());
        d_bias.copy_from(h_bias.data(), h_bias.size());

        std::vector<float> without_fusion;
        std::vector<float> with_fusion;

        for (int pass = 0; pass < 2; ++pass) {
            const bool fusion = pass == 1;
            MatmulBiasActConfig config;
            config.use_cuda_fusion = fusion;
            config.activation = ActivationType::ReLU;

            FusedMatmulBiasAct op(config);

            CUDA_CHECK(cudaMemset(d_C.data(), 0, m * n * sizeof(float)));
            op.forward(d_A.data(), d_B.data(), d_bias.data(), d_C.data(), m, n, k, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> out(m * n, 0.0f);
            d_C.copy_to(out.data(), out.size());
            (fusion ? with_fusion : without_fusion) = out;
        }

        ASSERT_EQ(without_fusion.size(), with_fusion.size());
        for (size_t i = 0; i < without_fusion.size(); ++i) {
            EXPECT_FLOAT_EQ(without_fusion[i], with_fusion[i]) << "element " << i << ": the fusion flag changed the output";
        }
        // And the values are the right ones: 1*2 summed over k=2 is 4, plus bias 0.5, ReLU keeps it.
        EXPECT_FLOAT_EQ(without_fusion[0], 4.5f);
    }

}  // namespace cuda::neural::fusion::test
