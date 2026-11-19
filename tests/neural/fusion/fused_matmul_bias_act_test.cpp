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

}  // namespace cuda::neural::fusion::test
