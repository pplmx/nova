#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/neural/fusion/fused_matmul_bias_act.h>
#include <cuda/neural/fusion/kernel_fusion.h>

namespace cuda::neural::fusion::test {

class FusionEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDevice(&device_);
    }

    int device_ = 0;
};

TEST_F(FusionEdgeCaseTest, FusionPolicyMinElements) {
    auto& manager = FusionPolicyManager::instance();

    FusionPolicy policy;
    policy.min_elements_for_fusion = 1000;
    manager.set_policy(policy);

    EXPECT_FALSE(manager.should_fuse("matmul_bias", 500));
    EXPECT_TRUE(manager.should_fuse("matmul_bias", 2000));
}

TEST_F(FusionEdgeCaseTest, FusionPolicyMaxDepth) {
    auto& manager = FusionPolicyManager::instance();

    FusionPolicy policy;
    policy.max_fusion_depth = 2;
    manager.set_policy(policy);

    auto retrieved = manager.get_policy();
    EXPECT_EQ(retrieved.max_fusion_depth, 2);
}

TEST_F(FusionEdgeCaseTest, PolicySummaryFormat) {
    auto& manager = FusionPolicyManager::instance();

    FusionPolicy policy;
    policy.fuse_matmul_bias = true;
    policy.fuse_matmul_bias_activation = false;
    policy.fuse_layernorm_softmax = true;
    manager.set_policy(policy);

    std::string summary = manager.get_policy_summary();
    EXPECT_TRUE(summary.find("FusionPolicy") != std::string::npos);
}

TEST_F(FusionEdgeCaseTest, ReentrantPolicyChanges) {
    auto& manager = FusionPolicyManager::instance();

    for (int i = 0; i < 5; ++i) {
        FusionPolicy policy;
        policy.fuse_matmul_bias = (i % 2 == 0);
        policy.fuse_matmul_bias_activation = true;
        policy.fuse_layernorm_softmax = false;
        manager.set_policy(policy);

        auto retrieved = manager.get_policy();
        EXPECT_EQ(retrieved.fuse_matmul_bias, (i % 2 == 0));
    }
}

TEST_F(FusionEdgeCaseTest, FusionPolicyAllOperations) {
    auto& manager = FusionPolicyManager::instance();

    manager.enable_op("matmul_bias");
    manager.enable_op("matmul_bias_activation");
    manager.enable_op("layernorm_softmax");

    EXPECT_TRUE(manager.should_fuse("matmul_bias", 1000));
    EXPECT_TRUE(manager.should_fuse("matmul_bias_activation", 1000));
    EXPECT_TRUE(manager.should_fuse("layernorm_softmax", 1000));

    manager.disable_op("matmul_bias");
    manager.disable_op("matmul_bias_activation");
    manager.disable_op("layernorm_softmax");

    EXPECT_FALSE(manager.should_fuse("matmul_bias", 1000));
    EXPECT_FALSE(manager.should_fuse("matmul_bias_activation", 1000));
    EXPECT_FALSE(manager.should_fuse("layernorm_softmax", 1000));
}

TEST_F(FusionEdgeCaseTest, ActivationTypeEnumValues) {
    EXPECT_EQ(static_cast<int>(ActivationType::None), 0);
    EXPECT_EQ(static_cast<int>(ActivationType::ReLU), 1);
    EXPECT_EQ(static_cast<int>(ActivationType::Sigmoid), 2);
    EXPECT_EQ(static_cast<int>(ActivationType::Tanh), 3);
    EXPECT_EQ(static_cast<int>(ActivationType::GELU), 4);
}

}  // namespace cuda::neural::fusion::test
