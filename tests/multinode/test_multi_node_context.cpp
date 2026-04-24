/**
 * @file test_multi_node_context.cpp
 * @brief Unit tests for multi-node context and hierarchical collectives
 *
 * Tests MultiNodeContext, HierarchicalAllReduce, and HierarchicalBarrier.
 * Tests are skipped if NCCL is not enabled or multi-node not available.
 */

#include <gtest/gtest.h>

#include "cuda/multinode/multi_node_context.h"

namespace cuda::multinode {

class MultiNodeContextTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
        auto& ctx = MultiNodeContext::instance();
        if (ctx.is_initialized()) {
            ctx.shutdown();
        }
    }
};

TEST_F(MultiNodeContextTest, GetInstanceReturnsSameObject) {
    auto& instance1 = MultiNodeContext::get_instance();
    auto& instance2 = MultiNodeContext::get_instance();
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(MultiNodeContextTest, InstanceMethodDelegatesToGetInstance) {
    auto& instance1 = MultiNodeContext::instance();
    auto& instance2 = MultiNodeContext::get_instance();
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(MultiNodeContextTest, DefaultStateNotInitialized) {
    auto& ctx = MultiNodeContext::instance();
    EXPECT_FALSE(ctx.is_initialized());
}

TEST_F(MultiNodeContextTest, SingleNodeInitialization) {
    auto& ctx = MultiNodeContext::instance();

    MultiNodeContext::NodeConfig config;
    config.world_rank = 0;
    config.world_size = 1;
    config.local_rank = 0;
    config.local_size = 1;
    config.node_id = 0;
    config.hostname = "localhost";

    ctx.initialize(config);

    EXPECT_TRUE(ctx.is_initialized());
    EXPECT_EQ(ctx.global_rank(), 0);
    EXPECT_EQ(ctx.global_size(), 1);
    EXPECT_EQ(ctx.local_rank(), 0);
    EXPECT_EQ(ctx.local_size(), 1);
    EXPECT_EQ(ctx.num_nodes(), 1);
}

TEST_F(MultiNodeContextTest, MainNodeIdentification) {
    auto& ctx = MultiNodeContext::instance();

    MultiNodeContext::NodeConfig config;
    config.world_rank = 0;
    config.world_size = 8;
    config.local_rank = 0;
    config.local_size = 4;
    config.node_id = 0;
    config.hostname = "node0";

    ctx.initialize(config);

    EXPECT_TRUE(ctx.is_main_node());
    EXPECT_TRUE(ctx.is_main_process());
}

TEST_F(MultiNodeContextTest, NonMainNodeIdentification) {
    auto& ctx = MultiNodeContext::instance();

    MultiNodeContext::NodeConfig config;
    config.world_rank = 4;
    config.world_size = 8;
    config.local_rank = 0;
    config.local_size = 4;
    config.node_id = 1;
    config.hostname = "node1";

    ctx.initialize(config);

    EXPECT_FALSE(ctx.is_main_node());
    EXPECT_FALSE(ctx.is_main_process());
}

TEST_F(MultiNodeContextTest, NumNodesCalculation) {
    auto& ctx = MultiNodeContext::instance();

    MultiNodeContext::NodeConfig config;
    config.world_rank = 0;
    config.world_size = 4;
    config.local_rank = 0;
    config.local_size = 2;
    config.node_id = 0;
    config.hostname = "node0";

    ctx.initialize(config);

#if NOVA_NCCL_ENABLED
    EXPECT_EQ(ctx.num_nodes(), 2);
#else
    EXPECT_EQ(ctx.num_nodes(), 1);
#endif
}

TEST_F(MultiNodeContextTest, DoubleInitializeIsSafe) {
    auto& ctx = MultiNodeContext::instance();

    MultiNodeContext::NodeConfig config;
    config.world_rank = 0;
    config.world_size = 1;
    config.local_rank = 0;
    config.local_size = 1;
    config.node_id = 0;
    config.hostname = "localhost";

    ctx.initialize(config);
    EXPECT_NO_THROW(ctx.initialize(config));
}

TEST_F(MultiNodeContextTest, ShutdownResetsState) {
    auto& ctx = MultiNodeContext::instance();

    MultiNodeContext::NodeConfig config;
    config.world_rank = 0;
    config.world_size = 1;
    config.local_rank = 0;
    config.local_size = 1;
    config.node_id = 0;
    config.hostname = "localhost";

    ctx.initialize(config);
    EXPECT_TRUE(ctx.is_initialized());

    ctx.shutdown();
    EXPECT_FALSE(ctx.is_initialized());
}

TEST_F(MultiNodeContextTest, StreamsCreatedOnInitialization) {
    auto& ctx = MultiNodeContext::instance();

    MultiNodeContext::NodeConfig config;
    config.world_rank = 0;
    config.world_size = 1;
    config.local_rank = 0;
    config.local_size = 1;
    config.node_id = 0;
    config.hostname = "localhost";

    ctx.initialize(config);

#if NOVA_NCCL_ENABLED
    EXPECT_NE(ctx.get_local_stream(), nullptr);
    EXPECT_NE(ctx.get_global_stream(), nullptr);
#else
    GTEST_SKIP() << "NCCL not enabled - streams not created in fallback mode";
#endif
}

class HierarchicalAllReduceTest : public ::testing::Test {
protected:
};

TEST_F(HierarchicalAllReduceTest, Construction) {
#if NOVA_NCCL_ENABLED
    HierarchicalAllReduce reducer(nullptr, nullptr, nullptr, nullptr);
    EXPECT_NO_THROW(reducer.synchronize());
#else
    GTEST_SKIP() << "NCCL not enabled";
#endif
}

TEST_F(HierarchicalAllReduceTest, AllReduceWithNullComms) {
#if NOVA_NCCL_ENABLED
    HierarchicalAllReduce reducer(nullptr, nullptr, nullptr, nullptr);

    std::vector<float> send_data(1024, 1.0f);
    std::vector<float> recv_data(1024, 0.0f);

    reducer.all_reduce(send_data.data(), recv_data.data(), 1024, 0, 0);

    EXPECT_EQ(recv_data[0], 1.0f);
#else
    GTEST_SKIP() << "NCCL not enabled";
#endif
}

class HierarchicalBarrierTest : public ::testing::Test {
protected:
};

TEST_F(HierarchicalBarrierTest, Construction) {
#if NOVA_NCCL_ENABLED
    HierarchicalBarrier barrier(nullptr, nullptr, nullptr);
    EXPECT_NO_THROW(barrier.synchronize());
#else
    GTEST_SKIP() << "NCCL not enabled";
#endif
}

TEST_F(HierarchicalBarrierTest, WaitWithNullComms) {
#if NOVA_NCCL_ENABLED
    HierarchicalBarrier barrier(nullptr, nullptr, nullptr);
    EXPECT_NO_THROW(barrier.wait());
#else
    GTEST_SKIP() << "NCCL not enabled";
#endif
}

}  // namespace cuda::multinode
