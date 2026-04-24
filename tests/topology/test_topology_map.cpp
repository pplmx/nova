/**
 * @file test_topology_map.cpp
 * @brief Unit tests for topology detection and NIC classification
 *
 * Tests TopologyDetector, TopologyMap, and CollectiveSelector.
 */

#include <gtest/gtest.h>

#include "cuda/topology/topology_map.h"

namespace cuda::topology {

class TopologyMapTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(TopologyMapTest, DefaultConstruction) {
    TopologyMap map;
    EXPECT_EQ(map.node_id, 0);
    EXPECT_TRUE(map.hostname.empty());
    EXPECT_TRUE(map.nics.empty());
}

TEST_F(TopologyMapTest, PathClassificationIntraNode) {
    TopologyMap map;
    map.node_id = 0;
    map.hostname = "node0";

    EXPECT_EQ(map.classify_path(0, "node0"), PathType::IntraNode);
}

TEST_F(TopologyMapTest, PathClassificationInterNode) {
    TopologyMap map;
    map.node_id = 0;
    map.hostname = "node0";

    EXPECT_EQ(map.classify_path(1, "node1"), PathType::InterNode);
}

TEST_F(TopologyMapTest, SelectBestNicPrefersRdma) {
    TopologyMap map;
    NicInfo eth_nic;
    eth_nic.name = "eth0";
    eth_nic.type = NicType::Ethernet;
    eth_nic.bandwidth_gbps = 100;
    eth_nic.is_rdma_capable = false;

    NicInfo ib_nic;
    ib_nic.name = "ib0";
    ib_nic.type = NicType::InfiniBand;
    ib_nic.bandwidth_gbps = 200;
    ib_nic.is_rdma_capable = true;

    map.nics = {eth_nic, ib_nic};

    NicInfo best = map.select_best_nic();
    EXPECT_EQ(best.name, "ib0");
    EXPECT_EQ(best.type, NicType::InfiniBand);
    EXPECT_TRUE(best.is_rdma_capable);
}

TEST_F(TopologyMapTest, SelectBestNicFallsBackToFirst) {
    TopologyMap map;
    NicInfo nic1;
    nic1.name = "eth0";
    nic1.type = NicType::Ethernet;
    nic1.bandwidth_gbps = 100;
    nic1.is_rdma_capable = false;

    map.nics = {nic1};

    NicInfo best = map.select_best_nic();
    EXPECT_EQ(best.name, "eth0");
}

TEST_F(TopologyMapTest, HasRdmaTrueForInfiniBand) {
    TopologyMap map;
    NicInfo ib_nic;
    ib_nic.is_rdma_capable = true;
    map.nics = {ib_nic};

    EXPECT_TRUE(map.has_rdma());
}

TEST_F(TopologyMapTest, HasRdmaFalseForEthernet) {
    TopologyMap map;
    NicInfo eth_nic;
    eth_nic.is_rdma_capable = false;
    map.nics = {eth_nic};

    EXPECT_FALSE(map.has_rdma());
}

TEST_F(TopologyMapTest, NetworkTypeInfiniBand) {
    TopologyMap map;
    NicInfo nic;
    nic.type = NicType::InfiniBand;
    nic.is_rdma_capable = true;
    map.nics = {nic};

    EXPECT_EQ(map.network_type(), "InfiniBand");
}

TEST_F(TopologyMapTest, NetworkTypeEthernet) {
    TopologyMap map;
    NicInfo nic;
    nic.type = NicType::Ethernet;
    nic.is_rdma_capable = false;
    map.nics = {nic};

    EXPECT_EQ(map.network_type(), "Ethernet");
}

TEST_F(TopologyMapTest, NetworkTypeRoCE) {
    TopologyMap map;
    NicInfo nic;
    nic.type = NicType::RoCE;
    nic.is_rdma_capable = true;
    map.nics = {nic};

    EXPECT_EQ(map.network_type(), "RoCE");
}

class CollectiveSelectorTest : public ::testing::Test {
protected:
};

TEST_F(CollectiveSelectorTest, SelectAllReduceRingForSmallMessages) {
    AlgorithmConfig config = CollectiveSelector::select_for_allreduce(
        1024, true, "InfiniBand");

    EXPECT_EQ(config.algorithm, CollectiveAlgorithm::Ring);
}

TEST_F(CollectiveSelectorTest, SelectAllReduceTreeForLargeMessages) {
    AlgorithmConfig config = CollectiveSelector::select_for_allreduce(
        64 * 1024 * 1024, false, "Ethernet");

    EXPECT_EQ(config.algorithm, CollectiveAlgorithm::Tree);
}

TEST_F(CollectiveSelectorTest, SelectAllReduceCollNetForLargeRdmaMessages) {
    AlgorithmConfig config = CollectiveSelector::select_for_allreduce(
        128 * 1024 * 1024, true, "InfiniBand");

    EXPECT_EQ(config.algorithm, CollectiveAlgorithm::CollNet);
}

TEST_F(CollectiveSelectorTest, SelectBroadcastRingDefault) {
    AlgorithmConfig config = CollectiveSelector::select_for_broadcast(
        1024, 2);

    EXPECT_EQ(config.algorithm, CollectiveAlgorithm::Ring);
}

TEST_F(CollectiveSelectorTest, SelectBroadcastTreeForManyReceivers) {
    AlgorithmConfig config = CollectiveSelector::select_for_broadcast(
        64 * 1024 * 1024, 16);

    EXPECT_EQ(config.algorithm, CollectiveAlgorithm::Tree);
}

TEST_F(CollectiveSelectorTest, SelectAllGatherDefault) {
    AlgorithmConfig config = CollectiveSelector::select_for_allgather(
        1024, 4);

    EXPECT_EQ(config.algorithm, CollectiveAlgorithm::Ring);
}

TEST_F(CollectiveSelectorTest, FromEnvReadsNCCLAlgo) {
    setenv("NCCL_ALGO", "Tree", 1);

    AlgorithmConfig config = CollectiveSelector::from_env();
    EXPECT_EQ(config.algorithm, CollectiveAlgorithm::Tree);

    unsetenv("NCCL_ALGO");
}

TEST_F(CollectiveSelectorTest, FromEnvDefaultsToRing) {
    unsetenv("NCCL_ALGO");

    AlgorithmConfig config = CollectiveSelector::from_env();
    EXPECT_EQ(config.algorithm, CollectiveAlgorithm::Ring);
}

class NcclTopologyContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        local_map_.hostname = "node0";
        NicInfo nic;
        nic.name = "eth0";
        nic.type = NicType::Ethernet;
        nic.is_rdma_capable = false;
        local_map_.nics = {nic};
    }

    TopologyMap local_map_;
};

TEST_F(NcclTopologyContextTest, ConstructorSetsLocalTopology) {
    NcclTopologyContext ctx(local_map_);

    EXPECT_GE(ctx.get_local_comm_size(), 1);
    EXPECT_GE(ctx.get_cross_node_comm_size(), 1);
}

TEST_F(NcclTopologyContextTest, ComputeConfigsGroupsSameHostname) {
    TopologyMap map1;
    map1.hostname = "testnode";

    TopologyMap map2;
    map2.hostname = "testnode";

    NcclTopologyContext ctx(map1);

    std::vector<TopologyMap> all_nodes = {map1, map2};

    auto configs = ctx.compute_communicator_configs(all_nodes);

    EXPECT_EQ(configs.size(), 2u);
    EXPECT_EQ(configs[0].color, configs[1].color);
    EXPECT_EQ(ctx.get_num_nodes(), 1);
}

TEST_F(NcclTopologyContextTest, ComputeConfigsDifferentHostnames) {
    TopologyMap map2;
    map2.hostname = "node1";

    NcclTopologyContext ctx(local_map_);

    std::vector<TopologyMap> all_nodes = {local_map_, map2};

    auto configs = ctx.compute_communicator_configs(all_nodes);

    EXPECT_EQ(configs.size(), 2u);
    EXPECT_EQ(configs[0].color, 0);
    EXPECT_EQ(configs[1].color, 1);
    EXPECT_EQ(ctx.get_num_nodes(), 2);
}

TEST_F(NcclTopologyContextTest, ComputeConfigsDifferentNodes) {
    TopologyMap map2;
    map2.hostname = "node1";

    NcclTopologyContext ctx(local_map_);

    std::vector<TopologyMap> all_nodes = {local_map_, map2};

    auto configs = ctx.compute_communicator_configs(all_nodes);

    EXPECT_EQ(configs.size(), 2u);
    EXPECT_EQ(configs[0].color, 0);
    EXPECT_EQ(configs[1].color, 1);
    EXPECT_EQ(ctx.get_num_nodes(), 2);
}

TEST_F(NcclTopologyContextTest, ConfigCarriesRdmaRequirement) {
    local_map_.nics[0].is_rdma_capable = true;

    NcclTopologyContext ctx(local_map_);

    std::vector<TopologyMap> all_nodes = {local_map_};
    auto configs = ctx.compute_communicator_configs(all_nodes);

    EXPECT_TRUE(configs[0].require_rdma);
}

}  // namespace cuda::topology
