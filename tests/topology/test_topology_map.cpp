/**
 * @file test_topology_map.cpp
 * @brief Unit tests for topology detection and NIC classification
 *
 * Tests TopologyDetector, TopologyMap, and CollectiveSelector.
 */

#include <gtest/gtest.h>

#include <stdexcept>

#include "cuda_alloc_injection.h"
#include "cuda/topology/topology_map.h"

namespace cuda::topology {

    class TopologyMapTest : public ::testing::Test {
    protected:
        void SetUp() override {}
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
        AlgorithmConfig config = CollectiveSelector::select_for_allreduce(1024, true, "InfiniBand");

        EXPECT_EQ(config.algorithm, CollectiveAlgorithm::Ring);
    }

    TEST_F(CollectiveSelectorTest, SelectAllReduceTreeForLargeMessages) {
        AlgorithmConfig config = CollectiveSelector::select_for_allreduce(64 * 1024 * 1024, false, "Ethernet");

        EXPECT_EQ(config.algorithm, CollectiveAlgorithm::Tree);
    }

    TEST_F(CollectiveSelectorTest, SelectAllReduceCollNetForLargeRdmaMessages) {
        AlgorithmConfig config = CollectiveSelector::select_for_allreduce(128 * 1024 * 1024, true, "InfiniBand");

        EXPECT_EQ(config.algorithm, CollectiveAlgorithm::CollNet);
    }

    TEST_F(CollectiveSelectorTest, SelectBroadcastRingDefault) {
        AlgorithmConfig config = CollectiveSelector::select_for_broadcast(1024, 2);

        EXPECT_EQ(config.algorithm, CollectiveAlgorithm::Ring);
    }

    TEST_F(CollectiveSelectorTest, SelectBroadcastTreeForManyReceivers) {
        AlgorithmConfig config = CollectiveSelector::select_for_broadcast(64 * 1024 * 1024, 16);

        EXPECT_EQ(config.algorithm, CollectiveAlgorithm::Tree);
    }

    TEST_F(CollectiveSelectorTest, SelectAllGatherDefault) {
        AlgorithmConfig config = CollectiveSelector::select_for_allgather(1024, 4);

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

    // get_local_comm_size() and get_cross_node_comm_size() returned members that were set
    // to 1 in the constructor and never assigned again, so both reported 1 for every
    // topology. The existing case asserted >= 1, which a constant satisfies -- that is how
    // the stubs survived.
    //
    // What the two are supposed to mean: a communicator that splits across hosts needs to
    // know how many ranks stay inside one host (local) and how many are elsewhere (cross).
    // After compute_communicator_configs() has been told about every node, those counts are
    // determined by the hostnames.
    TEST_F(NcclTopologyContextTest, CommSizesReflectTheTopologyThatWasComputed) {
        NcclTopologyContext ctx(local_map_);

        // The rank asking the question is itself in the list, so the fixture's own hostname
        // must appear among the nodes -- leaving it out made local 0, which the first
        // version of this case tripped over. Three ranks on node0, one on another host.
        std::vector<TopologyMap> nodes(4);
        nodes[0].hostname = local_map_.hostname;
        nodes[1].hostname = local_map_.hostname;
        nodes[2].hostname = local_map_.hostname;
        nodes[3].hostname = "elsewhere";

        const auto configs = ctx.compute_communicator_configs(nodes);
        ASSERT_EQ(configs.size(), 4u);

        EXPECT_EQ(ctx.get_num_nodes(), 2) << "two distinct hostnames";
        // The local communicator covers this rank's own host; with the constructor's
        // local_topology_ being nodes' host we cannot know which, so assert the property
        // that must hold either way: local + cross == total ranks, and neither is the
        // hardcoded 1 unless the topology really is one rank per host.
        EXPECT_EQ(ctx.get_local_comm_size() + ctx.get_cross_node_comm_size(), 4) << "every rank is either local or cross-node";
        EXPECT_EQ(ctx.get_local_comm_size(), 3) << "three ranks share this host";
        EXPECT_EQ(ctx.get_cross_node_comm_size(), 1) << "one rank is elsewhere";
    }

    // Regression: select_best_nic() was `is_rdma_capable && bandwidth > best`, which reduced to
    // "RDMA and a POSITIVE bandwidth". An InfiniBand port has no /sys/class/net/<if>/speed --
    // that attribute is Ethernet-only -- so its bandwidth is 0 and it could never be selected.
    // Measured on this host: four active IB ports present, and the function returned a 10 GbE
    // veth, from which network_type() reported "Ethernet". select_for_allreduce() only
    // considers CollNet when network_type() is "InfiniBand", so the fabric could never be used.
    TEST_F(TopologyMapTest, SelectsAnRdmaPortOverAFasterLookingEthernetOne) {
        TopologyMap map;
        NicInfo ethernet;
        ethernet.name = "eth0";
        ethernet.type = NicType::Ethernet;
        ethernet.bandwidth_gbps = 10000;

        NicInfo infiniband;
        infiniband.name = "ib0";
        infiniband.type = NicType::InfiniBand;
        infiniband.bandwidth_gbps = 0;  // no speed attribute for IB
        infiniband.is_rdma_capable = true;

        map.nics = {ethernet, infiniband};

        EXPECT_EQ(map.select_best_nic().name, "ib0") << "an RDMA port outranks a plain Ethernet one";
        EXPECT_EQ(map.network_type(), "InfiniBand") << "and the network is named after it";
        EXPECT_TRUE(map.has_rdma());
    }

    // Among RDMA ports the fastest still wins, and among Ethernet interfaces too: the ranking
    // must not have turned into "the first RDMA port always wins".
    TEST_F(TopologyMapTest, PrefersMoreBandwidthWithinTheSameClass) {
        TopologyMap map;
        NicInfo slow;
        slow.name = "mlx5_0";
        slow.type = NicType::InfiniBand;
        slow.is_rdma_capable = true;
        slow.bandwidth_gbps = 100;

        NicInfo fast;
        fast.name = "mlx5_1";
        fast.type = NicType::InfiniBand;
        fast.is_rdma_capable = true;
        fast.bandwidth_gbps = 400;

        map.nics = {slow, fast};
        EXPECT_EQ(map.select_best_nic().name, "mlx5_1");

        map.nics = {};
        NicInfo eth_slow;
        eth_slow.name = "eth0";
        eth_slow.type = NicType::Ethernet;
        eth_slow.bandwidth_gbps = 1000;
        NicInfo eth_fast;
        eth_fast.name = "eth1";
        eth_fast.type = NicType::Ethernet;
        eth_fast.bandwidth_gbps = 25000;
        map.nics = {eth_slow, eth_fast};
        EXPECT_EQ(map.select_best_nic().name, "eth1");
    }

    // Regression: estimate_bandwidth() returned an int directly from a uint64_t function, so
    // the kernel's "-1" -- which means "this interface cannot report its speed" -- became
    // 18446744073709551615, the largest possible bandwidth. Six Docker bridges on this host
    // reported that value and outranked every real port. "Unknown" must read as 0.
    TEST_F(TopologyMapTest, UnreportedSpeedIsNotTreatedAsInfiniteBandwidth) {
        // A real interface whose driver reports no speed is what this asserts about; on a host
        // where every interface reports one there is nothing to observe, so pick any and check
        // the invariant that no interface claims an absurd bandwidth.
        const TopologyMap map = TopologyDetector::detect();
        for (const auto& nic : map.nics) {
            EXPECT_LT(nic.bandwidth_gbps, uint64_t(100) * 1000 * 1000) << nic.name << " reports an implausible bandwidth; -1 means unknown, not infinite";
        }

        // And the translation itself, through the public entry point: an interface that does
        // not exist has no speed file, so it must be 0 rather than anything else.
        EXPECT_EQ(TopologyDetector::estimate_bandwidth("nova-no-such-interface"), 0u);
    }

    // Regression: /sys/class/net also contains entries that are not interfaces with a type
    // file (bonding_masters is the one on this host). directory_iterator lists them, and
    // without a check they became NicInfo records of type Unknown -- and because
    // bonding_masters sorts first, it won select_best_nic() on an otherwise idle machine.
    // CollectiveProfiler::profile_allreduce creates two events, records both, synchronizes and reads
    // the elapsed time, and only then destroys them. Every step in between is a CUDA_CHECK, so a
    // failure on the second create (or anywhere after it) used to leak whatever had been taken, and
    // neither handle was initialised.
    TEST(CollectiveProfilerLeakTest, AFailedSecondEventCreateLeavesNothingBehind) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "no CUDA device";
        }
        (void)cudaGetLastError();

        CollectiveProfiler profiler;

        cuda_test::reset_event_count();

        // Let the first event through, fail the second.
        cuda_test::fail_event_after(1);
        EXPECT_THROW((void)profiler.profile_allreduce(nullptr, nullptr, 16, nullptr), std::exception);
        cuda_test::stop_failing_events();

        EXPECT_EQ(cuda_test::live_event_count(), 0) << "the event created before the failure must be destroyed";
    }

    TEST_F(TopologyMapTest, OnlyRealInterfacesAreEnumerated) {
        const TopologyMap map = TopologyDetector::detect();
        for (const auto& nic : map.nics) {
            EXPECT_FALSE(nic.name.empty());
            EXPECT_NE(nic.name, "bonding_masters") << "that is a sysfs control file, not an interface";
            EXPECT_NE(nic.name, "lo") << "loopback is not a fabric path";
        }
    }

}  // namespace cuda::topology
