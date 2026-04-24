#pragma once

/**
 * @file topology_map.h
 * @brief Node topology detection and NIC classification
 *
 * Provides infrastructure for detecting node topology, enumerating network
 * interfaces, and classifying communication paths for optimal collective selection.
 */

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::topology {

enum class NicType {
    Unknown,
    Ethernet,
    InfiniBand,
    RoCE,
};

enum class PathType {
    IntraNode,
    InterNode,
};

enum class CollectiveAlgorithm {
    Ring,
    Tree,
    CollNet,
    Pipeline,
};

struct NicInfo {
    std::string name;
    std::string mac_address;
    NicType type = NicType::Unknown;
    uint64_t bandwidth_gbps = 0;
    bool is_rdma_capable = false;
    std::string ip_address;
};

struct TopologyMap {
    int node_id = 0;
    std::string hostname;
    std::vector<NicInfo> nics;

    [[nodiscard]] PathType classify_path(int other_node_id,
                                          const std::string& other_hostname) const;

    [[nodiscard]] NicInfo select_best_nic() const;

    [[nodiscard]] bool has_rdma() const;

    [[nodiscard]] std::string network_type() const;
};

struct CommunicatorConfig {
    int color = 0;
    int key = 0;
    NicType preferred_nic_type = NicType::Unknown;
    bool require_rdma = false;
};

class TopologyDetector {
public:
    [[nodiscard]] static TopologyMap detect();

    [[nodiscard]] static std::string probe_nic_type(const std::string& interface);

    [[nodiscard]] static uint64_t estimate_bandwidth(const std::string& interface);

    [[nodiscard]] static bool is_rdma_capable(const std::string& interface);

    [[nodiscard]] static std::vector<NicInfo> enumerate_nics();
};

class NcclTopologyContext {
public:
    explicit NcclTopologyContext(const TopologyMap& local_topology);

    [[nodiscard]] std::vector<CommunicatorConfig>
    compute_communicator_configs(const std::vector<TopologyMap>& all_nodes);

    [[nodiscard]] int get_local_comm_size() const { return intra_node_count_; }

    [[nodiscard]] int get_cross_node_comm_size() const { return inter_node_count_; }

    [[nodiscard]] int get_num_nodes() const { return num_nodes_; }

private:
    TopologyMap local_topology_;
    int num_nodes_ = 1;
    int intra_node_count_ = 1;
    int inter_node_count_ = 1;
};

struct AlgorithmConfig {
    CollectiveAlgorithm algorithm = CollectiveAlgorithm::Ring;
    int min_message_size = 0;
    int max_message_size = 1 << 30;
    std::string nccl_tuning_env;
};

class CollectiveSelector {
public:
    [[nodiscard]] static AlgorithmConfig
    select_for_allreduce(uint64_t message_size_bytes,
                          bool has_rdma,
                          const std::string& network_type);

    [[nodiscard]] static AlgorithmConfig
    select_for_broadcast(uint64_t message_size_bytes, int num_receivers);

    [[nodiscard]] static AlgorithmConfig
    select_for_allgather(uint64_t message_size_bytes, int num_workers);

    static void set_nccl_env(const AlgorithmConfig& config);

    [[nodiscard]] static AlgorithmConfig from_env();
};

class CollectiveProfiler {
public:
    struct BenchmarkResult {
        double bandwidth_gbps = 0.0;
        double latency_us = 0.0;
        CollectiveAlgorithm algorithm = CollectiveAlgorithm::Ring;
    };

    [[nodiscard]] BenchmarkResult
    profile_allreduce(const void* send_buf,
                      void* recv_buf,
                      size_t count,
                      cudaStream_t stream);

    static void generate_report(const std::vector<BenchmarkResult>& results);
};

void validate_topology(const TopologyMap& local,
                       const std::vector<TopologyMap>& all_nodes);

}  // namespace cuda::topology
