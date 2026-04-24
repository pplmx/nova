#include "cuda/topology/topology_map.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <filesystem>
#include <unistd.h>

namespace {

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}

namespace cuda::topology {

PathType TopologyMap::classify_path(int other_node_id,
                                     const std::string& other_hostname) const {
    if (node_id == other_node_id || hostname == other_hostname) {
        return PathType::IntraNode;
    }
    return PathType::InterNode;
}

NicInfo TopologyMap::select_best_nic() const {
    NicInfo best;
    for (const auto& nic : nics) {
        if (nic.is_rdma_capable && nic.bandwidth_gbps > best.bandwidth_gbps) {
            best = nic;
        }
    }
    if (!best.name.empty()) {
        return best;
    }
    return nics.empty() ? NicInfo{} : nics[0];
}

bool TopologyMap::has_rdma() const {
    for (const auto& nic : nics) {
        if (nic.is_rdma_capable) return true;
    }
    return false;
}

std::string TopologyMap::network_type() const {
    NicInfo best = select_best_nic();
    switch (best.type) {
        case NicType::InfiniBand: return "InfiniBand";
        case NicType::RoCE: return "RoCE";
        case NicType::Ethernet: return "Ethernet";
        default: return "Unknown";
    }
}

TopologyMap TopologyDetector::detect() {
    TopologyMap map;

    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        map.hostname = hostname;
    }

    try {
        const std::string sys_class_net = "/sys/class/net";
        for (const auto& entry : std::filesystem::directory_iterator(sys_class_net)) {
            std::string name = entry.path().filename().string();
            if (name == "lo" || name == "docker0") continue;

            NicInfo nic;
            nic.name = name;

            const auto type_path = entry.path() / "type";
            if (file_exists(type_path.string())) {
                std::string type_str = read_file(type_path.string());
                int type = std::stoi(type_str);
                if (type == 32) {
                    nic.type = NicType::InfiniBand;
                } else {
                    nic.type = NicType::Ethernet;
                }
            }

            nic.is_rdma_capable = is_rdma_capable(name);
            nic.bandwidth_gbps = estimate_bandwidth(name);

            const auto addr_path = entry.path() / "address";
            if (file_exists(addr_path.string())) {
                nic.mac_address = read_file(addr_path.string());
                nic.mac_address.erase(
                    std::remove(nic.mac_address.begin(), nic.mac_address.end(), '\n'),
                    nic.mac_address.end()
                );
            }

            map.nics.push_back(nic);
        }
    } catch (const std::exception&) {
    }

    return map;
}

std::string TopologyDetector::probe_nic_type(const std::string& interface) {
    if (is_rdma_capable(interface)) {
        return "InfiniBand";
    }
    return "Ethernet";
}

uint64_t TopologyDetector::estimate_bandwidth(const std::string& interface) {
    const auto speed_path = "/sys/class/net/" + interface + "/speed";
    if (file_exists(speed_path)) {
        std::string speed_str = read_file(speed_path);
        if (!speed_str.empty()) {
            try {
                int speed = std::stoi(speed_str);
                return speed;
            } catch (const std::exception&) {
            }
        }
    }
    return 0;
}

bool TopologyDetector::is_rdma_capable(const std::string& interface) {
    const auto rdma_path = "/sys/class/net/" + interface + "/device/infiniband";
    return file_exists(rdma_path);
}

std::vector<NicInfo> TopologyDetector::enumerate_nics() {
    return detect().nics;
}

NcclTopologyContext::NcclTopologyContext(const TopologyMap& local_topology)
    : local_topology_(local_topology) {
    num_nodes_ = 1;
    intra_node_count_ = 1;
    inter_node_count_ = 1;
}

std::vector<CommunicatorConfig>
NcclTopologyContext::compute_communicator_configs(
    const std::vector<TopologyMap>& all_nodes) {

    std::vector<CommunicatorConfig> configs;

    std::map<std::string, int> node_group;
    for (size_t i = 0; i < all_nodes.size(); ++i) {
        node_group[all_nodes[i].hostname] = static_cast<int>(i);
    }

    int color = node_group[local_topology_.hostname];
    for (size_t i = 0; i < all_nodes.size(); ++i) {
        CommunicatorConfig cfg;
        cfg.color = node_group[all_nodes[i].hostname];
        cfg.key = static_cast<int>(i);
        cfg.preferred_nic_type = local_topology_.select_best_nic().type;
        cfg.require_rdma = local_topology_.has_rdma();
        configs.push_back(cfg);
    }

    num_nodes_ = static_cast<int>(node_group.size());

    return configs;
}

AlgorithmConfig CollectiveSelector::select_for_allreduce(
    uint64_t message_size_bytes,
    bool has_rdma,
    const std::string& network_type) {

    AlgorithmConfig config;

    constexpr uint64_t TREE_THRESHOLD = 32 * 1024 * 1024;
    constexpr uint64_t COLLNET_THRESHOLD = 64 * 1024 * 1024;

    if (has_rdma && network_type == "InfiniBand") {
        if (message_size_bytes >= COLLNET_THRESHOLD) {
            config.algorithm = CollectiveAlgorithm::CollNet;
        } else if (message_size_bytes >= TREE_THRESHOLD) {
            config.algorithm = CollectiveAlgorithm::Tree;
        } else {
            config.algorithm = CollectiveAlgorithm::Ring;
        }
    } else if (message_size_bytes >= TREE_THRESHOLD) {
        config.algorithm = CollectiveAlgorithm::Tree;
    } else {
        config.algorithm = CollectiveAlgorithm::Ring;
    }

    return config;
}

AlgorithmConfig CollectiveSelector::select_for_broadcast(
    uint64_t message_size_bytes,
    int num_receivers) {

    AlgorithmConfig config;

    constexpr uint64_t TREE_THRESHOLD = 32 * 1024 * 1024;

    if (num_receivers > 8 && message_size_bytes > TREE_THRESHOLD) {
        config.algorithm = CollectiveAlgorithm::Tree;
    } else {
        config.algorithm = CollectiveAlgorithm::Ring;
    }

    return config;
}

AlgorithmConfig CollectiveSelector::select_for_allgather(
    uint64_t message_size_bytes,
    int num_workers) {

    AlgorithmConfig config;
    config.algorithm = CollectiveAlgorithm::Ring;
    (void)message_size_bytes;
    (void)num_workers;
    return config;
}

namespace {

std::string algorithm_to_string(CollectiveAlgorithm algo) {
    switch (algo) {
        case CollectiveAlgorithm::Ring: return "Ring";
        case CollectiveAlgorithm::Tree: return "Tree";
        case CollectiveAlgorithm::CollNet: return "CollNet";
        case CollectiveAlgorithm::Pipeline: return "Pipeline";
        default: return "Ring";
    }
}

}

void CollectiveSelector::set_nccl_env(const AlgorithmConfig& config) {
    std::ostringstream env;
    env << "NCCL_ALGO=" << algorithm_to_string(config.algorithm);

    if (!config.nccl_tuning_env.empty()) {
        env << "," << config.nccl_tuning_env;
    }

    setenv("NCCL_TUNING", env.str().c_str(), 1);
}

AlgorithmConfig CollectiveSelector::from_env() {
    AlgorithmConfig config;

    const char* nccl_algo = std::getenv("NCCL_ALGO");
    if (nccl_algo) {
        if (std::strcmp(nccl_algo, "Ring") == 0) {
            config.algorithm = CollectiveAlgorithm::Ring;
        } else if (std::strcmp(nccl_algo, "Tree") == 0) {
            config.algorithm = CollectiveAlgorithm::Tree;
        } else if (std::strcmp(nccl_algo, "CollNet") == 0) {
            config.algorithm = CollectiveAlgorithm::CollNet;
        }
    }

    return config;
}

CollectiveProfiler::BenchmarkResult
CollectiveProfiler::profile_allreduce(const void* send_buf,
                                       void* recv_buf,
                                       size_t count,
                                       cudaStream_t stream) {
    BenchmarkResult result;
    result.algorithm = CollectiveAlgorithm::Ring;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);

#if NOVA_NCCL_ENABLED && NOVA_MPI_ENABLED
    NCCL_CHECK(ncclAllReduce(send_buf, recv_buf, count, ncclFloat, ncclSum,
                             NCCL_COMM_NULL, stream));
#endif

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    result.latency_us = ms * 1000.0;
    result.bandwidth_gbps = (count * sizeof(float)) / (ms * 1e-3) / 1e9;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

void CollectiveProfiler::generate_report(
    const std::vector<BenchmarkResult>& results) {

    std::printf("=== NCCL Collective Benchmark Report ===\n");
    for (const auto& r : results) {
        std::printf("Algorithm: %s, BW: %.2f GB/s, Latency: %.2f us\n",
                    algorithm_to_string(r.algorithm).c_str(),
                    r.bandwidth_gbps,
                    r.latency_us);
    }
}

void validate_topology(const TopologyMap& local,
                       const std::vector<TopologyMap>& all_nodes) {
    (void)local;
    if (all_nodes.empty()) {
        throw std::runtime_error("Empty topology list");
    }

    std::set<std::string> hostnames;
    for (const auto& node : all_nodes) {
        if (node.hostname.empty()) {
            throw std::runtime_error("Node with empty hostname in topology");
        }
        hostnames.insert(node.hostname);
    }

    if (hostnames.size() != all_nodes.size()) {
        throw std::runtime_error("Duplicate hostnames in topology");
    }
}

}  // namespace cuda::topology
