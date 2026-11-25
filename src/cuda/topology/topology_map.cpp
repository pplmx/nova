#include "cuda/topology/topology_map.h"
#include "cuda/device/error.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>

// Hostname lookup: <unistd.h> on POSIX, <winsock2.h> on Windows. Hostname is
// the only symbol this file needs a platform header for, and it is needed by
// detect(), whose NIC enumeration below is Linux-only.
#if defined(_WIN32)
#include <winsock2.h>
#else
#include <unistd.h>
#endif

namespace {

    /// Overwrite an environment variable.
    ///
    /// POSIX has setenv(). Windows has no setenv, only _putenv_s(), which also
    /// takes overwrite semantics so the two calls match. std::getenv() is
    /// portable and is what this file uses to read the value back.
    bool set_env(const char* name, const std::string& value) {
#if defined(_WIN32)
        return ::_putenv_s(name, value.c_str()) == 0;
#else
        return ::setenv(name, value.c_str(), 1) == 0;
#endif
    }

    bool file_exists(const std::string& path) {
        std::ifstream f(path);
        return f.good();
    }

    /// Parse a single integer out of a sysfs attribute.
    ///
    /// Returns nullopt for empty, non-numeric and out-of-range content instead of throwing:
    /// every caller here treats "the driver did not report this" as a legitimate answer, and
    /// an exception used to abort the whole enumeration.
    std::optional<long long> parse_sysfs_int(const std::string& text) {
        if (text.empty()) {
            return std::nullopt;
        }
        try {
            size_t consumed = 0;
            const long long value = std::stoll(text, &consumed);
            // Trailing whitespace/newline is expected; trailing junk means the file is not
            // the integer the caller expects.
            for (size_t i = consumed; i < text.size(); ++i) {
                const char c = text[i];
                if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                    return std::nullopt;
                }
            }
            return value;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

    std::string read_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.good()) {
            return "";
        }
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

}  // namespace

namespace cuda::topology {

    PathType TopologyMap::classify_path(int other_node_id, const std::string& other_hostname) const {
        if (node_id == other_node_id || hostname == other_hostname) {
            return PathType::IntraNode;
        }
        return PathType::InterNode;
    }

    NicInfo TopologyMap::select_best_nic() const {
        // Preference order, not a single "biggest number wins":
        //
        //   1. RDMA-capable and a known bandwidth    -- fastest fabric port
        //   2. RDMA-capable with no bandwidth figure -- an InfiniBand port reports no
        //      /sys/class/net/<if>/speed at all, because that attribute is Ethernet-only
        //   3. any interface with a known bandwidth
        //   4. whatever is left (the first NIC)
        //
        // The old body was `is_rdma_capable && bandwidth > best`, which silently reduced to
        // "RDMA and a positive bandwidth". InfiniBand ports have no speed file, so their
        // bandwidth is 0 and they could never win: measured on a host with four active IB
        // ports, this returned a 10 GbE veth. network_type() then reported "Ethernet", and
        // select_for_allreduce() -- which asks for "InfiniBand" before it will consider
        // CollNet -- could not select the algorithm the fabric was there for.
        // Lower is better. 0..3 are the four cases of the preference order above.
        const auto rank_of = [](const NicInfo& nic) {
            if (nic.is_rdma_capable) {
                return nic.bandwidth_gbps > 0 ? 0 : 1;
            }
            return nic.bandwidth_gbps > 0 ? 2 : 3;
        };

        const NicInfo* best = nullptr;
        for (const auto& nic : nics) {
            if (best == nullptr) {
                best = &nic;
                continue;
            }
            const int candidate_rank = rank_of(nic);
            const int current_rank = rank_of(*best);
            if (candidate_rank != current_rank) {
                if (candidate_rank < current_rank) {
                    best = &nic;
                }
                continue;
            }
            // Within a rank, more bandwidth wins; ties keep the earlier interface, which
            // keeps the result stable for a given host.
            if (nic.bandwidth_gbps > best->bandwidth_gbps) {
                best = &nic;
            }
        }
        return best != nullptr ? *best : NicInfo {};
    }

    bool TopologyMap::has_rdma() const {
        for (const auto& nic : nics) {
            if (nic.is_rdma_capable) {
                return true;
            }
        }
        return false;
    }

    std::string TopologyMap::network_type() const {
        NicInfo best = select_best_nic();
        switch (best.type) {
            case NicType::InfiniBand:
                return "InfiniBand";
            case NicType::RoCE:
                return "RoCE";
            case NicType::Ethernet:
                return "Ethernet";
            default:
                return "Unknown";
        }
    }

    TopologyMap TopologyDetector::detect() {
        TopologyMap map;

        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) == 0) {
            map.hostname = hostname;
        }

#if defined(__linux__)
        try {
            const std::string sys_class_net = "/sys/class/net";
            for (const auto& entry : std::filesystem::directory_iterator(sys_class_net)) {
                std::string name = entry.path().filename().string();
                if (name == "lo" || name == "docker0") {
                    continue;
                }

                // /sys/class/net holds a few entries that are not interfaces with a `type`
                // file -- bonding_masters is the clearest example -- and directory_iterator
                // happily lists them. Without this check such an entry became a NicInfo with
                // type Unknown. Measured: bonding_masters was the FIRST entry, so it won
                // select_best_nic() on every device below and reported the network as
                // Unknown although four InfiniBand ports were present.
                const auto type_path = entry.path() / "type";
                if (!file_exists(type_path.string())) {
                    continue;
                }

                NicInfo nic;
                nic.name = name;

                // A type file that exists but reads as empty or non-numeric is a NIC whose
                // sort the driver has not filled in; it used to reach std::stoi and throw,
                // which the enclosing catch then turned into "stop enumerating", so every
                // interface after it in the directory order was silently dropped.
                const std::string type_str = read_file(type_path.string());
                const auto parsed_type = parse_sysfs_int(type_str);
                if (!parsed_type.has_value()) {
                    nic.type = NicType::Unknown;
                } else if (*parsed_type == 32) {
                    nic.type = NicType::InfiniBand;
                } else {
                    nic.type = NicType::Ethernet;
                }

                nic.is_rdma_capable = is_rdma_capable(name);
                nic.bandwidth_gbps = estimate_bandwidth(name);

                const auto addr_path = entry.path() / "address";
                if (file_exists(addr_path.string())) {
                    nic.mac_address = read_file(addr_path.string());
                    nic.mac_address.erase(std::remove(nic.mac_address.begin(), nic.mac_address.end(), '\n'), nic.mac_address.end());
                }

                map.nics.push_back(nic);
            }
        } catch (const std::exception&) {
            // Deliberate and documented: topology detection is best-effort. A machine without
            // /sys/class/net (a container, an unusual kernel) has no NIC list to report, and
            // failing detection there would prevent the caller from doing anything at all.
            // Whatever was collected before the failure is returned. The caller distinguishes
            // "no NICs found" from "a NIC with unknown properties" by looking at the vector's
            // size, not by catching.
        }
#else
        // NIC enumeration reads /sys/class/net and per-interface type, address and
        // speed files. That is Linux sysfs; on Windows and macOS it would silently
        // yield an empty list anyway, so the hostname above is all that is reported
        // and the caller sees nics.empty() as it would on a Linux machine without
        // sysfs.
#endif

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
        const auto speed = parse_sysfs_int(read_file(speed_path));
        if (!speed.has_value()) {
            return 0;
        }
        // A negative speed means UNKNOWN, not infinite. "-1" is what the kernel writes for an
        // interface whose speed the driver cannot report (virtual and bridged interfaces,
        // some bonds). This returned the int directly from a uint64_t function, so -1 became
        // 18446744073709551615 -- the LARGEST possible bandwidth. Measured on this host:
        // six br-* interfaces reported that value, and because select_best_nic() takes the
        // maximum, a Docker bridge outranked the machine's four InfiniBand ports.
        if (*speed < 0) {
            return 0;
        }
        return static_cast<uint64_t>(*speed);
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

    std::vector<CommunicatorConfig> NcclTopologyContext::compute_communicator_configs(const std::vector<TopologyMap>& all_nodes) {
        std::vector<CommunicatorConfig> configs;

        std::map<std::string, int> node_group;
        for (size_t i = 0; i < all_nodes.size(); ++i) {
            node_group[all_nodes[i].hostname] = static_cast<int>(i);
        }

        for (size_t i = 0; i < all_nodes.size(); ++i) {
            CommunicatorConfig cfg;
            cfg.color = node_group[all_nodes[i].hostname];
            cfg.key = static_cast<int>(i);
            cfg.preferred_nic_type = local_topology_.select_best_nic().type;
            cfg.require_rdma = local_topology_.has_rdma();
            configs.push_back(cfg);
        }

        num_nodes_ = static_cast<int>(node_group.size());

        // local and cross describe the communicator split for THIS rank: how many ranks
        // share its host, and how many are elsewhere. Both were set to 1 in the
        // constructor and never assigned again, so they reported 1 for every topology --
        // a value the only existing assertion (>= 1) could not distinguish from a real
        // answer. Classify the same way classify_path does: same hostname is local.
        int local = 0;
        int cross = 0;
        for (const auto& node : all_nodes) {
            if (node.hostname == local_topology_.hostname) {
                ++local;
            } else {
                ++cross;
            }
        }
        intra_node_count_ = local;
        inter_node_count_ = cross;

        return configs;
    }

    AlgorithmConfig CollectiveSelector::select_for_allreduce(uint64_t message_size_bytes, bool has_rdma, const std::string& network_type) {
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

    AlgorithmConfig CollectiveSelector::select_for_broadcast(uint64_t message_size_bytes, int num_receivers) {
        AlgorithmConfig config;

        constexpr uint64_t TREE_THRESHOLD = 32 * 1024 * 1024;

        if (num_receivers > 8 && message_size_bytes > TREE_THRESHOLD) {
            config.algorithm = CollectiveAlgorithm::Tree;
        } else {
            config.algorithm = CollectiveAlgorithm::Ring;
        }

        return config;
    }

    AlgorithmConfig CollectiveSelector::select_for_allgather(uint64_t message_size_bytes, int num_workers) {
        AlgorithmConfig config;

        // allgather used to return Ring unconditionally, ignoring both arguments,
        // while select_for_allreduce and select_for_broadcast applied the same
        // large-message threshold. The intent is the same heuristic: a ring passes
        // data around and wins for small messages, while a tree (or collnet with
        // enough ranks) shortens the critical path once the message is large.
        constexpr uint64_t TREE_THRESHOLD = 32 * 1024 * 1024;

        if (num_workers > 8 && message_size_bytes > TREE_THRESHOLD) {
            config.algorithm = CollectiveAlgorithm::Tree;
        } else {
            config.algorithm = CollectiveAlgorithm::Ring;
        }

        return config;
    }

    namespace {

        std::string algorithm_to_string(CollectiveAlgorithm algo) {
            switch (algo) {
                case CollectiveAlgorithm::Ring:
                    return "Ring";
                case CollectiveAlgorithm::Tree:
                    return "Tree";
                case CollectiveAlgorithm::CollNet:
                    return "CollNet";
                case CollectiveAlgorithm::Pipeline:
                    return "Pipeline";
                default:
                    return "Ring";
            }
        }

    }  // namespace

    void CollectiveSelector::set_nccl_env(const AlgorithmConfig& config) {
        // NCCL reads the algorithm from NCCL_ALGO. Writing "NCCL_ALGO=Tree" into
        // NCCL_TUNING, as this did, leaves NCCL_ALGO unset and the selection has no
        // effect at all — measured: NCCL_ALGO was (unset) while NCCL_TUNING held
        // "NCCL_ALGO=CollNet".
        set_env("NCCL_ALGO", algorithm_to_string(config.algorithm));

        // Any extra tuning fragment the caller supplied is appended to NCCL_ALGO,
        // which is where NCCL accepts the comma-separated form (e.g. "Ring,Tree").
        if (!config.nccl_tuning_env.empty()) {
            const std::string combined = algorithm_to_string(config.algorithm) + "," + config.nccl_tuning_env;
            set_env("NCCL_ALGO", combined);
        }
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

    CollectiveProfiler::BenchmarkResult CollectiveProfiler::profile_allreduce(const void* send_buf, void* recv_buf, size_t count, cudaStream_t stream) {
        BenchmarkResult result;
        result.algorithm = CollectiveAlgorithm::Ring;

        // Both events released by a guard: the creates, the records, the synchronize and the
        // elapsed-time query between them all can throw, and the destroys used to sit after all of
        // them. Neither handle was initialised, so a failure on the second create left nothing a
        // cleanup could have released even in principle.
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        struct EventGuard {
            cudaEvent_t& start;
            cudaEvent_t& stop;
            ~EventGuard() noexcept {
                if (start != nullptr) {
                    (void)cudaEventDestroy(start);
                }
                if (stop != nullptr) {
                    (void)cudaEventDestroy(stop);
                }
            }
        } guard {start, stop};

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));

#if NOVA_NCCL_ENABLED && NOVA_MPI_ENABLED
        NCCL_CHECK(ncclAllReduce(send_buf, recv_buf, count, ncclFloat, ncclSum, NCCL_COMM_NULL, stream));
#endif

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        result.latency_us = ms * 1000.0;
        result.bandwidth_gbps = (count * sizeof(float)) / (ms * 1e-3) / 1e9;

        return result;
    }

    void CollectiveProfiler::generate_report(const std::vector<BenchmarkResult>& results) {
        std::printf("=== NCCL Collective Benchmark Report ===\n");
        for (const auto& r : results) {
            std::printf("Algorithm: %s, BW: %.2f GB/s, Latency: %.2f us\n", algorithm_to_string(r.algorithm).c_str(), r.bandwidth_gbps, r.latency_us);
        }
    }

    void validate_topology(const TopologyMap& local, const std::vector<TopologyMap>& all_nodes) {
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
