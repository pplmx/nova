#pragma once

/**
 * @file multi_node_context.h
 * @brief Multi-node NCCL communicator management
 *
 * Provides centralized management of cluster-scale NCCL communicators,
 * combining MPI rank discovery with topology-aware communicator creation.
 */

#include "cuda/nccl/nccl_context.h"
#include "cuda/mpi/mpi_context.h"
#include "cuda/topology/topology_map.h"

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::multinode {

class MultiNodeContext {
public:
    static MultiNodeContext& get_instance();
    static MultiNodeContext& instance() { return get_instance(); }

    struct NodeConfig {
        int world_rank = 0;
        int world_size = 1;
        int local_rank = 0;
        int local_size = 1;
        int node_id = 0;
        std::string hostname;
        topology::TopologyMap topology;
    };

    void initialize(const NodeConfig& config);
    void shutdown();

    [[nodiscard]] bool is_initialized() const { return initialized_; }
    [[nodiscard]] int global_rank() const { return global_rank_; }
    [[nodiscard]] int global_size() const { return global_size_; }
    [[nodiscard]] int local_rank() const { return local_rank_; }
    [[nodiscard]] int local_size() const { return local_size_; }
    [[nodiscard]] int num_nodes() const { return num_nodes_; }
    [[nodiscard]] int node_id() const { return node_id_; }
    [[nodiscard]] bool is_main_node() const { return node_id_ == 0; }
    [[nodiscard]] bool is_main_process() const { return global_rank_ == 0; }

#if NOVA_NCCL_ENABLED
    [[nodiscard]] ncclComm_t get_global_comm() const { return global_comm_; }
    [[nodiscard]] ncclComm_t get_local_comm() const { return local_comm_; }
#else
    [[nodiscard]] void* get_global_comm() const { return nullptr; }
    [[nodiscard]] void* get_local_comm() const { return nullptr; }
#endif

    [[nodiscard]] cudaStream_t get_global_stream() const { return global_stream_; }
    [[nodiscard]] cudaStream_t get_local_stream() const { return local_stream_; }

private:
    MultiNodeContext() = default;
    ~MultiNodeContext();

    void initialize_intra_node_comm();
    void initialize_inter_node_comm();

    bool initialized_ = false;
    int global_rank_ = 0;
    int global_size_ = 1;
    int local_rank_ = 0;
    int local_size_ = 1;
    int num_nodes_ = 1;
    int node_id_ = 0;

#if NOVA_NCCL_ENABLED
    ncclComm_t global_comm_ = nullptr;
    ncclComm_t local_comm_ = nullptr;
#else
    void* global_comm_ = nullptr;
    void* local_comm_ = nullptr;
#endif

    cudaStream_t global_stream_ = nullptr;
    cudaStream_t local_stream_ = nullptr;
};

class HierarchicalAllReduce {
public:
    HierarchicalAllReduce(void* local_comm,
                          void* global_comm,
                          cudaStream_t local_stream,
                          cudaStream_t global_stream);

    void all_reduce(const void* send_buf,
                    void* recv_buf,
                    size_t count,
                    int dtype,
                    int op);

    void synchronize();

private:
    void* local_comm_;
    void* global_comm_;
    cudaStream_t local_stream_;
    cudaStream_t global_stream_;
};

class HierarchicalBarrier {
public:
    explicit HierarchicalBarrier(void* local_comm,
                                  void* global_comm,
                                  cudaStream_t stream);

    void wait();
    void synchronize();

private:
    void* local_comm_;
    void* global_comm_;
    cudaStream_t stream_;
};

}  // namespace cuda::multinode
