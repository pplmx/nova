#include "cuda/multinode/multi_node_context.h"

#include <cstdio>
#include <cstring>
#include <memory>

namespace cuda::multinode {

namespace {

MultiNodeContext* global_instance = nullptr;

}

MultiNodeContext& MultiNodeContext::get_instance() {
    if (!global_instance) {
        global_instance = new MultiNodeContext();
    }
    return *global_instance;
}

MultiNodeContext::~MultiNodeContext() {
    shutdown();
}

void MultiNodeContext::initialize(const NodeConfig& config) {
    if (initialized_) {
        return;
    }

    global_rank_ = config.world_rank;
    global_size_ = config.world_size;
    local_rank_ = config.local_rank;
    local_size_ = config.local_size;
    node_id_ = config.node_id;
    num_nodes_ = config.world_size / config.local_size;

    CUDA_CHECK(cudaSetDevice(local_rank_));
    CUDA_CHECK(cudaStreamCreate(&global_stream_));
    CUDA_CHECK(cudaStreamCreate(&local_stream_));

#if NOVA_NCCL_ENABLED
    initialize_intra_node_comm();
    initialize_inter_node_comm();
#else
    if (config.world_size > 1) {
        std::fprintf(stderr, "Warning: Multi-node requested but NCCL not available. "
                              "Falling back to single-node mode.\n");
        global_size_ = 1;
        num_nodes_ = 1;
    }
#endif

    initialized_ = true;
}

void MultiNodeContext::initialize_intra_node_comm() {
#if NOVA_NCCL_ENABLED
    auto& nccl = cuda::nccl::NcclContext::instance();
    if (!nccl.initialized()) {
        nccl.initialize();
    }
    local_comm_ = nccl.get_comm(local_rank_);
#endif
}

void MultiNodeContext::initialize_inter_node_comm() {
#if NOVA_NCCL_ENABLED
    if (global_size_ > 1) {
        std::fprintf(stderr, "Inter-node NCCL communicator requires MPI environment\n");
    }
#else
    (void)global_rank_;
#endif
}

void MultiNodeContext::shutdown() {
    if (!initialized_) {
        return;
    }

#if NOVA_NCCL_ENABLED
    if (global_comm_ != nullptr) {
        ncclCommDestroy(global_comm_);
        global_comm_ = nullptr;
    }
#endif

    if (global_stream_) {
        cudaStreamDestroy(global_stream_);
        global_stream_ = nullptr;
    }
    if (local_stream_) {
        cudaStreamDestroy(local_stream_);
        local_stream_ = nullptr;
    }

    initialized_ = false;
}

HierarchicalAllReduce::HierarchicalAllReduce(void* local_comm,
                                             void* global_comm,
                                             cudaStream_t local_stream,
                                             cudaStream_t global_stream)
    : local_comm_(local_comm),
      global_comm_(global_comm),
      local_stream_(local_stream),
      global_stream_(global_stream) {}

void HierarchicalAllReduce::all_reduce(const void* send_buf,
                                       void* recv_buf,
                                       size_t count,
                                       int dtype,
                                       int op) {
#if NOVA_NCCL_ENABLED
    if (local_comm_ && global_comm_) {
        NCCL_CHECK(ncclAllReduce(send_buf, recv_buf, count,
                                 static_cast<ncclDataType_t>(dtype),
                                 static_cast<ncclRedOp_t>(op),
                                 static_cast<ncclComm_t>(local_comm_),
                                 local_stream_));

        cudaStreamSynchronize(local_stream_);

        NCCL_CHECK(ncclAllReduce(recv_buf, recv_buf, count,
                                 static_cast<ncclDataType_t>(dtype),
                                 static_cast<ncclRedOp_t>(op),
                                 static_cast<ncclComm_t>(global_comm_),
                                 global_stream_));
    }
#else
    if (send_buf != recv_buf && count > 0) {
        std::memcpy(recv_buf, send_buf, count);
    }
    (void)dtype;
    (void)op;
#endif
}

void HierarchicalAllReduce::synchronize() {
#if NOVA_NCCL_ENABLED
    if (global_stream_) {
        cudaStreamSynchronize(global_stream_);
    }
#endif
}

HierarchicalBarrier::HierarchicalBarrier(void* local_comm,
                                         void* global_comm,
                                         cudaStream_t stream)
    : local_comm_(local_comm),
      global_comm_(global_comm),
      stream_(stream) {}

void HierarchicalBarrier::wait() {
#if NOVA_NCCL_ENABLED
    if (local_comm_) {
        NCCL_CHECK(ncclBarrier(static_cast<ncclComm_t>(local_comm_), stream_));
    }
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }
    if (global_comm_) {
        NCCL_CHECK(ncclBarrier(static_cast<ncclComm_t>(global_comm_), stream_));
    }
#endif
}

void HierarchicalBarrier::synchronize() {
#if NOVA_NCCL_ENABLED
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }
#endif
}

}  // namespace cuda::multinode
