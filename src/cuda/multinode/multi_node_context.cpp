#include "cuda/multinode/multi_node_context.h"
#include "cuda/device/error.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace cuda::multinode {

    namespace {

        MultiNodeContext* global_instance = nullptr;

        // Byte size of one element for the ncclDataType_t enum values used across the
        // codebase. Used by the single-rank/null-comm fallback to copy the right
        // number of bytes (the previous copy used `count` bytes, truncating non-char
        // buffers).
        size_t nccl_type_size(int dtype) {
            switch (dtype) {
                case 1:  // ncclInt32
                case 3:  // ncclFloat32
                case 7:  // ncclUint32
                    return 4;
                case 2:  // ncclHalf
                    return 2;
                case 4:  // ncclFloat64
                case 5:  // ncclInt64
                case 6:  // ncclUint64
                    return 8;
                case 0:  // ncclChar
                case 8:  // ncclInt8
                case 9:  // ncclUint8
                    return 1;
                default:
                    return 4;
            }
        }

    }  // namespace

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

        // num_nodes_ is the quotient of these two, so local_size 0 is an integer division by
        // zero. That is not an exception anything can catch: measured, initialize() with
        // world_size 4 and local_size 0 aborted the process with SIGFPE (floating point
        // exception, core dumped, exit 136) before the caller could react. Validate here, where
        // the mistake is still attributable, and name it.
        if (config.local_size <= 0) {
            throw std::invalid_argument("MultiNodeContext::initialize: local_size must be positive (it divides world_size to yield the node count)");
        }
        if (config.world_size <= 0) {
            throw std::invalid_argument("MultiNodeContext::initialize: world_size must be positive");
        }
        if (config.local_size > config.world_size) {
            // num_nodes_ would truncate to 0, which reads as "no nodes" while the context is
            // initialized and claiming a world size.
            throw std::invalid_argument("MultiNodeContext::initialize: local_size (" + std::to_string(config.local_size) + ") exceeds world_size ("
                                        + std::to_string(config.world_size) + ")");
        }
        if (config.world_rank < 0 || config.world_rank >= config.world_size) {
            throw std::invalid_argument("MultiNodeContext::initialize: world_rank " + std::to_string(config.world_rank) + " is outside [0, " + std::to_string(config.world_size)
                                        + ")");
        }
        if (config.local_rank < 0 || config.local_rank >= config.local_size) {
            throw std::invalid_argument("MultiNodeContext::initialize: local_rank " + std::to_string(config.local_rank) + " is outside [0, " + std::to_string(config.local_size)
                                        + ")");
        }

        global_rank_ = config.world_rank;
        global_size_ = config.world_size;
        local_rank_ = config.local_rank;
        local_size_ = config.local_size;
        node_id_ = config.node_id;
        num_nodes_ = config.world_size / config.local_size;

        CUDA_CHECK(cudaSetDevice(local_rank_));

        // Both streams and the NCCL communicators below are released only by shutdown(), which
        // begins `if (!initialized_) return;`. A failure before initialized_ is set therefore left
        // whatever had been created with no way to reach it -- measured: failing the second stream
        // creation leaked global_stream_ permanently, and a later initialize() overwrote the handle
        // instead of reusing it. The acquisitions run without throwing and are released here.
        auto release_streams = [this] {
            if (local_stream_) {
                cudaStreamDestroy(local_stream_);
                local_stream_ = nullptr;
            }
            if (global_stream_) {
                cudaStreamDestroy(global_stream_);
                global_stream_ = nullptr;
            }
        };

        cudaError_t stream_err = cudaStreamCreate(&global_stream_);
        if (stream_err == cudaSuccess) {
            stream_err = cudaStreamCreate(&local_stream_);
        }
        if (stream_err != cudaSuccess) {
            release_streams();
            CUDA_CHECK(stream_err);
        }

#if NOVA_NCCL_ENABLED
        try {
            initialize_intra_node_comm();
            initialize_inter_node_comm();
        } catch (...) {
            release_streams();
            throw;
        }
#else
        if (config.world_size > 1) {
            std::fprintf(stderr,
                         "Warning: Multi-node requested but NCCL not available. "
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

    HierarchicalAllReduce::HierarchicalAllReduce(void* local_comm, void* global_comm, cudaStream_t local_stream, cudaStream_t global_stream)
        : local_comm_(local_comm),
          global_comm_(global_comm),
          local_stream_(local_stream),
          global_stream_(global_stream) {}

    void HierarchicalAllReduce::all_reduce(const void* send_buf, void* recv_buf, size_t count, int dtype, int op) {
#if NOVA_NCCL_ENABLED
        if (local_comm_ && global_comm_) {
            NCCL_CHECK(
                ncclAllReduce(send_buf, recv_buf, count, static_cast<ncclDataType_t>(dtype), static_cast<ncclRedOp_t>(op), static_cast<ncclComm_t>(local_comm_), local_stream_));

            CUDA_CHECK(cudaStreamSynchronize(local_stream_));

            NCCL_CHECK(
                ncclAllReduce(recv_buf, recv_buf, count, static_cast<ncclDataType_t>(dtype), static_cast<ncclRedOp_t>(op), static_cast<ncclComm_t>(global_comm_), global_stream_));
            return;
        }
        // Null comms (e.g. single-rank or test stubs): all_reduce is the identity,
        // so fall back to a local copy of the correct element count.
        if (send_buf != recv_buf && count > 0) {
            std::memcpy(recv_buf, send_buf, count * nccl_type_size(dtype));
        }
#else
        if (send_buf != recv_buf && count > 0) {
            std::memcpy(recv_buf, send_buf, count * nccl_type_size(dtype));
        }
        (void)op;
#endif
    }

    void HierarchicalAllReduce::synchronize() {
#if NOVA_NCCL_ENABLED
        if (global_stream_) {
            CUDA_CHECK(cudaStreamSynchronize(global_stream_));
        }
#endif
    }

    HierarchicalBarrier::HierarchicalBarrier(void* local_comm, void* global_comm, cudaStream_t stream)
        : local_comm_(local_comm),
          global_comm_(global_comm),
          stream_(stream) {}

    void HierarchicalBarrier::wait() {
#if NOVA_NCCL_ENABLED
        if (stream_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
        if (global_comm_) {
            int dummy = 0;
            NCCL_CHECK(ncclAllReduce(&dummy, &dummy, 1, ncclInt, ncclMin, static_cast<ncclComm_t>(global_comm_), stream_));
        }
#endif
    }

    void HierarchicalBarrier::synchronize() {
#if NOVA_NCCL_ENABLED
        if (stream_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
#endif
    }

}  // namespace cuda::multinode
