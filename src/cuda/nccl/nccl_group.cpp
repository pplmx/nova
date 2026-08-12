/**
 * @file nccl_group.cpp
 * @brief NCCL group operations implementation
 *
 * Implements batching of multiple NCCL collective operations.
 */

#include "cuda/nccl/nccl_group.h"

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::nccl {

NcclGroupHandle::NcclGroupHandle(NcclContext& ctx, cudaStream_t stream)
    : ctx_(ctx), stream_(stream) {

#ifndef NOVA_NCCL_ENABLED
    return;
#else
    if (ctx.has_nccl()) {
        ncclGroupStart();
    }
#endif
}

NcclGroupHandle::~NcclGroupHandle() {
    if (!executed_ && has_operations()) {
        execute();
    }
}

void NcclGroupHandle::add_all_reduce(
    const void* send_data,
    void* recv_data,
    size_t count,
    ncclDataType_t dtype,
    ncclRedOp_t op) {

    operations_.push_back([
        this, send_data, recv_data, count, dtype, op
    ]() -> NcclResult {
#ifndef NOVA_NCCL_ENABLED
        return NcclResult{.code = ncclInternalError, .error_message = "NCCL not enabled"};
#else
        if (!ctx_.has_nccl()) {
            return NcclResult{.code = ncclInternalError, .error_message = "NCCL not initialized"};
        }
        ncclComm_t comm;
        try {
            comm = ctx_.current_comm();
        } catch (const std::exception& e) {
            return NcclResult{.code = ncclInvalidArgument, .error_message = e.what()};
        }
        return safe_nccl_call(
            [&]() {
                return ncclAllReduce(send_data, recv_data, count, dtype, op, comm, stream_);
            },
            comm, 30000, &ctx_);
#endif
    });
}

void NcclGroupHandle::add_broadcast(
    const void* send_data,
    void* recv_data,
    size_t count,
    ncclDataType_t dtype,
    int root) {

    operations_.push_back([
        this, send_data, recv_data, count, dtype, root
    ]() -> NcclResult {
#ifndef NOVA_NCCL_ENABLED
        return NcclResult{.code = ncclInternalError, .error_message = "NCCL not enabled"};
#else
        if (!ctx_.has_nccl()) {
            return NcclResult{.code = ncclInternalError, .error_message = "NCCL not initialized"};
        }
        ncclComm_t comm;
        try {
            comm = ctx_.current_comm();
        } catch (const std::exception& e) {
            return NcclResult{.code = ncclInvalidArgument, .error_message = e.what()};
        }
        return safe_nccl_call(
            [&]() {
                return ncclBroadcast(send_data, recv_data, count, dtype, root, comm, stream_);
            },
            comm, 30000, &ctx_);
#endif
    });
}

void NcclGroupHandle::add_all_gather(
    const void* send_data,
    void* recv_data,
    size_t send_count,
    ncclDataType_t dtype) {

    operations_.push_back([
        this, send_data, recv_data, send_count, dtype
    ]() -> NcclResult {
#ifndef NOVA_NCCL_ENABLED
        return NcclResult{.code = ncclInternalError, .error_message = "NCCL not enabled"};
#else
        if (!ctx_.has_nccl()) {
            return NcclResult{.code = ncclInternalError, .error_message = "NCCL not initialized"};
        }
        ncclComm_t comm;
        try {
            comm = ctx_.current_comm();
        } catch (const std::exception& e) {
            return NcclResult{.code = ncclInvalidArgument, .error_message = e.what()};
        }
        return safe_nccl_call(
            [&]() {
                return ncclAllGather(send_data, recv_data, send_count, dtype, comm, stream_);
            },
            comm, 30000, &ctx_);
#endif
    });
}

void NcclGroupHandle::add_reduce_scatter(
    const void* send_data,
    void* recv_data,
    size_t recv_count,
    ncclDataType_t dtype,
    ncclRedOp_t op) {

    operations_.push_back([
        this, send_data, recv_data, recv_count, dtype, op
    ]() -> NcclResult {
#ifndef NOVA_NCCL_ENABLED
        return NcclResult{.code = ncclInternalError, .error_message = "NCCL not enabled"};
#else
        if (!ctx_.has_nccl()) {
            return NcclResult{.code = ncclInternalError, .error_message = "NCCL not initialized"};
        }
        ncclComm_t comm;
        try {
            comm = ctx_.current_comm();
        } catch (const std::exception& e) {
            return NcclResult{.code = ncclInvalidArgument, .error_message = e.what()};
        }
        return safe_nccl_call(
            [&]() {
                return ncclReduceScatter(send_data, recv_data, recv_count, dtype, op, comm, stream_);
            },
            comm, 30000, &ctx_);
#endif
    });
}

NcclResult NcclGroupHandle::execute() {
    return execute_internal();
}

NcclResult NcclGroupHandle::execute_internal() {
    if (executed_) {
        return NcclResult{.code = ncclInternalError, .error_message = "Already executed"};
    }

#ifndef NOVA_NCCL_ENABLED
    executed_ = true;
    if (operations_.empty()) {
        return NcclResult{.code = ncclSuccess};
    }
    return NcclResult{.code = ncclInternalError, .error_message = "NCCL not enabled"};
#else
    if (!ctx_.has_nccl()) {
        executed_ = true;
        return NcclResult{.code = ncclInternalError, .error_message = "NCCL not initialized"};
    }

    ncclGroupEnd();
    executed_ = true;

    NcclResult result{.code = ncclSuccess, .error_message = ""};
    for (auto& op : operations_) {
        auto op_result = op();
        if (!op_result.ok() && result.ok()) {
            result = op_result;
        }
    }

    if (result.ok()) {
        cudaError_t sync_err = cudaStreamSynchronize(stream_);
        if (sync_err != cudaSuccess) {
            result.code = ncclSystemError;
            result.error_message = cudaGetErrorString(sync_err);
        }
    }

    return result;
#endif
}

bool NcclGroupHandle::has_operations() const noexcept {
    return !operations_.empty();
}

size_t NcclGroupHandle::operation_count() const noexcept {
    return operations_.size();
}

}  // namespace cuda::nccl
