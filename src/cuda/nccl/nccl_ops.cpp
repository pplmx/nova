/**
 * @file nccl_ops.cpp
 * @brief Unified NCCL operations implementation
 */

#include "cuda/nccl/nccl_ops.h"

namespace cuda::nccl {

    NcclOps::NcclOps(NcclContext& ctx)
        : ctx_(ctx),
          all_gather_(std::make_unique<NcclAllGather>(ctx)),
          reduce_scatter_(std::make_unique<NcclReduceScatter>(ctx)) {}

    bool NcclOps::has_nccl() const {
        return ctx_.has_nccl();
    }

    NcclResult NcclOps::all_gather(const void* send_data, void* recv_data, size_t send_count, ncclDataType_t dtype, cudaStream_t stream) {
        return all_gather_->all_gather_async(send_data, recv_data, send_count, dtype, stream);
    }

    NcclResult NcclOps::reduce_scatter(const void* send_data, void* recv_data, size_t recv_count, ncclDataType_t dtype, ncclRedOp_t op, cudaStream_t stream) {
        return reduce_scatter_->reduce_scatter_async(send_data, recv_data, recv_count, dtype, op, stream);
    }

    int NcclOps::device_count() const {
        return ctx_.device_count();
    }

}  // namespace cuda::nccl
