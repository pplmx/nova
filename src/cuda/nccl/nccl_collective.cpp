/**
 * @file nccl_collective.cpp
 * @brief NcclCollective base class implementation
 */

#include "cuda/nccl/nccl_collective.h"

namespace cuda::nccl {

NcclCollective::NcclCollective(NcclContext& ctx)
    : ctx_(ctx) {}

bool NcclCollective::has_nccl() const {
    return ctx_.has_nccl();
}

ncclComm_t NcclCollective::get_comm(int device) const {
    return ctx_.get_comm(device);
}

ncclComm_t NcclCollective::current_comm() const {
    return ctx_.current_comm();
}

cudaStream_t NcclCollective::get_stream(int device) const {
    return ctx_.get_stream(device);
}

int NcclCollective::device_count() const {
    return ctx_.device_count();
}

}  // namespace cuda::nccl
