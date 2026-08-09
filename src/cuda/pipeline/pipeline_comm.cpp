/**
 * @file pipeline_comm.cpp
 * @brief Pipeline communicator implementation
 */

#include "cuda/pipeline/pipeline_comm.h"
#include "cuda/device/error.h"

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::pipeline {

PipelineCommunicators::PipelineCommunicators(
    ::cuda::nccl::NcclContext& ctx,
    int tp_degree,
    int dp_degree)
    : tp_degree_(tp_degree),
      dp_degree_(dp_degree),
      supports_split_(false) {

#if NOVA_NCCL_ENABLED
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count >= tp_degree_ * dp_degree_) {
        tp_comms_.resize(tp_degree_);
        dp_comms_.resize(dp_degree_);

        for (int i = 0; i < tp_degree_; ++i) {
            tp_comms_[i] = ctx.get_comm(i);
        }

        for (int i = 0; i < dp_degree_; ++i) {
            dp_comms_[i] = ctx.get_comm(i * tp_degree_);
        }

        supports_split_ = true;
    }
#else
    (void)ctx;
#endif
}

PipelineCommunicators::~PipelineCommunicators() {
}

ncclComm_t PipelineCommunicators::get_tp_comm(int tp_rank) const {
    if (tp_rank >= 0 && tp_rank < static_cast<int>(tp_comms_.size())) {
        return tp_comms_[tp_rank];
    }
    return nullptr;
}

ncclComm_t PipelineCommunicators::get_dp_comm(int dp_rank) const {
    if (dp_rank >= 0 && dp_rank < static_cast<int>(dp_comms_.size())) {
        return dp_comms_[dp_rank];
    }
    return nullptr;
}

int PipelineCommunicators::total_devices() const {
    return tp_degree_ * dp_degree_;
}

int PipelineCommunicators::tp_degree() const {
    return tp_degree_;
}

int PipelineCommunicators::dp_degree() const {
    return dp_degree_;
}

bool PipelineCommunicators::supports_split() const {
    return supports_split_;
}

}  // namespace cuda::pipeline
