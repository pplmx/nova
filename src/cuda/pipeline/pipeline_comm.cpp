/**
 * @file pipeline_comm.cpp
 * @brief Pipeline communicator implementation
 */

#include "cuda/pipeline/pipeline_comm.h"
#include "cuda/device/error.h"

#include <stdexcept>
#include <string>

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

    // A non-positive degree made total_devices() report 0 while
    // supports_split() reported true, and a negative one surfaced as a raw
    // std::length_error from vector::resize ("vector::_M_default_append").
    // Reject the configuration with a message that names it.
    if (tp_degree_ <= 0 || dp_degree_ <= 0) {
        throw std::invalid_argument(
            "PipelineCommunicators: tp_degree and dp_degree must be positive, got tp=" +
            std::to_string(tp_degree_) + " dp=" + std::to_string(dp_degree_));
    }

#if NOVA_NCCL_ENABLED
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count < tp_degree_ * dp_degree_) {
        throw std::invalid_argument(
            "PipelineCommunicators: tp*dp (" + std::to_string(tp_degree_ * dp_degree_) +
            ") exceeds the " + std::to_string(device_count) + " available devices");
    }

    tp_comms_.resize(static_cast<size_t>(tp_degree_));
    dp_comms_.resize(static_cast<size_t>(dp_degree_));

    // get_comm() takes a DEVICE id, not an index into a group. The lattice is
    // row-major with tp as the fast axis, so:
    //   TP group of dp_rank d  -> devices {d*tp, d*tp + 1, ..., d*tp + tp-1}
    //   DP group of tp_rank t  -> devices {t, t + tp, t + 2tp, ...}
    // The old code addressed both by position, which for tp_degree == 1 or
    // dp_degree == 1 happened to coincide with the device ids and hid the bug.
    for (int t = 0; t < tp_degree_; ++t) {
        dp_comms_[static_cast<size_t>(t)] = ctx.get_comm(t);
    }
    for (int d = 0; d < dp_degree_; ++d) {
        for (int t = 0; t < tp_degree_; ++t) {
            tp_comms_[static_cast<size_t>(d * tp_degree_ + t)] =
                ctx.get_comm(d * tp_degree_ + t);
        }
    }

    supports_split_ = true;
#else
    // No NCCL: there are no communicators to hand out. Report that honestly
    // instead of claiming the split succeeded with empty vectors.
    (void)ctx;
    supports_split_ = false;
#endif
}

PipelineCommunicators::~PipelineCommunicators() {
}

ncclComm_t PipelineCommunicators::get_tp_comm(int tp_rank) const {
    // tp_rank is the rank WITHIN a TP group (0..tp_degree-1); every group shares
    // the same communicator layout, so rank t maps to the t-th handle.
    if (tp_rank >= 0 && tp_rank < tp_degree_) {
        return tp_comms_[static_cast<size_t>(tp_rank)];
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
