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

    PipelineCommunicators::PipelineCommunicators(::cuda::nccl::NcclContext& ctx, int tp_degree, int dp_degree)
        : tp_degree_(tp_degree),
          dp_degree_(dp_degree),
          supports_split_(false) {
        // A non-positive degree made total_devices() report 0 while
        // supports_split() reported true, and a negative one surfaced as a raw
        // std::length_error from vector::resize ("vector::_M_default_append").
        // Reject the configuration with a message that names it.
        if (tp_degree_ <= 0 || dp_degree_ <= 0) {
            throw std::invalid_argument("PipelineCommunicators: tp_degree and dp_degree must be positive, got tp=" + std::to_string(tp_degree_)
                                        + " dp=" + std::to_string(dp_degree_));
        }

#if NOVA_NCCL_ENABLED
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        if (device_count < tp_degree_ * dp_degree_) {
            throw std::invalid_argument("PipelineCommunicators: tp*dp (" + std::to_string(tp_degree_ * dp_degree_) + ") exceeds the " + std::to_string(device_count)
                                        + " available devices");
        }

        tp_comms_.resize(static_cast<size_t>(tp_degree_));
        dp_comms_.resize(static_cast<size_t>(dp_degree_));

        // get_comm() takes a DEVICE id, not an index into a group. The lattice is
        // row-major with tp as the fast axis, so:
        //   TP group of dp_rank d  -> devices {d*tp, d*tp + 1, ..., d*tp + tp-1}
        //   DP group of tp_rank t  -> devices {t, t + tp, t + 2tp, ...}
        // The old code addressed both by position, which for tp_degree == 1 or
        // dp_degree == 1 happened to coincide with the device ids and hid the bug.
        //
        // The accessors below take a rank WITHIN a group and so address one vector
        // entry each; these vectors therefore hold one group's handles, not the
        // whole tp*dp lattice. Filling them with every device of the lattice wrote
        // dp_degree_*tp_degree_ entries into a tp_degree_-sized vector whenever
        // dp_degree_ > 1. Exposed here: the TP handles of DP group 0 and the DP
        // handles of TP rank 0. A caller working in another group cannot be served
        // by an accessor that has no group argument, and must not read these.
        for (int t = 0; t < tp_degree_; ++t) {
            tp_comms_[static_cast<size_t>(t)] = ctx.get_comm(t);
        }
        for (int d = 0; d < dp_degree_; ++d) {
            dp_comms_[static_cast<size_t>(d)] = ctx.get_comm(d * tp_degree_);
        }

        supports_split_ = true;
#else
        // No NCCL: there are no communicators to hand out. Report that honestly
        // instead of claiming the split succeeded with empty vectors.
        (void)ctx;
        supports_split_ = false;
#endif
    }

    PipelineCommunicators::~PipelineCommunicators() {}

    ncclComm_t PipelineCommunicators::get_tp_comm(int tp_rank) const {
        // tp_rank is the rank WITHIN a TP group (0..tp_degree-1); every group shares
        // the same communicator layout, so rank t maps to the t-th handle.
        //
        // Bounded by the vector, not by tp_degree_: the two agree only on the NCCL
        // path, and this getter is reachable on the no-NCCL path where tp_comms_ is
        // empty and tp_degree_ is whatever the caller asked for. Once that was
        // checked against tp_degree_ instead, get_tp_comm(0) on such a build indexed
        // an empty vector and segfaulted (tests/sparse/pipeline_comm_test.cpp,
        // OverSubscribedLatticeIsRejectedOrReported).
        if (tp_rank >= 0 && tp_rank < static_cast<int>(tp_comms_.size())) {
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
