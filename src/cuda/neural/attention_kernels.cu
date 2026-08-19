/**
 * @file attention_kernels.cu
 * @brief Device multi-head SDPA kernels (milestone v2.29, TASK-044/045)
 *
 * Milestone v2.29 (DEC-015): the public sdpa_forward/sdpa_backward in
 * tensor_parallel_attention.cpp compute multi-head scaled-dot-product
 * attention ON THE HOST — full D2H copies, host loops over
 * m x local_heads x m x head_dim, H2D copies back — once per block per
 * train_step. The v2.26 file header deferred this to "a later performance
 * pass"; Round-28 LEARN (M3) named it the binding deep-transformer
 * training-latency constraint (same D2H/H2D family v2.27 removed from the
 * optimizer). This module ports sdpa_forward/sdpa_backward to CUDA kernels
 * behind the detail:: entry points declared in tensor_parallel_attention.h,
 * mirroring optimizers_kernels.cu: the P2 implementation replaces the host
 * round-trips with per-(row, head) kernels (shared-memory max-subtracted
 * softmax over m keys, block reductions, exact port of the host algorithm).
 *
 * P1 (TASK-044): these throw — RED against attention_kernels_test.cpp.
 */

#include "cuda/neural/tensor_parallel_attention.h"

#include <stdexcept>

namespace cuda::neural::detail {

void sdpa_forward_device(const float* q, const float* k, const float* v,
                         float* out, int m, int local_heads, int head_dim,
                         cudaStream_t stream) {
    (void)q; (void)k; (void)v; (void)out; (void)m; (void)local_heads;
    (void)head_dim; (void)stream;
    throw std::logic_error(
        "detail::sdpa_forward_device: not implemented (milestone v2.29 P1 "
        "stub — device kernels land in P2, TASK-045)");
}

void sdpa_backward_device(const float* q, const float* k, const float* v,
                          const float* dout, float* dq, float* dk, float* dv,
                          int m, int local_heads, int head_dim,
                          cudaStream_t stream) {
    (void)q; (void)k; (void)v; (void)dout; (void)dq; (void)dk; (void)dv;
    (void)m; (void)local_heads; (void)head_dim; (void)stream;
    throw std::logic_error(
        "detail::sdpa_backward_device: not implemented (milestone v2.29 P1 "
        "stub — device kernels land in P2, TASK-045)");
}

}  // namespace cuda::neural::detail
