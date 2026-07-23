/**
 * @file pipeline_scheduler.cpp
 * @brief Pipeline scheduler implementation
 */

#include "cuda/pipeline/pipeline_scheduler.h"
#include "cuda/device/error.h"

#include <algorithm>
#include <cmath>

namespace cuda::pipeline {

PipelineScheduler::PipelineScheduler(
    ::cuda::nccl::NcclContext& ctx,
    int num_stages,
    int num_microbatches,
    int microbatch_size)
    : ctx_(ctx),
      num_stages_(num_stages),
      num_microbatches_(num_microbatches),
      microbatch_size_(microbatch_size),
      schedule_type_(ScheduleType::OneForwardOneBackward) {

    CUDA_CHECK(cudaEventCreate(&completion_event_));
}

PipelineScheduler::~PipelineScheduler() {
    // Not using CUDA_CHECK here: throwing in a destructor is undefined behavior
    cudaEventDestroy(completion_event_);
}

void PipelineScheduler::set_forward_fn(ForwardFn fn) {
    forward_fn_ = std::move(fn);
}

void PipelineScheduler::set_backward_fn(BackwardFn fn) {
    backward_fn_ = std::move(fn);
}

void PipelineScheduler::set_schedule_type(ScheduleType type) {
    schedule_type_ = type;
}

void PipelineScheduler::run() {
    if (schedule_type_ == ScheduleType::OneForwardOneBackward) {
        schedule_1f1b();
    } else {
        schedule_interleaved();
    }
    CUDA_CHECK(cudaEventRecord(completion_event_, 0));
    completed_ = true;
}

void PipelineScheduler::wait() {
    if (completed_) {
        CUDA_CHECK(cudaEventSynchronize(completion_event_));
    }
}

void PipelineScheduler::schedule_1f1b() {
    if (!forward_fn_ || !backward_fn_) {
        return;
    }

    int K = num_stages_;
    int M = num_microbatches_;

    for (int step = 0; step < M + K - 1; ++step) {
        if (step < K) {
            run_forward(0, step);
        } else if (step < M) {
            run_backward(step - K, M - 1);
            run_forward(step, step - K + 1);
        } else {
            run_backward(step - K, M + K - 1 - step);
        }
    }
}

void PipelineScheduler::schedule_interleaved() {
    if (!forward_fn_ || !backward_fn_) {
        return;
    }

    int K = num_stages_;
    int M = num_microbatches_;
    int num_chunks = K;

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        for (int mb = chunk; mb < M; mb += num_chunks) {
            for (int stage = 0; stage < K; ++stage) {
                run_forward(stage, mb);
            }
        }

        for (int mb = chunk; mb < M; mb += num_chunks) {
            for (int stage = K - 1; stage >= 0; --stage) {
                run_backward(stage, mb);
            }
        }
    }
}

void PipelineScheduler::run_forward(int stage, int microbatch) {
    if (forward_fn_ && stage >= 0 && stage < num_stages_ && microbatch >= 0 && microbatch < num_microbatches_) {
        forward_fn_(stage, microbatch);
    }
}

void PipelineScheduler::run_backward(int stage, int microbatch) {
    if (backward_fn_ && stage >= 0 && stage < num_stages_ && microbatch >= 0 && microbatch < num_microbatches_) {
        backward_fn_(stage, microbatch);
    }
}

int PipelineScheduler::num_stages() const {
    return num_stages_;
}

int PipelineScheduler::num_microbatches() const {
    return num_microbatches_;
}

float PipelineScheduler::bubble_overhead_percent() const {
    if (num_stages_ <= 1) {
        return 0.0f;
    }

    int K = num_stages_;
    int M = num_microbatches_;

    float total_slots = static_cast<float>(M + K - 1);
    float bubble_slots = static_cast<float>(K - 1);
    float bubble_percent = (bubble_slots / total_slots) * 100.0f;

    return bubble_percent;
}

int recommend_microbatches(int num_stages) {
    constexpr int MIN_MICROBATCHES = 4;
    int recommended = num_stages * MIN_MICROBATCHES;
    return std::max(recommended, 16);
}

}  // namespace cuda::pipeline
