#pragma once

/**
 * @file pipeline_scheduler.h
 * @brief Pipeline parallelism scheduling
 *
 * Implements 1F1B (one-forward-one-backward) and interleaved
 * schedules for training deep models across multiple GPUs.
 */

#include "cuda/nccl/nccl_context.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <vector>

namespace cuda::pipeline {

    /**
     * @enum ScheduleType
     * @brief Pipeline schedule type
     */
    enum class ScheduleType {
        /** Standard 1F1B: one forward, one backward per iteration
         *  Minimizes memory, moderate bubble overhead */
        OneForwardOneBackward,

        /** Interleaved: multiple forwards before corresponding backwards
         *  Reduces bubbles but increases memory usage */
        Interleaved
    };

    /**
     * @class PipelineScheduler
     * @brief Scheduler for pipeline parallelism
     *
     * Orchestrates forward and backward passes across pipeline stages.
     * Implements 1F1B schedule with bubble hiding.
     *
     * @example
     * @code
     * PipelineScheduler scheduler(ctx, num_stages=4, num_microbatches=16);
     * scheduler.set_forward_fn([&](int stage, int mb) { run_forward(stage, mb); });
     * scheduler.set_backward_fn([&](int stage, int mb) { run_backward(stage, mb); });
     * scheduler.run();
     * scheduler.wait();
     * @endcode
     */
    class PipelineScheduler {
    public:
        using ForwardFn = std::function<void(int stage, int microbatch)>;
        using BackwardFn = std::function<void(int stage, int microbatch)>;

        /**
         * @brief Construct pipeline scheduler
         *
         * All three counts must be positive, and they are checked here rather than trusted.
         * Nothing validated them before and every derived value was nonsense in a different
         * way: with num_microbatches = 0 bubble_overhead_percent() reported exactly 100%,
         * because its numerator K - 1 equalled its denominator M + K - 1 for a pipeline that
         * would run nothing; with num_stages negative it reported 0% because the
         * `num_stages_ <= 1` early return swallowed it; and run() iterated `stage < -2`,
         * doing no work while still recording the completion event and setting
         * completed_ = true. A caller reading those numbers had no way to tell a
         * configuration mistake from a measurement.
         *
         * @param ctx NCCL context
         * @param num_stages Number of pipeline stages; must be positive
         * @param num_microbatches Number of micro batches; must be positive
         * @param microbatch_size Samples per microbatch; must be positive
         * @throws std::invalid_argument if any count is not positive
         */
        PipelineScheduler(::cuda::nccl::NcclContext& ctx, int num_stages, int num_microbatches, int microbatch_size);

        ~PipelineScheduler();

        // Non-copyable
        PipelineScheduler(const PipelineScheduler&) = delete;
        PipelineScheduler& operator=(const PipelineScheduler&) = delete;

        /**
         * @brief Set forward pass function
         * @param fn Function to execute forward pass
         */
        void set_forward_fn(ForwardFn fn);

        /**
         * @brief Set backward pass function
         * @param fn Function to execute backward pass
         */
        void set_backward_fn(BackwardFn fn);

        /**
         * @brief Set schedule type
         * @param type 1F1B or interleaved
         */
        void set_schedule_type(ScheduleType type);

        /**
         * @brief Run the pipeline schedule
         */
        void run();

        /**
         * @brief Wait for completion
         */
        void wait();

        /**
         * @brief Get number of stages
         */
        [[nodiscard]] int num_stages() const;

        /**
         * @brief Get number of microbatches
         */
        [[nodiscard]] int num_microbatches() const;

        /**
         * @brief Calculate bubble overhead percentage
         * @return Bubble overhead as percentage
         */
        [[nodiscard]] float bubble_overhead_percent() const;

    private:
        void schedule_1f1b();
        void schedule_interleaved();
        void run_forward(int stage, int microbatch);
        void run_backward(int stage, int microbatch);

        ::cuda::nccl::NcclContext& ctx_;
        int num_stages_;
        int num_microbatches_;
        int microbatch_size_;
        ScheduleType schedule_type_;
        ForwardFn forward_fn_;
        BackwardFn backward_fn_;
        cudaEvent_t completion_event_;
        bool completed_ = false;
    };

    /**
     * @brief Calculate recommended number of microbatches for given stages
     * @param num_stages Number of pipeline stages
     * @return Recommended microbatch count (M >= 4*K rule)
     */
    [[nodiscard]]
    int recommend_microbatches(int num_stages);

}  // namespace cuda::pipeline
