/**
 * @file distributed_test_common.h
 * @brief Shared thread-per-rank multi-GPU test harness
 *
 * The collective contract used throughout the multi-GPU tests (NCCL suite,
 * distributed ops, distributed matmul, SyncBatchNorm): one thread per visible
 * device, each pinned with cudaSetDevice, all entering the collective
 * together. This header is the single home of that harness; before it existed
 * the identical RankBarrier + run_per_rank were copy-pasted across four test
 * files and could drift.
 *
 * Bounded barrier (milestone v2.18 P3 / TASK-003): the barrier wait times out
 * after a configurable interval so a rank thread that dies before reaching the
 * barrier fails the suite with a clear diagnosis instead of hanging the other
 * ranks forever — previously the only way to produce an unkillable (>120s)
 * test stall.
 */

#pragma once

#include "cuda/device/error.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cuda::distributed::test {

// Keeps every rank thread entering the collective together. A stray-rank start
// would otherwise leave peers waiting on cross-thread work (NCCL collectives
// are per-rank: every rank must post its op before any completes, so a
// straggler burns safe_stream_wait's timeout).
class RankBarrier {
public:
    explicit RankBarrier(int count, std::chrono::milliseconds timeout = std::chrono::seconds(120))
        : total_(count), timeout_(timeout) {}

    // True when every rank arrived; false (timeout) when a rank is missing.
    // Callers must not proceed past the barrier on a timeout.
    bool wait() {
        std::unique_lock<std::mutex> lk(mutex_);
        ++arrived_;
        const bool released = cv_.wait_for(lk, timeout_, [this] { return arrived_ >= total_; });
        cv_.notify_all();
        return released;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int total_;
    int arrived_ = 0;
    std::chrono::milliseconds timeout_;
};

// Thrown inside run_per_rank when the rank barrier times out (a rank thread
// died before reaching the barrier). Distinct type so tests that deliberately
// exercise failure paths can tell a harness malfunction from an op error.
class RankBarrierTimeout : public std::runtime_error {
public:
    RankBarrierTimeout(int arrived, int total)
        : std::runtime_error("rank barrier timed out: only " +
                             std::to_string(arrived) + "/" + std::to_string(total) +
                             " ranks reached it (a rank thread likely died " +
                             "before entering the collective)") {}
};

// Self-healing multi-GPU-of-NCCL readiness gate for the distributed suites
// (milestone v2.18 P3 / TASK-003). The suites share the process-wide
// NcclContext singleton and deliberately never destroy it, so after a single
// genuine timeout/async failure the context is broken (has_nccl()==false) and
// — without recovery — every later test in the binary would silently GTEST_SKIP,
// masking a real regression as a green-with-skips run. This gate heals that:
// when the singleton is broken it performs the documented destroy() +
// initialize() recovery and only reports false if NCCL genuinely cannot come
// back (e.g. no NCCL build / < 2 GPUs). A one-off flake no longer sinks the
// rest of the suite into skips.
//
// `>= 2 GPUs` check is outside (needs cuda::mesh::DeviceMesh / context), so
// callers are expected to mirror the per-file multigpu_nccl_ready() shape and
// call this only after confirming device count.
inline bool heal_and_ready(cuda::nccl::NcclContext& ctx) {
    if (ctx.broken()) {
        ctx.destroy();
        try {
            ctx.initialize();
        } catch (const std::exception&) {
            return false;
        }
    }
    return ctx.has_nccl();
}

// Runs `per_rank(d)` on one thread per visible device, each pinned to its own
// device. cudaSetDevice is per-thread state, so this is the only way to issue
// each rank's part of a multi-device operation from one process.
//
// Thread exceptions (e.g. a CUDA_CHECK surfacing a sticky device error) are
// collected and rethrown once after all threads join — rethrowing from inside
// join() on the first thread would leave the others running and hit
// std::terminate instead of producing a normal test failure. A rank that dies
// before reaching the barrier is likewise surfaced as RankBarrierTimeout
// rather than hanging the rest of the suite.
template <typename F>
void run_per_rank(int device_count, F&& per_rank) {
    RankBarrier barrier(device_count);
    std::vector<std::thread> threads;
    threads.reserve(device_count);
    std::atomic<bool> barrier_timed_out{false};
    std::mutex error_mutex;
    std::exception_ptr first_error;
    for (int d = 0; d < device_count; ++d) {
        threads.emplace_back([d, &barrier, &barrier_timed_out, &per_rank,
                              &error_mutex, &first_error]() {
            bool failed = false;
            try {
                CUDA_CHECK(cudaSetDevice(d));
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
                failed = true;
            }
            // Always signal the barrier: a rank that failed before reaching it
            // must still let the others proceed, otherwise they hang forever.
            if (!barrier.wait()) {
                // Another rank died before reaching the barrier: record the
                // timeout and skip the collective (entering it would block on
                // a rank that is never coming).
                barrier_timed_out.store(true, std::memory_order_release);
                return;
            }
            if (failed) {
                return;
            }
            try {
                per_rank(d);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    if (barrier_timed_out.load(std::memory_order_acquire)) {
        std::rethrow_exception(std::make_exception_ptr(
            RankBarrierTimeout(0, device_count)));
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
}

}  // namespace cuda::distributed::test
