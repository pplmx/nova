#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "cuda/device/error.h"

namespace cuda::production {

    enum class StreamPriority {
        Low = -1,
        Normal = 0,
        High = 1,
    };

    class PriorityStream {
    public:
        PriorityStream() = default;
        PriorityStream(cudaStream_t stream, StreamPriority priority)
            : stream_(stream),
              priority_(priority) {}

        [[nodiscard]] cudaStream_t get() const { return stream_; }
        [[nodiscard]] StreamPriority priority() const { return priority_; }
        [[nodiscard]] explicit operator bool() const { return stream_ != nullptr; }

    private:
        cudaStream_t stream_ {nullptr};
        StreamPriority priority_ {StreamPriority::Normal};
    };

    /**
     * @class PriorityStreamPool
     * @brief Fixed-size pool of CUDA streams at three priorities
     *
     * acquire() takes a stream out of the pool (creating one if the pool cannot supply it) and
     * release() puts it back. The two are meant to be paired per handle: release() ignores a
     * handle the pool already holds, so releasing the same stream twice is a no-op rather than a
     * duplicate. That guard exists because the alternative was measured — before it, a second
     * release stored the handle twice and the destructor called cudaStreamDestroy on both, which
     * dumped core.
     *
     * A stream the pool never handed out should not be released to it. When the pool is already
     * at twice its nominal size, release() destroys the stream instead of storing it.
     */
    class PriorityStreamPool {
    public:
        static constexpr int MIN_PRIORITY = -2;
        static constexpr int MAX_PRIORITY = 0;

        PriorityStreamPool() = default;

        explicit PriorityStreamPool(size_t pool_size);

        ~PriorityStreamPool() { cleanup(); }

        PriorityStreamPool(const PriorityStreamPool&) = delete;
        PriorityStreamPool& operator=(const PriorityStreamPool&) = delete;

        /// Takes a stream for the given priority, creating one if the pool is empty.
        [[nodiscard]] PriorityStream acquire(StreamPriority priority);

        /// Returns a stream to its priority pool. A null handle and one the pool already holds
        /// are both ignored.
        void release(PriorityStream stream);

        [[nodiscard]] size_t available_count(StreamPriority priority) const;
        [[nodiscard]] size_t total_available() const;

        /// Synchronises and destroys every pooled stream. Safe to call more than once, and
        /// called by the destructor.
        void cleanup();

    private:
        std::vector<PriorityStream> create_streams(size_t count, StreamPriority priority);

        std::vector<PriorityStream> low_priority_pool_;
        std::vector<PriorityStream> normal_priority_pool_;
        std::vector<PriorityStream> high_priority_pool_;

        mutable std::mutex mutex_;
        size_t pool_size_ = 4;
    };

}  // namespace cuda::production
