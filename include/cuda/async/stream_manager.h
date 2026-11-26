#pragma once

/**
 * @file stream_manager.h
 * @brief Stream lifecycle management with priority configuration
 */

#include <algorithm>
#include <cuda_runtime.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "cuda/stream/stream.h"

namespace cuda::async {

    struct StreamConfig {
        unsigned int priority = 0;
        unsigned int flags = cudaStreamNonBlocking;
    };

    struct PriorityRange {
        int min_priority;
        int max_priority;
    };

    /// The priority bounds for this device, in the order std::clamp requires: min_priority is the
    /// LOWEST numerical value (the highest-priority stream) and max_priority the HIGHEST.
    ///
    /// cudaDeviceGetStreamPriorityRange reports the pair the other way round -- its first out
    /// parameter is the *least* priority, which is the numerically larger value. Passing them
    /// straight into a struct named min/max therefore inverted the range: measured on this machine
    /// min_priority = 0 and max_priority = -5, so std::clamp(priority, 0, -5) had lo > hi and
    /// violated its precondition -- undefined behaviour, and in practice the clamping never took
    /// effect. The suite's own case asserts the inverted relationship (max - min <= 0), which is how
    /// the bug stayed visible without being recognised.
    inline PriorityRange get_stream_priority_range() {
        int least_priority = 0;
        int greatest_priority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));

        PriorityRange range;
        // greatest_priority is the most privileged stream, which CUDA spells with the smaller
        // number; the driver reports it second.
        range.min_priority = greatest_priority;
        range.max_priority = least_priority;
        return range;
    }

    class StreamManager {
    public:
        StreamManager() = default;

        ~StreamManager() {
            streams_.clear();
            priority_to_index_.clear();
        }

        StreamManager(const StreamManager&) = delete;
        StreamManager& operator=(const StreamManager&) = delete;

        StreamManager(StreamManager&&) = default;
        StreamManager& operator=(StreamManager&&) = default;

        cudaStream_t get_stream(int requested_priority = 0) {
            auto it = priority_to_index_.find(requested_priority);
            if (it != priority_to_index_.end()) {
                return streams_[it->second]->get();
            }

            auto range = get_stream_priority_range();
            int clamped_priority = std::clamp(requested_priority, range.min_priority, range.max_priority);

            auto stream = std::make_unique<cuda::stream::Stream>(clamped_priority, cudaStreamNonBlocking);
            cudaStream_t handle = stream->get();

            priority_to_index_[requested_priority] = streams_.size();
            streams_.push_back(std::move(stream));

            return handle;
        }

        void initialize(int num_streams, int max_priority = 0) {
            auto range = get_stream_priority_range();
            int range_size = range.max_priority - range.min_priority + 1;

            if (range_size <= 0) {
                for (int i = 0; i < num_streams; ++i) {
                    get_stream(i);
                }
                return;
            }

            for (int i = 0; i < num_streams; ++i) {
                int offset = (range_size * i) / num_streams;
                int priority = range.min_priority + offset;
                get_stream(priority);
            }
        }

        void synchronize_all() {
            for (auto& stream : streams_) {
                stream->synchronize();
            }
        }

        std::vector<bool> query_all() const {
            std::vector<bool> results;
            results.reserve(streams_.size());
            for (const auto& stream : streams_) {
                results.push_back(stream->query());
            }
            return results;
        }

        cudaStream_t get_high_priority_stream() {
            auto range = get_stream_priority_range();
            return get_stream(range.min_priority);
        }

        cudaStream_t get_low_priority_stream() {
            auto range = get_stream_priority_range();
            return get_stream(range.max_priority);
        }

        size_t num_streams() const { return streams_.size(); }

    private:
        std::vector<std::unique_ptr<cuda::stream::Stream>> streams_;
        std::unordered_map<int, size_t> priority_to_index_;
    };

    inline StreamManager& global_stream_manager() {
        static StreamManager manager;
        return manager;
    }

}  // namespace cuda::async
