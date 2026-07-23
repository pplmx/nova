#include "cuda/production/priority_stream.h"
#include "cuda/device/error.h"

namespace cuda::production {

PriorityStreamPool::PriorityStreamPool(size_t pool_size) : pool_size_(pool_size) {
    low_priority_pool_ = create_streams(pool_size_, StreamPriority::Low);
    normal_priority_pool_ = create_streams(pool_size_, StreamPriority::Normal);
    high_priority_pool_ = create_streams(pool_size_, StreamPriority::High);
}

std::vector<PriorityStream> PriorityStreamPool::create_streams(size_t count,
                                                               StreamPriority priority) {
    std::vector<PriorityStream> streams;
    streams.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        cudaStream_t stream = nullptr;
        int cuda_priority = 0;

        switch (priority) {
            case StreamPriority::Low:
                cuda_priority = MIN_PRIORITY;
                break;
            case StreamPriority::Normal:
                cuda_priority = 0;
                break;
            case StreamPriority::High:
                cuda_priority = MAX_PRIORITY;
                break;
        }

        cudaError_t err = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, cuda_priority);
        if (err == cudaSuccess && stream != nullptr) {
            streams.emplace_back(stream, priority);
        }
    }

    return streams;
}

PriorityStream PriorityStreamPool::acquire(StreamPriority priority) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto& pool = (priority == StreamPriority::Low)   ? low_priority_pool_
                : (priority == StreamPriority::High) ? high_priority_pool_
                                                    : normal_priority_pool_;

    if (!pool.empty()) {
        auto stream = pool.back();
        pool.pop_back();
        return stream;
    }

    cudaStream_t new_stream = nullptr;
    int cuda_priority = (priority == StreamPriority::Low)   ? MIN_PRIORITY
                       : (priority == StreamPriority::High) ? MAX_PRIORITY
                                                            : 0;

    CUDA_CHECK(cudaStreamCreateWithPriority(&new_stream, cudaStreamNonBlocking, cuda_priority));

    return PriorityStream(new_stream, priority);
}

void PriorityStreamPool::release(PriorityStream stream) {
    if (!stream) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto& pool = (stream.priority() == StreamPriority::Low)   ? low_priority_pool_
                : (stream.priority() == StreamPriority::High) ? high_priority_pool_
                                                            : normal_priority_pool_;

    if (pool.size() < pool_size_ * 2) {
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        pool.push_back(stream);
    } else {
        CUDA_CHECK(cudaStreamDestroy(stream.get()));
    }
}

size_t PriorityStreamPool::available_count(StreamPriority priority) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const auto& pool = (priority == StreamPriority::Low)   ? low_priority_pool_
                      : (priority == StreamPriority::High) ? high_priority_pool_
                                                          : normal_priority_pool_;

    return pool.size();
}

size_t PriorityStreamPool::total_available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return low_priority_pool_.size() + normal_priority_pool_.size() +
           high_priority_pool_.size();
}

void PriorityStreamPool::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& stream : low_priority_pool_) {
        if (stream) {
            // cleanup() is a public method that can be called at any time
            // (not just from destructor), so CUDA_CHECK is safe here
            CUDA_CHECK(cudaStreamDestroy(stream.get()));
        }
    }
    for (auto& stream : normal_priority_pool_) {
        if (stream) {
            CUDA_CHECK(cudaStreamDestroy(stream.get()));
        }
    }
    for (auto& stream : high_priority_pool_) {
        if (stream) {
            CUDA_CHECK(cudaStreamDestroy(stream.get()));
        }
    }

    low_priority_pool_.clear();
    normal_priority_pool_.clear();
    high_priority_pool_.clear();
}

}  // namespace cuda::production
