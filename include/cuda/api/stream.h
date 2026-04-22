#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace cuda::api {

class Stream {
public:
    explicit Stream(unsigned int flags = 0) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    ~Stream() {
        cudaStreamDestroy(stream_);
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = {};
    }

    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = {};
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }
    cudaStream_t operator*() const { return stream_; }

    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    bool query() const {
        return cudaStreamQuery(stream_) == cudaSuccess;
    }

private:
    cudaStream_t stream_{};
};

inline std::unique_ptr<Stream> make_stream(unsigned int flags = 0) {
    return std::make_unique<Stream>(flags);
}

class Event {
public:
    explicit Event(unsigned int flags = 0) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }

    ~Event() {
        cudaEventDestroy(event_);
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    Event(Event&& other) noexcept : event_(other.event_) {
        other.event_ = {};
    }

    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            if (event_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = {};
        }
        return *this;
    }

    cudaEvent_t get() const { return event_; }
    cudaEvent_t operator*() const { return event_; }

    void record(const Stream& stream) {
        CUDA_CHECK(cudaEventRecord(event_, stream.get()));
    }

    void synchronize() const {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    bool query() const {
        return cudaEventQuery(event_) == cudaSuccess;
    }

    float elapsed_time(const Event& start, const Event& end) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
        return ms;
    }

private:
    cudaEvent_t event_{};
};

inline std::unique_ptr<Event> make_event(unsigned int flags = 0) {
    return std::make_unique<Event>(flags);
}

} // namespace cuda::api
