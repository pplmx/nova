#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "cuda/device/error.h"
#include "cuda/stream/stream.h"

namespace cuda::stream {

    class Event {
    public:
        Event() { CUDA_CHECK(cudaEventCreate(&event_)); }

        explicit Event(unsigned int flags) { CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags)); }

        ~Event() {
            if (event_) {
                cudaEventDestroy(event_);
            }
        }

        Event(const Event&) = delete;
        Event& operator=(const Event&) = delete;

        Event(Event&& other) noexcept
            : event_(other.event_) {
            other.event_ = nullptr;
        }

        Event& operator=(Event&& other) noexcept {
            if (this != &other) {
                if (event_) {
                    cudaEventDestroy(event_);
                }
                event_ = other.event_;
                other.event_ = nullptr;
            }
            return *this;
        }

        cudaEvent_t get() const { return event_; }
        cudaEvent_t operator*() const { return event_; }

        void record(const Stream& stream) { CUDA_CHECK(cudaEventRecord(event_, stream.get())); }

        void synchronize() const { CUDA_CHECK(cudaEventSynchronize(event_)); }

        bool query() const {
            cudaError_t err = cudaEventQuery(event_);
            if (err == cudaSuccess) {
                return true;
            }
            if (err == cudaErrorNotReady) {
                return false;
            }
            // Don't pass `err` through the CUDA_CHECK macro: it expands to
            // `cudaError_t err = err;`, shadowing this variable and
            // self-initializing from an uninitialized value (UB). Throw directly.
            throw ::cuda::device::CudaException(err, __FILE__, __LINE__);
            return false;
        }

        [[nodiscard]] static float elapsed_time(const Event& start, const Event& end) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
            return ms;
        }

    private:
        cudaEvent_t event_{nullptr};
    };

    [[nodiscard]] inline std::unique_ptr<Event> make_event() {
        return std::make_unique<Event>();
    }

    [[nodiscard]] inline std::unique_ptr<Event> make_event(unsigned int flags) {
        return std::make_unique<Event>(flags);
    }

}  // namespace cuda::stream
