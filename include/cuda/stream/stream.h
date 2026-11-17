#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "cuda/device/error.h"

namespace cuda::stream {

    /**
     * @class Stream
     * @brief Move-only owner of a cudaStream_t
     *
     * Creates in the constructor and destroys in the destructor. Copying is deleted; moving
     * leaves the source holding nullptr so its destructor does nothing. Moving onto an
     * existing Stream destroys that one's stream first, so no handle is dropped.
     *
     * The file carried no documentation, which for the type every other module uses is worth
     * correcting: a reader who assumed copyability or that the source stays valid after a
     * move would be wrong on both counts.
     *
     * query() reports cudaErrorNotReady as false and rethrows any other error; its result is
     * advisory — a stream that reports idle has finished enqueueing work, not necessarily
     * finished running it on the device.
     */
    class Stream {
    public:
        Stream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }

        explicit Stream(unsigned int flags) { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags)); }

        Stream(unsigned int priority, unsigned int flags) { CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, flags, priority)); }

        ~Stream() {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
        }

        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;

        Stream(Stream&& other) noexcept
            : stream_(other.stream_) {
            other.stream_ = nullptr;
        }

        Stream& operator=(Stream&& other) noexcept {
            if (this != &other) {
                if (stream_) {
                    cudaStreamDestroy(stream_);
                }
                stream_ = other.stream_;
                other.stream_ = nullptr;
            }
            return *this;
        }

        cudaStream_t get() const { return stream_; }
        cudaStream_t operator*() const { return stream_; }

        void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

        bool query() const {
            cudaError_t err = cudaStreamQuery(stream_);
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

    private:
        cudaStream_t stream_{nullptr};
    };

    [[nodiscard]] inline std::unique_ptr<Stream> make_stream() {
        return std::make_unique<Stream>();
    }

    [[nodiscard]] inline std::unique_ptr<Stream> make_stream(unsigned int flags) {
        return std::make_unique<Stream>(flags);
    }

}  // namespace cuda::stream
