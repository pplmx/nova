/**
 * @file common.cu
 * @brief MeshStreams implementation
 */

#include "cuda/distributed/common.h"
#include "cuda/device/error.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace cuda::distributed {

    MeshStreams& MeshStreams::instance() {
        // Heap-allocated singleton that is intentionally never destroyed.
        //
        // A function-local static's destructor runs at process exit, and the
        // destructor here calls cudaEventDestroy/cudaStreamDestroy. Driving libcuda
        // during __run_exit_handlers, after the per-thread driver context is already
        // being torn down, makes cuEventDestroy SEGV inside the driver — observed as
        // an "end-of-run SIGSEGV" (EXIT=139) after an otherwise fully-green suite,
        // exactly like the NcclContext singleton fixed in the same way.
        //
        // The CUDA driver reclaims all GPU resources when the process exits, so
        // orderly teardown at exit is unnecessary.
        static MeshStreams* instance = new MeshStreams();
        return *instance;
    }

    MeshStreams::~MeshStreams() {
        std::lock_guard<std::mutex> lock(mutex_);
        // Destroy events first
        for (auto event : events_) {
            if (event) {
                cudaEventDestroy(event);
            }
        }
        events_.clear();

        // Then destroy streams
        for (auto stream : streams_) {
            if (stream) {
                cudaStreamDestroy(stream);
            }
        }
        streams_.clear();

        initialized_ = false;
    }

    void MeshStreams::reset() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto event : events_) {
            if (event) {
                (void)cudaEventDestroy(event);
            }
        }
        events_.clear();

        for (auto stream : streams_) {
            if (stream) {
                (void)cudaStreamDestroy(stream);
            }
        }
        streams_.clear();

        device_count_ = 0;
        initialized_ = false;
        (void)cudaGetLastError();
    }

    void MeshStreams::initialize(int device_count) {
        // Held for the whole call, not just the check. Releasing it between the resize
        // and the fill would let a concurrent get_stream() observe a vector of null
        // handles, and letting two threads past `initialized_` would create two sets of
        // streams and leak one.
        std::lock_guard<std::mutex> lock(mutex_);

        if (initialized_) {
            return;
        }

        if (device_count <= 0) {
            throw std::invalid_argument("Device count must be positive");
        }

        // Publish the count only once the arrays exist, and only after they are filled:
        // initialized_ is the flag every reader gates on, so it is set last.
        streams_.resize(static_cast<size_t>(device_count));
        events_.resize(static_cast<size_t>(device_count));
        device_count_ = device_count;

        // The two CUDA_CHECKs per iteration throw, and a failure part-way through used to leave
        // every stream and event created before it unreferenced: the vectors held them, but
        // nothing released them, because the destructor only runs on a fully constructed object.
        // Release the partial set on the way out.
        // cudaSetDevice is per-thread state, so the loop has to put the caller's device back:
        // leaving it on the last device made every subsequent kernel and allocation of the calling
        // thread land there.
        int caller_device = -1;
        const bool had_caller_device = cudaGetDevice(&caller_device) == cudaSuccess;

        try {
            for (int i = 0; i < device_count; ++i) {
                // Set device before creating stream/event
                CUDA_CHECK(cudaSetDevice(i));

                // Create non-blocking stream for concurrency
                CUDA_CHECK(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));

                // Create event with default flags
                CUDA_CHECK(cudaEventCreate(&events_[i]));
            }
        } catch (...) {
            for (int i = 0; i < device_count; ++i) {
                if (events_[i] != nullptr) {
                    (void)cudaEventDestroy(events_[i]);
                    events_[i] = nullptr;
                }
                if (streams_[i] != nullptr) {
                    (void)cudaStreamDestroy(streams_[i]);
                    streams_[i] = nullptr;
                }
            }
            if (had_caller_device) {
                (void)cudaSetDevice(caller_device);
            }
            throw;
        }

        if (had_caller_device) {
            (void)cudaSetDevice(caller_device);
        }

        initialized_ = true;
    }

    cudaStream_t MeshStreams::get_stream(int device) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            throw std::runtime_error("MeshStreams not initialized");
        }
        if (device < 0 || device >= device_count_) {
            throw std::out_of_range("Device index out of range");
        }
        return streams_[device];
    }

    cudaEvent_t MeshStreams::get_event(int device) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            throw std::runtime_error("MeshStreams not initialized");
        }
        if (device < 0 || device >= device_count_) {
            throw std::out_of_range("Device index out of range");
        }
        return events_[device];
    }

    void MeshStreams::synchronize_all() {
        // Copy the handles under the lock and drive CUDA outside it: cudaStreamSynchronize
        // can block for a long time, and holding the mutex across it would block every
        // get_stream() in the process. The copy is safe because initialize() is the only
        // writer and it publishes under the same lock.
        std::vector<cudaStream_t> streams;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!initialized_) {
                return;
            }
            streams = streams_;
        }

        int caller_device = -1;
        const bool had_caller_device = cudaGetDevice(&caller_device) == cudaSuccess;

        for (size_t i = 0; i < streams.size(); ++i) {
            CUDA_CHECK(cudaSetDevice(static_cast<int>(i)));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }

        // Synchronising every device's streams must not leave the caller pinned to the last one.
        if (had_caller_device) {
            (void)cudaSetDevice(caller_device);
        }
    }

    void MeshStreams::wait_event(int dst_device, int src_device, cudaEvent_t event) {
        cudaStream_t dst_stream = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!initialized_) {
                throw std::runtime_error("MeshStreams not initialized");
            }

            if (dst_device < 0 || dst_device >= device_count_ || src_device < 0 || src_device >= device_count_) {
                throw std::out_of_range("Device index out of range");
            }
            dst_stream = streams_[static_cast<size_t>(dst_device)];
        }

        // Set destination device and make it wait on source device's event. Outside the
        // lock: cudaStreamWaitEvent is a driver call and must not block other readers.
        int caller_device = -1;
        const bool had_caller_device = cudaGetDevice(&caller_device) == cudaSuccess;

        CUDA_CHECK(cudaSetDevice(dst_device));
        CUDA_CHECK(cudaStreamWaitEvent(dst_stream, event, 0));

        if (had_caller_device) {
            (void)cudaSetDevice(caller_device);
        }
    }

}  // namespace cuda::distributed
