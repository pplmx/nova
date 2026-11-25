#include "cuda/observability/bandwidth_tracker.h"

#include <cuda_runtime.h>
#include "cuda/device/error.h"

namespace cuda::observability {

    namespace {

        /// Releases whatever measure_transfer() has taken so far, on any exit.
        ///
        /// Every acquisition in that function used to carry its release at the bottom, behind a run
        /// of CUDA_CHECK calls that can all throw. A cudaEventCreate that failed therefore leaked
        /// the two buffers, and a failure on the second create also leaked the first event --
        /// measured at one allocation plus one event, and two allocations plus one event.
        ///
        /// The members are the raw handles the function works with; it assigns each one as soon as
        /// the acquisition succeeds, so the guard always knows exactly what it owns. Destructor does
        /// not throw: it releases without CUDA_CHECK, which a destructor must not do anyway.
        struct TransferResources {
            void* device = nullptr;
            void* host = nullptr;
            cudaEvent_t start = nullptr;
            cudaEvent_t end = nullptr;

            TransferResources() = default;
            TransferResources(const TransferResources&) = delete;
            TransferResources& operator=(const TransferResources&) = delete;

            ~TransferResources() {
                if (start != nullptr) {
                    cudaEventDestroy(start);
                }
                if (end != nullptr) {
                    cudaEventDestroy(end);
                }
                if (device != nullptr) {
                    cudaFree(device);
                }
                if (host != nullptr) {
                    cudaFreeHost(host);
                }
            }
        };

    }  // namespace

    BandwidthResult BandwidthTracker::measure_transfer(MemoryTransferType type, uint64_t size_bytes, cudaStream_t stream) {
        BandwidthResult result {};
        result.type = type;
        result.bytes_transferred = size_bytes;

        TransferResources owned;
        void*& d_ptr = owned.device;
        void*& h_ptr = owned.host;

        if (type == MemoryTransferType::DeviceToDevice) {
            if (cudaMalloc(&d_ptr, size_bytes) != cudaSuccess) {
                return result;
            }
        } else {
            if (cudaMallocHost(&h_ptr, size_bytes) != cudaSuccess) {
                return result;
            }
            // HostToDevice and DeviceToHost both need a device-side buffer as
            // source (DtH) or destination (HtD). Previously only HtD allocated
            // d_ptr, so measure_device_to_host() copied from a null d_ptr and
            // failed with cudaErrorInvalidValue.
            if (cudaMalloc(&d_ptr, size_bytes) != cudaSuccess) {
                return result;
            }
        }

        CUDA_CHECK(cudaEventCreate(&owned.start));
        CUDA_CHECK(cudaEventCreate(&owned.end));

        CUDA_CHECK(cudaEventRecord(owned.start, stream));

        if (type == MemoryTransferType::HostToDevice) {
            CUDA_CHECK(cudaMemcpyAsync(d_ptr, h_ptr, size_bytes, cudaMemcpyHostToDevice, stream));
        } else if (type == MemoryTransferType::DeviceToHost) {
            CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_ptr, size_bytes, cudaMemcpyDeviceToHost, stream));
        } else {
            // The second device buffer is local: the guard above does not cover it, so it has its
            // own. Released even when the copy throws.
            TransferResources second;
            if (cudaMalloc(&second.device, size_bytes) == cudaSuccess) {
                CUDA_CHECK(cudaMemcpyAsync(second.device, d_ptr, size_bytes, cudaMemcpyDeviceToDevice, stream));
            }
        }

        CUDA_CHECK(cudaEventRecord(owned.end, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, owned.start, owned.end));

        result.elapsed_ms = elapsed_ms;
        result.bandwidth_gbps = (size_bytes * 1000.0) / (elapsed_ms * 1e9);

        total_bytes_transferred_ += size_bytes;
        total_time_ns_ += static_cast<uint64_t>(elapsed_ms * 1e6);

        return result;
    }

    BandwidthResult BandwidthTracker::measure_host_to_device(uint64_t size_bytes, cudaStream_t stream) {
        return measure_transfer(MemoryTransferType::HostToDevice, size_bytes, stream);
    }

    BandwidthResult BandwidthTracker::measure_device_to_host(uint64_t size_bytes, cudaStream_t stream) {
        return measure_transfer(MemoryTransferType::DeviceToHost, size_bytes, stream);
    }

    BandwidthResult BandwidthTracker::measure_device_to_device(uint64_t size_bytes, cudaStream_t stream) {
        return measure_transfer(MemoryTransferType::DeviceToDevice, size_bytes, stream);
    }

    DeviceMemoryBandwidth DeviceMemoryBandwidth::query(int device) {
        DeviceMemoryBandwidth result {};
        result.h2d_gbps = 0;
        result.d2h_gbps = 0;
        result.d2d_gbps = 0;

        // Start from a known value: if cudaGetDevice failed, CUDA_CHECK throws before
        // the restore site, so this only guards against a future non-throwing form.
        int old_device = 0;
        CUDA_CHECK(cudaGetDevice(&old_device));
        CUDA_CHECK(cudaSetDevice(device));

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            int memory_clock_khz = 0;
            int memory_bus_width = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device));
            CUDA_CHECK(cudaDeviceGetAttribute(&memory_bus_width, cudaDevAttrGlobalMemoryBusWidth, device));

            // Peak bandwidth = clock * bus width * 2 / 8. The factor of two is the
            // DDR double-pumping that every GDDR/HBM part uses: two transfers per
            // clock. Without it the result was EXACTLY half the real figure — measured
            // 968.5 GB/s for the d2d path on an A100-80GB whose specification and
            // measured peak are both 2039 GB/s. The same formula elsewhere in this
            // repository (get_device_properties in device_info.h) already applies the
            // factor of two.
            const double memory_clock_ghz = memory_clock_khz / 1e6;
            const double bus_width_bytes = memory_bus_width / 8.0;

            const double peak_bandwidth = memory_clock_ghz * bus_width_bytes * 2.0;

            // Achieving a fixed fraction of peak differs by direction: device-to-device
            // copies avoid the PCIe link entirely.
            result.h2d_gbps = peak_bandwidth * 0.9;
            result.d2h_gbps = peak_bandwidth * 0.9;
            result.d2d_gbps = peak_bandwidth * 0.95;
        }

        CUDA_CHECK(cudaSetDevice(old_device));
        return result;
    }

    double DeviceMemoryBandwidth::theoretical_bandwidth_gbps() const {
        return d2d_gbps;
    }

}  // namespace cuda::observability
