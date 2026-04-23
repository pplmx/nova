#pragma once

/**
 * @file async_copy.h
 * @brief Async copy primitives with stream support
 */

#include <cuda_runtime.h>

#include <cstddef>

#include "cuda/async/pinned_memory.h"
#include "cuda/device/error.h"
#include "cuda/stream/stream.h"

namespace cuda::async {

enum class CopyDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost
};

struct CopyContext {
    const void* src = nullptr;
    void* dst = nullptr;
    size_t bytes = 0;
    CopyDirection direction = CopyDirection::HostToDevice;
    cudaStream_t stream = nullptr;
    const char* label = nullptr;
};

inline cudaMemcpyKind get_memcpy_kind(CopyDirection dir) {
    switch (dir) {
        case CopyDirection::HostToDevice:
            return cudaMemcpyHostToDevice;
        case CopyDirection::DeviceToHost:
            return cudaMemcpyDeviceToHost;
        case CopyDirection::DeviceToDevice:
            return cudaMemcpyDeviceToDevice;
        case CopyDirection::HostToHost:
        default:
            return cudaMemcpyHostToHost;
    }
}

inline void async_copy(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
}

inline void async_copy(void* dst, const void* src, size_t bytes,
                      CopyDirection dir, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, get_memcpy_kind(dir), stream));
}

inline void async_copy(const CopyContext& ctx) {
    CUDA_CHECK(cudaMemcpyAsync(ctx.dst, ctx.src, ctx.bytes,
                               get_memcpy_kind(ctx.direction), ctx.stream));
}

inline void async_copy_h2d(void* device_ptr, const void* host_ptr,
                           size_t bytes, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(device_ptr, host_ptr, bytes,
                               cudaMemcpyHostToDevice, stream));
}

inline void async_copy_d2h(void* host_ptr, const void* device_ptr,
                           size_t bytes, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(host_ptr, device_ptr, bytes,
                               cudaMemcpyDeviceToHost, stream));
}

inline void async_copy_d2d(void* dst_device, const void* src_device,
                           size_t bytes, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst_device, src_device, bytes,
                               cudaMemcpyDeviceToDevice, stream));
}

template <typename T>
inline void async_copy_from_pinned(T* device_ptr, const PinnedMemory<T>& pinned,
                                   size_t offset, size_t count, cudaStream_t stream) {
    if (!pinned.data() || !device_ptr || count == 0) {
        return;
    }
    const T* src = pinned.data() + offset;
    size_t bytes = count * sizeof(T);
    CUDA_CHECK(cudaMemcpyAsync(device_ptr, src, bytes,
                               cudaMemcpyHostToDevice, stream));
}

template <typename T>
inline void async_copy_to_pinned(PinnedMemory<T>& pinned, const T* device_ptr,
                                 size_t offset, size_t count, cudaStream_t stream) {
    if (!pinned.data() || !device_ptr || count == 0) {
        return;
    }
    T* dst = pinned.data() + offset;
    size_t bytes = count * sizeof(T);
    CUDA_CHECK(cudaMemcpyAsync(dst, device_ptr, bytes,
                               cudaMemcpyDeviceToHost, stream));
}

inline void async_copy_2d(void* dst, size_t dpitch,
                          const void* src, size_t spitch,
                          size_t width, size_t height,
                          CopyDirection dir, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch,
                                 width, height, get_memcpy_kind(dir), stream));
}

}  // namespace cuda::async