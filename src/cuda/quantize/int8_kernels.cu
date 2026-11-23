#include <cuda/quantize/int8_kernels.hpp>
#include "cuda/device/error.h"
#include "cuda/memory/unique_ptr.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <vector>
#include <algorithm>

namespace nova {
    namespace quantize {
        namespace cuda {

            namespace detail {

                template <bool SYMMETRIC>
                __global__ void quantize_f32_to_int8_kernel(const float* __restrict__ src, int8_t* __restrict__ dst, size_t n, float scale, float zero_point) {
                    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= n) {
                        return;
                    }

                    float val = src[idx];
                    int32_t quantized;

                    if (SYMMETRIC) {
                        quantized = static_cast<int32_t>(__float2int_rn(val / scale));
                    } else {
                        quantized = static_cast<int32_t>(__float2int_rn((val - zero_point) / scale));
                    }

                    quantized = max(-127, min(127, quantized));
                    dst[idx] = static_cast<int8_t>(quantized);
                }

                template <bool SYMMETRIC>
                __global__ void dequantize_int8_to_f32_kernel(const int8_t* __restrict__ src, float* __restrict__ dst, size_t n, float scale, float zero_point) {
                    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= n) {
                        return;
                    }

                    int8_t val = src[idx];
                    float dequantized;

                    if (SYMMETRIC) {
                        dequantized = static_cast<float>(val) * scale;
                    } else {
                        // This must invert the quantize kernel, which computes
                        // q = round((v - zero_point) / scale). The old form was
                        // (q + zero_point) * scale, which is the inverse of q = v/scale + zero_point
                        // -- a different scheme. Composing the two did not return the input:
                        //   v' = ((v - zp)/scale + zp) * scale = v + zp*scale - zp
                        // so every asymmetric round trip was offset by zp*(scale-1). Measured on
                        // data spanning [0, 1] with the calibrator's parameters (scale 0.003937,
                        // zp -127): every sample quantized to code 127 and dequantized to 0.0,
                        // a relative error of 1.0.
                        dequantized = static_cast<float>(val) * scale + zero_point;
                    }

                    dst[idx] = dequantized;
                }

                // A `quantize_f32_to_int8_vectorized_kernel` used to live here. It was never
                // launched from anywhere, and it could not have been: it took `int4* dst` and
                // wrote `make_int4(qx, qy, qz, qw)` -- 16 bytes -- for the 4 int8 codes that one
                // float4 of input produces, which occupy 4 bytes. Measured for n = 1024: the
                // destination needs 1024 bytes and that write pattern spans 4096, so wiring it up
                // would have run 3072 bytes past the end of the buffer and stored int32 codes
                // where packed int8 was expected. Removed rather than fixed because there is no
                // caller to serve; a vectorised path should be added together with a test that
                // compares it against the scalar kernel.
                __global__ void build_histogram_kernel(const float* __restrict__ data, uint32_t* __restrict__ histogram, size_t n, float min_val, float max_val, int num_bins) {
                    // Dynamic shared memory sized to num_bins at launch: the old
                    // fixed __shared__ uint32_t[256] wrote smem[bin] for bin up to num_bins-1
                    // (the HistogramCalibrator default num_bins is 2048) — an out-of-bounds
                    // shared-memory write, and bins >= 256 were never flushed to the host
                    // histogram, so calibration produced a wrong threshold.
                    extern __shared__ uint32_t smem[];

                    const int tid = static_cast<int>(threadIdx.x);
                    const int step = static_cast<int>(blockDim.x);
                    for (int i = tid; i < num_bins; i += step) {
                        smem[i] = 0;
                    }
                    __syncthreads();

                    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < n) {
                        float val = data[idx];
                        float range = max_val - min_val;
                        if (range > 1e-6f) {
                            int bin = static_cast<int>((val - min_val) / range * (num_bins - 1));
                            bin = max(0, min(num_bins - 1, bin));
                            atomicAdd(&smem[bin], 1u);
                        }
                    }

                    __syncthreads();

                    for (int i = tid; i < num_bins; i += step) {
                        atomicAdd(&histogram[i], smem[i]);
                    }
                }

                __global__ void compute_minmax_kernel(const float* __restrict__ data, size_t n, float* __restrict__ block_mins, float* __restrict__ block_maxs) {
                    __shared__ float smin[256];
                    __shared__ float smax[256];

                    size_t tid = threadIdx.x;
                    size_t idx = blockIdx.x * blockDim.x + tid;
                    size_t block_id = blockIdx.x;

                    float local_min = FLT_MAX;
                    float local_max = -FLT_MAX;

                    for (size_t i = idx; i < n; i += gridDim.x * blockDim.x) {
                        float val = data[i];
                        local_min = fminf(local_min, val);
                        local_max = fmaxf(local_max, val);
                    }

                    smin[tid] = local_min;
                    smax[tid] = local_max;
                    __syncthreads();

                    for (size_t s = blockDim.x / 2; s > 0; s /= 2) {
                        if (tid < s && idx + s < n) {
                            smin[tid] = fminf(smin[tid], smin[tid + s]);
                            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
                        }
                        __syncthreads();
                    }

                    if (tid == 0) {
                        block_mins[block_id] = smin[0];
                        block_maxs[block_id] = smax[0];
                    }
                }

                __global__ void quantize_with_scale_from_histogram_kernel(
                    const float* __restrict__ src, int8_t* __restrict__ dst, size_t n, float min_val, float max_val, float percentile) {
                    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= n) {
                        return;
                    }

                    float range = max_val - min_val;
                    if (range < 1e-6f) {
                        dst[idx] = 0;
                        return;
                    }

                    float scale = range / 254.0f;
                    float val = src[idx];
                    int32_t quantized = static_cast<int32_t>(__float2int_rn((val - min_val) / scale - 127.0f));
                    quantized = max(-127, min(127, quantized));

                    dst[idx] = static_cast<int8_t>(quantized);
                }

            }  // namespace detail

            void quantize_f32_to_int8(const float* src, int8_t* dst, size_t n, QuantizationParams params, cudaStream_t stream) {
                // Nothing to convert. Both siblings compute_minmax and build_histogram already
                // return early for an empty input; without it the zero-sized grid produced
                // "invalid configuration argument" — measured for both quantize and dequantize.
                if (n == 0) {
                    return;
                }

                constexpr size_t block_size = 256;
                size_t grid_size = (n + block_size - 1) / block_size;

                if (params.symmetric) {
                    detail::quantize_f32_to_int8_kernel<true><<<grid_size, block_size, 0, stream>>>(src, dst, n, params.scale, params.zero_point);
                } else {
                    detail::quantize_f32_to_int8_kernel<false><<<grid_size, block_size, 0, stream>>>(src, dst, n, params.scale, params.zero_point);
                }

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    fprintf(stderr, "CUDA error in quantize_f32_to_int8: %s\n", cudaGetErrorString(err));
                }
            }

            void dequantize_int8_to_f32(const int8_t* src, float* dst, size_t n, QuantizationParams params, cudaStream_t stream) {
                // Nothing to convert. Both siblings compute_minmax and build_histogram already
                // return early for an empty input; without it the zero-sized grid produced
                // "invalid configuration argument" — measured for both quantize and dequantize.
                if (n == 0) {
                    return;
                }

                constexpr size_t block_size = 256;
                size_t grid_size = (n + block_size - 1) / block_size;

                if (params.symmetric) {
                    detail::dequantize_int8_to_f32_kernel<true><<<grid_size, block_size, 0, stream>>>(src, dst, n, params.scale, params.zero_point);
                } else {
                    detail::dequantize_int8_to_f32_kernel<false><<<grid_size, block_size, 0, stream>>>(src, dst, n, params.scale, params.zero_point);
                }

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    fprintf(stderr, "CUDA error in dequantize_int8_to_f32: %s\n", cudaGetErrorString(err));
                }
            }

            void quantize_f32_to_int8_async(const float* src, int8_t* dst, size_t n, QuantizationParams params, cudaStream_t stream) {
                quantize_f32_to_int8(src, dst, n, params, stream);
            }

            void dequantize_int8_to_f32_async(const int8_t* src, float* dst, size_t n, QuantizationParams params, cudaStream_t stream) {
                dequantize_int8_to_f32(src, dst, n, params, stream);
            }

            void quantize_f32_to_int8_with_calibration(const float* src, int8_t* dst, size_t n, float min_val, float max_val, cudaStream_t stream) {
                // Nothing to convert. Both siblings compute_minmax and build_histogram already
                // return early for an empty input; without it the zero-sized grid produced
                // "invalid configuration argument" — measured for both quantize and dequantize.
                if (n == 0) {
                    return;
                }

                constexpr size_t block_size = 256;
                size_t grid_size = (n + block_size - 1) / block_size;

                detail::quantize_with_scale_from_histogram_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, n, min_val, max_val, 99.99f);

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    fprintf(stderr, "CUDA error in quantize_f32_to_int8_with_calibration: %s\n", cudaGetErrorString(err));
                }
            }

            void build_histogram(const float* data, uint32_t* histogram, size_t n, float min_val, float max_val, int num_bins, cudaStream_t stream) {
                // An empty sample set has no histogram, and a zero-bin histogram has no output.
                // Both used to fall through to cudaMalloc(0) / cudaMemset(..., 0) and fail with a
                // bare "invalid argument" that named neither the parameter nor the cause — measured
                // for n == 0 and num_bins == 0.
                if (n == 0) {
                    return;
                }
                if (num_bins <= 0) {
                    throw ::std::invalid_argument("build_histogram: num_bins must be positive, got " + ::std::to_string(num_bins));
                }

                bool data_needs_copy = false;
                cudaPointerAttributes attr;
                if (cudaPointerGetAttributes(&attr, data) != cudaSuccess || attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) {
                    data_needs_copy = true;
                }

                const float* d_data = data;
                // Owned by unique_ptr so the throwing calls below release it. This was a raw pointer
                // freed near the end of the function: cudaMalloc succeeding and then cudaMemcpy or one of
                // the CUDA_CHECKs below failing left the allocation with no owner.
                ::cuda::memory::unique_ptr<float> d_data_owned(data_needs_copy ? n : 0);
                if (data_needs_copy) {
                    CUDA_CHECK(cudaMemcpy(d_data_owned.get(), data, n * sizeof(float), cudaMemcpyHostToDevice));
                    d_data = d_data_owned.get();
                }

                CUDA_CHECK(cudaMemset(histogram, 0, num_bins * sizeof(uint32_t)));

                constexpr size_t block_size = 256;
                size_t grid_size = (n + block_size - 1) / block_size;

                // Dynamic shared memory must cover every possible bin: the
                // calibrator's default num_bins=2048 previously overran a fixed 256-entry
                // shared buffer.
                //
                // The old note here claimed 2048 bins were "comfortably within the 48KB static
                // shared-memory limit", which understated the boundary: this is DYNAMIC shared
                // memory, whose per-block ceiling is 48KB unless raised with
                // cudaFuncSetAttribute, and it is allocated per block as num_bins * 4 bytes. At
                // num_bins = 16384 that is 64KB, the launch is rejected with "invalid argument",
                // and the failure was invisible: the error only reached stderr, the function
                // returned normally, and the destination histogram stayed all zero. Measured:
                // with n = 4096 samples, num_bins = 12288 (48KB) sums to 4096 as expected, while
                // num_bins = 16384 and 32768 both sum to 0. A calibrator reading that histogram
                // computes its threshold from nothing.
                const size_t shared_bytes = static_cast<size_t>(num_bins) * sizeof(uint32_t);
                constexpr size_t kMaxDynamicShared = 48u * 1024u;
                if (shared_bytes > kMaxDynamicShared) {
                    throw ::std::invalid_argument(
                        "build_histogram: num_bins=" + ::std::to_string(num_bins) + " needs " + ::std::to_string(shared_bytes)
                        + " bytes of dynamic shared memory, above the " + ::std::to_string(kMaxDynamicShared)
                        + "-byte per-block limit; reduce num_bins (a histogram with more bins than samples carries no extra "
                          "information)");
                }
                detail::build_histogram_kernel<<<grid_size, block_size, shared_bytes, stream>>>(d_data, histogram, n, min_val, max_val, num_bins);

                if (stream == 0) {
                    cudaDeviceSynchronize();
                } else {
                    cudaStreamSynchronize(stream);
                }

                // A launch can also fail for reasons this function cannot predict, so the error is
                // still reported -- but it is now raised rather than printed, because returning
                // successfully with a histogram that was never written is worse than failing.
                CUDA_CHECK(cudaGetLastError());
            }

            void compute_minmax(const float* data, size_t n, float* min_val, float* max_val, cudaStream_t stream) {
                constexpr size_t block_size = 256;
                size_t grid_size = std::min<size_t>((n + block_size - 1) / block_size, 256);

                if (grid_size == 0) {
                    *min_val = 0.0f;
                    *max_val = 0.0f;
                    return;
                }

                bool data_needs_copy = false;
                cudaPointerAttributes attr;
                if (cudaPointerGetAttributes(&attr, data) != cudaSuccess || attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) {
                    data_needs_copy = true;
                }

                const float* d_data = data;
                // All three buffers are owned by unique_ptr: the CUDA_CHECKs below throw, and the frees sat
                // past every one of them, so a failure left whichever buffers had been allocated with no
                // owner.
                ::cuda::memory::unique_ptr<float> d_data_owned(data_needs_copy ? n : 0);
                if (data_needs_copy) {
                    CUDA_CHECK(cudaMemcpy(d_data_owned.get(), data, n * sizeof(float), cudaMemcpyHostToDevice));
                    d_data = d_data_owned.get();
                }

                ::cuda::memory::unique_ptr<float> d_block_mins_owned(grid_size);
                ::cuda::memory::unique_ptr<float> d_block_maxs_owned(grid_size);
                float* d_block_mins = d_block_mins_owned.get();
                float* d_block_maxs = d_block_maxs_owned.get();

                detail::compute_minmax_kernel<<<grid_size, block_size, 0, stream>>>(d_data, n, d_block_mins, d_block_maxs);

                if (stream == 0) {
                    cudaDeviceSynchronize();
                }

                std::vector<float> h_block_mins(grid_size);
                std::vector<float> h_block_maxs(grid_size);
                CUDA_CHECK(cudaMemcpy(h_block_mins.data(), d_block_mins, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_block_maxs.data(), d_block_maxs, grid_size * sizeof(float), cudaMemcpyDeviceToHost));

                float local_min = FLT_MAX;
                float local_max = -FLT_MAX;
                for (size_t i = 0; i < grid_size; ++i) {
                    local_min = std::fmin(local_min, h_block_mins[i]);
                    local_max = std::fmax(local_max, h_block_maxs[i]);
                }

                // min_val/max_val may be either device pointers (write via H2D copy) or
                // ordinary host pointers (write directly). cudaPointerGetAttributes
                // succeeds on both: cudaMemoryTypeDevice for device memory, any other type
                // (host, managed, unregistered) for host-accessible memory.
                const bool min_is_device = (cudaPointerGetAttributes(&attr, min_val) == cudaSuccess) && (attr.type == cudaMemoryTypeDevice);

                if (!min_is_device) {
                    *min_val = local_min;
                    *max_val = local_max;
                } else {
                    CUDA_CHECK(cudaMemcpy(min_val, &local_min, sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(max_val, &local_max, sizeof(float), cudaMemcpyHostToDevice));
                }

                // The buffers are released by their unique_ptr destructors. A synchronisation is still
                // wanted before returning: freeing while the kernel is in flight would leave the driver
                // reading memory that has been returned to the pool.
                if (d_data_owned.get() != nullptr) {
                    CUDA_CHECK(cudaStreamSynchronize(stream == 0 ? 0 : stream));
                }
            }

        }  // namespace cuda
    }  // namespace quantize
}  // namespace nova
