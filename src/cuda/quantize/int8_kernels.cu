#include <cuda/quantize/int8_kernels.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

namespace nova {
namespace quantize {
namespace cuda {

namespace detail {

template<bool SYMMETRIC>
__global__ void quantize_f32_to_int8_kernel(
    const float* __restrict__ src,
    int8_t* __restrict__ dst,
    size_t n,
    float scale,
    float zero_point) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

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

template<bool SYMMETRIC>
__global__ void dequantize_int8_to_f32_kernel(
    const int8_t* __restrict__ src,
    float* __restrict__ dst,
    size_t n,
    float scale,
    float zero_point) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int8_t val = src[idx];
    float dequantized;

    if (SYMMETRIC) {
        dequantized = static_cast<float>(val) * scale;
    } else {
        dequantized = (static_cast<float>(val) + zero_point) * scale;
    }

    dst[idx] = dequantized;
}

template<bool SYMMETRIC>
__global__ void quantize_f32_to_int8_vectorized_kernel(
    const float4* __restrict__ src,
    int4* __restrict__ dst,
    size_t n4,
    float scale,
    float zero_point) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n4) return;

    float4 vals = src[idx];

    int32_t qx, qy, qz, qw;

    if (SYMMETRIC) {
        qx = static_cast<int32_t>(__float2int_rn(vals.x / scale));
        qy = static_cast<int32_t>(__float2int_rn(vals.y / scale));
        qz = static_cast<int32_t>(__float2int_rn(vals.z / scale));
        qw = static_cast<int32_t>(__float2int_rn(vals.w / scale));
    } else {
        qx = static_cast<int32_t>(__float2int_rn((vals.x - zero_point) / scale));
        qy = static_cast<int32_t>(__float2int_rn((vals.y - zero_point) / scale));
        qz = static_cast<int32_t>(__float2int_rn((vals.z - zero_point) / scale));
        qw = static_cast<int32_t>(__float2int_rn((vals.w - zero_point) / scale));
    }

    qx = max(-127, min(127, qx));
    qy = max(-127, min(127, qy));
    qz = max(-127, min(127, qz));
    qw = max(-127, min(127, qw));

    dst[idx] = make_int4(qx, qy, qz, qw);
}

__global__ void build_histogram_kernel(
    const float* __restrict__ data,
    uint32_t* __restrict__ histogram,
    size_t n,
    float min_val,
    float max_val,
    int num_bins) {

    __shared__ uint32_t smem[256];

    size_t tid = threadIdx.x;
    if (tid < 256) {
        smem[tid] = 0;
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) {
        float val = data[idx];
        float range = max_val - min_val;
        if (range > 1e-6f) {
            int bin = static_cast<int>((val - min_val) / range * (num_bins - 1));
            bin = max(0, min(num_bins - 1, bin));
            atomicAdd(&smem[bin], 1);
        }
    }

    __syncthreads();

    if (tid < 256) {
        atomicAdd(&histogram[tid], smem[tid]);
    }
}

__global__ void compute_minmax_kernel(
    const float* __restrict__ data,
    size_t n,
    float* __restrict__ min_out,
    float* __restrict__ max_out) {

    __shared__ float smin[256];
    __shared__ float smax[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

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
        unsigned int* min_as_int = reinterpret_cast<unsigned int*>(min_out);
        unsigned int* max_as_int = reinterpret_cast<unsigned int*>(max_out);
        unsigned int old_min = atomicMin(min_as_int, __float_as_uint(smin[0]));
        unsigned int old_max = atomicMax(max_as_int, __float_as_uint(smax[0]));
        (void)old_min;
        (void)old_max;
    }
}

__global__ void quantize_with_scale_from_histogram_kernel(
    const float* __restrict__ src,
    int8_t* __restrict__ dst,
    size_t n,
    float min_val,
    float max_val,
    float percentile) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

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

} // namespace detail

void quantize_f32_to_int8(
    const float* src, int8_t* dst, size_t n,
    QuantizationParams params,
    cudaStream_t stream) {

    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    if (params.symmetric) {
        detail::quantize_f32_to_int8_kernel<true><<<grid_size, block_size, 0, stream>>>(
            src, dst, n, params.scale, params.zero_point);
    } else {
        detail::quantize_f32_to_int8_kernel<false><<<grid_size, block_size, 0, stream>>>(
            src, dst, n, params.scale, params.zero_point);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in quantize_f32_to_int8: %s\n", cudaGetErrorString(err));
    }
}

void dequantize_int8_to_f32(
    const int8_t* src, float* dst, size_t n,
    QuantizationParams params,
    cudaStream_t stream) {

    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    if (params.symmetric) {
        detail::dequantize_int8_to_f32_kernel<true><<<grid_size, block_size, 0, stream>>>(
            src, dst, n, params.scale, params.zero_point);
    } else {
        detail::dequantize_int8_to_f32_kernel<false><<<grid_size, block_size, 0, stream>>>(
            src, dst, n, params.scale, params.zero_point);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in dequantize_int8_to_f32: %s\n", cudaGetErrorString(err));
    }
}

void quantize_f32_to_int8_async(
    const float* src, int8_t* dst, size_t n,
    QuantizationParams params,
    cudaStream_t stream) {

    quantize_f32_to_int8(src, dst, n, params, stream);
}

void dequantize_int8_to_f32_async(
    const int8_t* src, float* dst, size_t n,
    QuantizationParams params,
    cudaStream_t stream) {

    dequantize_int8_to_f32(src, dst, n, params, stream);
}

void quantize_f32_to_int8_with_calibration(
    const float* src, int8_t* dst, size_t n,
    float min_val, float max_val,
    cudaStream_t stream) {

    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    detail::quantize_with_scale_from_histogram_kernel<<<grid_size, block_size, 0, stream>>>(
        src, dst, n, min_val, max_val, 99.99f);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in quantize_f32_to_int8_with_calibration: %s\n", cudaGetErrorString(err));
    }
}

void build_histogram(
    const float* data, uint32_t* histogram,
    size_t n, float min_val, float max_val,
    int num_bins,
    cudaStream_t stream) {

    cudaMemset(histogram, 0, num_bins * sizeof(uint32_t));

    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    detail::build_histogram_kernel<<<grid_size, block_size, 0, stream>>>(
        data, histogram, n, min_val, max_val, num_bins);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in build_histogram: %s\n", cudaGetErrorString(err));
    }
}

void compute_minmax(
    const float* data, size_t n,
    float* min_val, float* max_val,
    cudaStream_t stream) {

    cudaMemset(min_val, 0x7F, sizeof(float));
    cudaMemset(max_val, 0x80, sizeof(float));

    constexpr size_t block_size = 256;
    size_t grid_size = 256;

    detail::compute_minmax_kernel<<<grid_size, block_size, 0, stream>>>(
        data, n, min_val, max_val);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_minmax: %s\n", cudaGetErrorString(err));
    }
}

} // namespace cuda
} // namespace quantize
} // namespace nova
