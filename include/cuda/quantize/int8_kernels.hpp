#ifndef NOVA_CUDA_QUANTIZE_INT8_KERNELS_HPP
#define NOVA_CUDA_QUANTIZE_INT8_KERNELS_HPP

#include <cuda_runtime.h>
#include <cstdint>

namespace nova {
namespace quantize {
namespace cuda {

struct QuantizationParams {
    float scale{1.0f};
    float zero_point{0.0f};
    bool symmetric{true};

    QuantizationParams() = default;

    QuantizationParams(float s, float zp = 0.0f, bool sym = true)
        : scale(s), zero_point(zp), symmetric(sym) {}
};

void quantize_f32_to_int8(
    const float* src, int8_t* dst, size_t n,
    QuantizationParams params,
    cudaStream_t stream = 0
);

void dequantize_int8_to_f32(
    const int8_t* src, float* dst, size_t n,
    QuantizationParams params,
    cudaStream_t stream = 0
);

void quantize_f32_to_int8_async(
    const float* src, int8_t* dst, size_t n,
    QuantizationParams params,
    cudaStream_t stream = 0
);

void dequantize_int8_to_f32_async(
    const int8_t* src, float* dst, size_t n,
    QuantizationParams params,
    cudaStream_t stream = 0
);

void quantize_f32_to_int8_with_calibration(
    const float* src, int8_t* dst, size_t n,
    float min_val, float max_val,
    cudaStream_t stream = 0
);

void build_histogram(
    const float* data, uint32_t* histogram,
    size_t n, float min_val, float max_val,
    int num_bins = 256,
    cudaStream_t stream = 0
);

void compute_minmax(
    const float* data, size_t n,
    float* min_val, float* max_val,
    cudaStream_t stream = 0
);

} // namespace cuda
} // namespace quantize
} // namespace nova

#endif // NOVA_CUDA_QUANTIZE_INT8_KERNELS_HPP
