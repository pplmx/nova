#pragma once

/**
 * @file fft_types.h
 * @brief FFT type definitions and enums
 */

#include <cufft.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace cuda::fft {

enum class Direction {
    Forward = CUFFT_FORWARD,
    Inverse = CUFFT_INVERSE
};

enum class Result {
    Success = 0,
    InvalidPlan = CUFFT_INVALID_PLAN,
    AllocFailed = CUFFT_ALLOC_FAILED,
    InvalidType = CUFFT_INVALID_TYPE,
    InternalError = CUFFT_INTERNAL_ERROR,
    NoWorkSpace = CUFFT_NO_WORKSPACE,
    NotSupported = CUFFT_UNALIGNED_DATA
};

enum class TransformType {
    RealToComplex = CUFFT_R2C,
    DoubleRealToComplex = CUFFT_D2Z,
    ComplexToReal = CUFFT_C2R,
    ComplexToComplex = CUFFT_C2C
};

template <typename T>
struct Complex;

template <>
struct Complex<float> {
    using type = cuComplex;
};

template <>
struct Complex<double> {
    using type = cuDoubleComplex;
};

template <typename T>
using ComplexT = typename Complex<T>::type;

struct FFTConfig {
    size_t size = 0;
    Direction direction = Direction::Forward;
    int batch = 1;
    int stream_id = 0;
    TransformType type = TransformType::RealToComplex;
};

inline constexpr bool is_power_of_two(size_t n) {
    return (n & (n - 1)) == 0;
}

}  // namespace cuda::fft
