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

/**
 * @brief cuFFT status codes
 *
 * Every member is a real cufftResult value. NotSupported previously mapped to
 * CUFFT_UNALIGNED_DATA (0x9), while CUFFT_NOT_SUPPORTED is 0x10, so a genuine
 * "not supported" answer was reported as "data is unaligned" — and the nine
 * other codes below were missing entirely, making them unrepresentable.
 */
enum class Result {
    Success = CUFFT_SUCCESS,
    InvalidPlan = CUFFT_INVALID_PLAN,
    AllocFailed = CUFFT_ALLOC_FAILED,
    InvalidType = CUFFT_INVALID_TYPE,
    InvalidValue = CUFFT_INVALID_VALUE,
    InternalError = CUFFT_INTERNAL_ERROR,
    ExecFailed = CUFFT_EXEC_FAILED,
    SetupFailed = CUFFT_SETUP_FAILED,
    InvalidSize = CUFFT_INVALID_SIZE,
    UnalignedData = CUFFT_UNALIGNED_DATA,
    IncompleteParameterList = CUFFT_INCOMPLETE_PARAMETER_LIST,
    InvalidDevice = CUFFT_INVALID_DEVICE,
    ParseError = CUFFT_PARSE_ERROR,
    NoWorkSpace = CUFFT_NO_WORKSPACE,
    NotImplemented = CUFFT_NOT_IMPLEMENTED,
    LicenseError = CUFFT_LICENSE_ERROR,
    NotSupported = CUFFT_NOT_SUPPORTED
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
    // n == 0 has to be excluded: 0 & (0-1) wraps to 0, so the bare bit trick
    // reports zero as a power of two, and callers that size a buffer or a plan
    // from it then ask for a zero-length transform.
    return n != 0 && (n & (n - 1)) == 0;
}

}  // namespace cuda::fft
