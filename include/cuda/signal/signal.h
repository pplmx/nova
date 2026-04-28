#pragma once

/**
 * @file signal.h
 * @brief GPU signal processing: convolution, wavelets, FIR filters
 * @author Nova CUDA Library
 * @version 2.3
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstddef>

#include "cuda/memory/buffer.h"

namespace cuda::signal {

enum class BoundaryMode { ZeroPadding, Reflect, Wrap };

struct ConvolutionResult {
    memory::Buffer<float> output;
    size_t output_size;
};

struct WaveletResult {
    memory::Buffer<float> coefficients;
    size_t levels;
};

struct FIRFilterResult {
    memory::Buffer<float> output;
    size_t output_size;
};

ConvolutionResult fft_convolution(const float* signal, size_t signal_len, const float* kernel, size_t kernel_len, BoundaryMode mode = BoundaryMode::ZeroPadding);
WaveletResult haar_wavelet_forward(const float* signal, size_t n);
memory::Buffer<float> haar_wavelet_inverse(const WaveletResult& coeffs, size_t original_size);
FIRFilterResult fir_filter(const float* signal, size_t signal_len, const float* coefficients, size_t num_coeffs);

}  // namespace cuda::signal
