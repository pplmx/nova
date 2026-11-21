#pragma once

/**
 * @file signal_processing.h
 * @brief GPU signal processing: convolution, wavelets, FIR filters
 * @author Nova CUDA Library
 * @version 2.3
 *
 * @par Pointer convention
 * Every input pointer is a **device** pointer and every result is device-resident. This was
 * true of three of the four entry points and was stated nowhere; fir_filter additionally
 * copied its arguments with cudaMemcpyHostToDevice, which under unified virtual addressing
 * happens to resolve a device address anyway, so the inconsistency produced correct answers
 * and went unnoticed. All four copy DeviceToDevice now, and the convention is stated here.
 */

#include <cstddef>

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

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

    /**
     * @brief Full linear convolution via cuFFT
     *
     * Output length is signal_len + kernel_len - 1. Both zero lengths give an empty result.
     *
     * @param signal Device pointer, signal_len elements
     * @param signal_len Signal length
     * @param kernel Device pointer, kernel_len elements
     * @param kernel_len Kernel length
     * @param mode Only BoundaryMode::ZeroPadding is implemented; anything else throws
     *             std::invalid_argument rather than silently returning a padded result
     * @return The convolution, device-resident, of length signal_len + kernel_len - 1
     */
    ConvolutionResult fft_convolution(const float* signal, size_t signal_len, const float* kernel, size_t kernel_len, BoundaryMode mode = BoundaryMode::ZeroPadding);

    /**
     * @brief Multi-level Haar wavelet decomposition
     *
     * @warning @p n must be a **power of two**. The in-place layout halves the region at each
     *          level and writes that level's detail coefficients immediately after the
     *          approximation, which only lands correctly when the halving is exact. For any
     *          other length the transform is silently lossy: an odd region leaves its last
     *          sample untransformed, and the detail write then overwrites it. Measured
     *          round-trip error before this was rejected: n=8 gives 9.5e-07, while n=5, 6, 7
     *          and 12 give 5.0, 5.5, 7.0 and 10.5. A non-power-of-two length now throws
     *          std::invalid_argument; pad to the next power of two and discard the padding
     *          after the inverse.
     *
     * @param signal Device pointer, n elements
     * @param n Signal length; a power of two
     * @return Coefficients in the standard in-place layout, and the number of levels
     * @throws std::invalid_argument if n is not a power of two
     */
    WaveletResult haar_wavelet_forward(const float* signal, size_t n);

    /**
     * @brief Reconstruct a signal from its Haar coefficients
     *
     * @param coeffs The result of haar_wavelet_forward()
     * @param original_size The length that was given to haar_wavelet_forward(); must be a
     *        power of two and must not exceed the coefficient buffer's length, or
     *        std::invalid_argument is thrown
     * @return The reconstructed signal, device-resident
     */
    memory::Buffer<float> haar_wavelet_inverse(const WaveletResult& coeffs, size_t original_size);

    /**
     * @brief Causal FIR filter with the given taps
     *
     * output[i] = sum over j of signal[i - j] * coefficients[j], for i - j >= 0. Output length
     * equals signal_len. An empty coefficient set yields all zeros, and a zero-length signal
     * yields an empty result.
     *
     * @param signal Device pointer, signal_len elements
     * @param signal_len Signal length
     * @param coefficients Device pointer, num_coeffs taps
     * @param num_coeffs Number of taps; may be zero
     */
    FIRFilterResult fir_filter(const float* signal, size_t signal_len, const float* coefficients, size_t num_coeffs);

}  // namespace cuda::signal
