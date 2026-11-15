#include "cuda/signal/signal_processing.h"

#include <cufft.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <cstdlib>

#include "cuda/device/error.h"

namespace cuda::signal {

namespace {

size_t next_optimal_fft_size(size_t n) {
    size_t size = 1;
    while (size < n) {
        size *= 2;
    }
    return size;
}

__global__ void haar_decompose_kernel(const float* input, float* approx, float* detail, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2) return;

    approx[idx] = (input[2 * idx] + input[2 * idx + 1]) * 0.70710678118f;
    detail[idx] = (input[2 * idx] - input[2 * idx + 1]) * 0.70710678118f;
}

__global__ void haar_reconstruct_kernel(const float* approx, const float* detail, float* output, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2) return;

    output[2 * idx] = (approx[idx] + detail[idx]) * 0.70710678118f;
    output[2 * idx + 1] = (approx[idx] - detail[idx]) * 0.70710678118f;
}

__global__ void fir_kernel(const float* signal, const float* coeffs, float* output, size_t signal_len, size_t num_coeffs) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= signal_len) return;

    // num_coeffs == 0 makes `num_coeffs - 1` wrap to SIZE_MAX, so the bounds test
    // below always took the start=0 branch and the loop then read coeffs[] past
    // its end. An empty filter has no taps, so every output is zero.
    if (num_coeffs == 0) {
        output[idx] = 0.0f;
        return;
    }

    float sum = 0.0f;
    const size_t span = num_coeffs - 1;
    const size_t start = (idx >= span) ? (idx - span) : 0;

    for (size_t j = start; j <= idx; ++j) {
        sum += signal[j] * coeffs[idx - j];
    }
    output[idx] = sum;
}

// Pointwise product of two spectra. A complex multiply needs all four terms;
// multiplying the real and imaginary parts separately is not the same operation
// and yields the wrong convolution.
__global__ void complex_multiply_kernel(cufftComplex* a, const cufftComplex* b, int bins) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bins) return;
    const float ar = a[i].x;
    const float ai = a[i].y;
    a[i].x = ar * b[i].x - ai * b[i].y;
    a[i].y = ar * b[i].y + ai * b[i].x;
}

// cuFFT's C2R is unnormalised: the result is scaled by the transform length.
__global__ void scale_kernel(float* data, float factor, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= factor;
    }
}

void check_cufft(cufftResult status, const char* file, int line) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " - cuFFT error: " + std::to_string(static_cast<int>(status)));
    }
}

#define CUFFT_CHECK(call) check_cufft(call, __FILE__, __LINE__)

}  // namespace

ConvolutionResult fft_convolution(const float* signal, size_t signal_len, const float* kernel, size_t kernel_len, BoundaryMode mode) {
    // The linear convolution is signal_len + kernel_len - 1 long, which underflows
    // to SIZE_MAX when both inputs are empty: the code then tried to allocate
    // 16 EiB. An empty input has an empty convolution.
    if (signal_len == 0 || kernel_len == 0) {
        ConvolutionResult empty;
        empty.output = memory::Buffer<float>(0);
        empty.output_size = 0;
        return empty;
    }

    // `mode` is not implemented — the result is the same full linear
    // convolution regardless. Rejecting it is better than silently ignoring it
    // and returning a buffer the caller believes honours the boundary rule.
    if (mode != BoundaryMode::ZeroPadding) {
        throw std::invalid_argument(
            "fft_convolution: only BoundaryMode::ZeroPadding is implemented");
    }

    const size_t output_size = signal_len + kernel_len - 1;
    const size_t fft_size = next_optimal_fft_size(output_size);

    ConvolutionResult result;
    result.output = memory::Buffer<float>(output_size);
    result.output_size = output_size;

    // Both inputs are device pointers, consistent with the rest of this module
    // (haar_wavelet_* and fir_filter also read device data). The previous
    // version copied the kernel one element at a time with cudaMemcpyHostToDevice,
    // so a device-resident kernel was read out of host memory.
    memory::Buffer<float> signal_padded(fft_size);
    memory::Buffer<float> kernel_padded(fft_size);
    CUDA_CHECK(cudaMemset(signal_padded.data(), 0, fft_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(kernel_padded.data(), 0, fft_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(signal_padded.data(), signal, signal_len * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(kernel_padded.data(), kernel, kernel_len * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // Linear convolution via the FFT needs the full complex pipeline: transform
    // both inputs, multiply the spectra POINTWISE, then invert. The previous
    // version ran a single R2C plan and then called cufftExecC2R on that same
    // plan with frequency data, and never multiplied anything — the kernel's
    // transform simply overwrote the signal's. Executing it raised
    // CUFFT_INVALID_VALUE.
    const int bins = static_cast<int>(fft_size / 2 + 1);
    const int block = 256;
    const int grid = (bins + block - 1) / block;

    memory::Buffer<cufftComplex> signal_spectrum(bins);
    memory::Buffer<cufftComplex> kernel_spectrum(bins);
    memory::Buffer<cufftComplex> product(bins);

    cufftHandle forward_signal;
    cufftHandle forward_kernel;
    cufftHandle inverse;
    CUFFT_CHECK(cufftPlan1d(&forward_signal, static_cast<int>(fft_size), CUFFT_R2C, 1));
    CUFFT_CHECK(cufftPlan1d(&forward_kernel, static_cast<int>(fft_size), CUFFT_R2C, 1));
    CUFFT_CHECK(cufftPlan1d(&inverse, static_cast<int>(fft_size), CUFFT_C2R, 1));

    CUFFT_CHECK(cufftExecR2C(forward_signal,
                             reinterpret_cast<cufftReal*>(signal_padded.data()),
                             signal_spectrum.data()));
    CUFFT_CHECK(cufftExecR2C(forward_kernel,
                             reinterpret_cast<cufftReal*>(kernel_padded.data()),
                             kernel_spectrum.data()));

    CUDA_CHECK(cudaMemcpy(product.data(), signal_spectrum.data(),
                          bins * sizeof(cufftComplex), cudaMemcpyDeviceToDevice));
    complex_multiply_kernel<<<grid, block>>>(product.data(), kernel_spectrum.data(), bins);
    CUDA_CHECK(cudaGetLastError());

    // C2R overwrites its input, so it gets its own copy of the product.
    CUFFT_CHECK(cufftExecC2R(inverse, product.data(),
                             reinterpret_cast<cufftReal*>(signal_padded.data())));

    scale_kernel<<<(static_cast<int>(fft_size) + block - 1) / block, block>>>(
        signal_padded.data(), 1.0f / static_cast<float>(fft_size),
        static_cast<int>(fft_size));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(result.output.data(), signal_padded.data(),
                          output_size * sizeof(float), cudaMemcpyDeviceToDevice));

    CUFFT_CHECK(cufftDestroy(forward_signal));
    CUFFT_CHECK(cufftDestroy(forward_kernel));
    CUFFT_CHECK(cufftDestroy(inverse));

    return result;
}

WaveletResult haar_wavelet_forward(const float* signal, size_t n) {
    if (n == 0) {
        WaveletResult empty;
        empty.coefficients = memory::Buffer<float>(0);
        empty.levels = 0;
        return empty;
    }

    size_t levels = 0;
    size_t temp_n = n;
    while (temp_n > 1) {
        temp_n /= 2;
        levels++;
    }

    WaveletResult result;
    result.coefficients = memory::Buffer<float>(n);
    result.levels = levels;

    CUDA_CHECK(cudaMemcpy(result.coefficients.data(), signal, n * sizeof(float), cudaMemcpyDeviceToDevice));

    memory::Buffer<float> temp_approx(n);
    memory::Buffer<float> temp_detail(n);

    // Standard in-place Haar layout: after level L the approximation of length
    // curr_n/2 lives at [0, curr_n/2) and that level's detail coefficients at
    // [curr_n/2, curr_n). The previous version copied only the approximation back
    // and dropped temp_detail entirely, so the transform was lossy and the
    // inverse had nothing to add back.
    for (size_t level = 0; level < levels; ++level) {
        const size_t curr_n = n >> level;
        if (curr_n < 2) break;

        haar_decompose_kernel<<<(curr_n / 2 + 255) / 256, 256>>>(
            result.coefficients.data(), temp_approx.data(), temp_detail.data(), curr_n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(result.coefficients.data(), temp_approx.data(),
                              (curr_n / 2) * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(result.coefficients.data() + curr_n / 2,
                              temp_detail.data(), (curr_n / 2) * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    return result;
}

memory::Buffer<float> haar_wavelet_inverse(const WaveletResult& coeffs, size_t original_size) {
    // The level count and the size have to describe the same transform. A
    // mismatched pair makes `original_size >> (levels - level)` reach zero, and
    // the reconstruct kernel is then launched with a zero block count.
    if (original_size == 0) {
        return memory::Buffer<float>(0);
    }
    if (coeffs.levels >= sizeof(size_t) * 8) {
        throw std::invalid_argument(
            "haar_wavelet_inverse: level count " + std::to_string(coeffs.levels) +
            " exceeds the representable depth for original_size " +
            std::to_string(original_size));
    }
    for (size_t level = coeffs.levels; level > 0; --level) {
        const size_t curr_n = original_size >> (coeffs.levels - level);
        if (curr_n < 2) {
            throw std::invalid_argument(
                "haar_wavelet_inverse: level count " + std::to_string(coeffs.levels) +
                " does not match original_size " + std::to_string(original_size));
        }
    }

    memory::Buffer<float> result(original_size);

    CUDA_CHECK(cudaMemcpy(result.data(), coeffs.coefficients.data(), original_size * sizeof(float), cudaMemcpyDeviceToDevice));

    memory::Buffer<float> approx(original_size);
    memory::Buffer<float> detail(original_size / 2);
    memory::Buffer<float> stage(original_size);

    // Reconstruction must start at the DEEPEST decomposition and grow outwards:
    // the smallest region (original_size >> (levels-1), down to 2) is rebuilt
    // first, and each pass doubles the region it reconstructs. Iterating the
    // other way round rebuilt the full-length region first, which overwrote the
    // detail coefficients the shallower levels still needed, and the result bore
    // no resemblance to the input.
    for (size_t step = 0; step < coeffs.levels; ++step) {
        const size_t curr_n = original_size >> (coeffs.levels - 1 - step);

        // Mirror of the forward layout: approximation in the first half, that
        // level's detail in the second.
        CUDA_CHECK(cudaMemcpy(approx.data(), result.data(), (curr_n / 2) * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(detail.data(), result.data() + curr_n / 2, (curr_n / 2) * sizeof(float), cudaMemcpyDeviceToDevice));

        // Write into a separate buffer: the kernel reads approx/detail, and
        // writing straight back into `result` would alias buffers that later
        // levels still need to read.
        haar_reconstruct_kernel<<<(curr_n / 2 + 255) / 256, 256>>>(
            approx.data(), detail.data(), stage.data(), curr_n);
        CUDA_CHECK(cudaGetLastError());

        // The reconstruction grows the region that holds reconstructed data;
        // the untouched tail of `result` (deeper-level details) is preserved.
        CUDA_CHECK(cudaMemcpy(result.data(), stage.data(), curr_n * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    return result;
}

FIRFilterResult fir_filter(const float* signal, size_t signal_len, const float* coefficients, size_t num_coeffs) {
    FIRFilterResult result;
    result.output = memory::Buffer<float>(signal_len);
    result.output_size = signal_len;

    memory::Buffer<float> d_signal(signal_len);
    memory::Buffer<float> d_coeffs(num_coeffs);
    memory::Buffer<float> d_output(signal_len);

    CUDA_CHECK(cudaMemcpy(d_signal.data(), signal, signal_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coeffs.data(), coefficients, num_coeffs * sizeof(float), cudaMemcpyHostToDevice));

    fir_kernel<<<(signal_len + 255) / 256, 256>>>(d_signal.data(), d_coeffs.data(), d_output.data(), signal_len, num_coeffs);

    CUDA_CHECK(cudaMemcpy(result.output.data(), d_output.data(), signal_len * sizeof(float), cudaMemcpyDeviceToDevice));

    return result;
}

}  // namespace cuda::signal
