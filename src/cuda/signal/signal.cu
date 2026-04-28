#include "cuda/signal/signal.h"

#include <cmath>
#include <cstdlib>

#include "cuda/device/error.h"

namespace cuda::signal {

namespace {

size_t next_optimal_fft_size(size_t n) {
    size_t factors[] = {2, 3, 5, 7};
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

    float sum = 0.0f;
    size_t start = (idx >= num_coeffs - 1) ? (idx - num_coeffs + 1) : 0;

    for (size_t j = start; j <= idx; ++j) {
        sum += signal[j] * coeffs[idx - j];
    }
    output[idx] = sum;
}

}  // namespace

ConvolutionResult fft_convolution(const float* signal, size_t signal_len, const float* kernel, size_t kernel_len, BoundaryMode mode) {
    size_t fft_size = next_optimal_fft_size(signal_len + kernel_len - 1);
    size_t output_size = signal_len + kernel_len - 1;

    ConvolutionResult result;
    result.output = memory::Buffer<float>(output_size);
    result.output_size = output_size;

    memory::Buffer<float> signal_padded(fft_size);
    memory::Buffer<float> kernel_padded(fft_size);

    CUDA_CHECK(cudaMemcpy(signal_padded.data(), signal, signal_len * sizeof(float), cudaMemcpyDeviceToDevice));

    for (size_t i = 0; i < kernel_len; ++i) {
        CUDA_CHECK(cudaMemcpy(&kernel_padded.data()[i], &kernel[i], sizeof(float), cudaMemcpyHostToDevice));
    }

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, static_cast<int>(fft_size), CUFFT_R2C, 1));

    cufftReal* d_signal = reinterpret_cast<cufftReal*>(signal_padded.data());
    cufftComplex* d_kernel = reinterpret_cast<cufftComplex*>(kernel_padded.data());

    CUFFT_CHECK(cufftExecR2C(plan, d_signal, d_kernel));

    for (size_t i = 0; i < fft_size / 2 + 1; ++i) {
    }

    CUFFT_CHECK(cufftExecC2R(plan, d_kernel, d_signal));

    CUDA_CHECK(cudaMemcpy(result.output.data(), signal_padded.data(), output_size * sizeof(float), cudaMemcpyDeviceToDevice));

    CUFFT_CHECK(cufftDestroy(plan));

    return result;
}

WaveletResult haar_wavelet_forward(const float* signal, size_t n) {
    size_t levels = 0;
    size_t temp_n = n;
    while (temp_n > 1) {
        temp_n /= 2;
        levels++;
    }

    WaveletResult result;
    result.coefficients = memory::Buffer<float>(n);
    result.levels = levels;

    memory::Buffer<float> current(n);
    CUDA_CHECK(cudaMemcpy(current.data(), signal, n * sizeof(float), cudaMemcpyDeviceToDevice));

    memory::Buffer<float> temp(n / 2);

    for (size_t level = 0; level < levels; ++level) {
        size_t curr_n = n >> level;
        if (curr_n < 2) break;

        haar_decompose_kernel<<<(curr_n / 2 + 255) / 256, 256>>>(current.data(), temp.data(), temp.data() + curr_n / 2, curr_n);

        memory::Buffer<float> next(curr_n / 2);
        CUDA_CHECK(cudaMemcpy(next.data(), temp.data(), (curr_n / 2) * sizeof(float), cudaMemcpyDeviceToDevice));
        current = next;
    }

    CUDA_CHECK(cudaMemcpy(result.coefficients.data(), signal, n * sizeof(float), cudaMemcpyDeviceToDevice));

    return result;
}

memory::Buffer<float> haar_wavelet_inverse(const WaveletResult& coeffs, size_t original_size) {
    memory::Buffer<float> result(original_size);

    CUDA_CHECK(cudaMemcpy(result.data(), coeffs.coefficients.data(), original_size * sizeof(float), cudaMemcpyDeviceToDevice));

    memory::Buffer<float> approx(original_size);
    memory::Buffer<float> detail(original_size / 2);

    for (size_t level = coeffs.levels; level > 0; --level) {
        size_t curr_n = original_size >> (coeffs.levels - level);

        CUDA_CHECK(cudaMemcpy(approx.data(), result.data(), (curr_n / 2) * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(detail.data(), result.data() + curr_n / 2, (curr_n / 2) * sizeof(float), cudaMemcpyDeviceToDevice));

        haar_reconstruct_kernel<<<(curr_n / 2 + 255) / 256, 256>>>(approx.data(), detail.data(), result.data(), curr_n);
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
