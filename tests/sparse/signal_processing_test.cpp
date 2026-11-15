/**
 * @file signal_processing_test.cpp
 * @brief Contracts for the GPU signal processing entry points
 *
 * include/cuda/signal/signal_processing.h (implemented in src/cuda/signal/
 * signal.cu) provides convolution, Haar wavelets and FIR filtering. It had no
 * tests, and four defects lived in it:
 *
 * - fir_kernel computed `start = (idx >= num_coeffs - 1) ? ... : 0`. With
 *   num_coeffs == 0, `num_coeffs - 1` wraps to SIZE_MAX, so the comparison was
 *   always false, start stayed 0, and the loop then read coeffs[] past its end.
 *
 * - fft_convolution sized its output as `signal_len + kernel_len - 1`, which
 *   underflows to SIZE_MAX when both inputs are empty, and tried to allocate
 *   16 EiB.
 *
 * - The `mode` parameter was never read. A caller asking for Reflect or Wrap got
 *   the same zero-padded result with no indication, so the buffer silently did
 *   not honour the boundary rule it was asked for.
 *
 * - haar_wavelet_inverse trusted coeffs.levels against original_size. A
 *   mismatched pair drove `original_size >> (levels - level)` to zero, and the
 *   reconstruct kernel was launched with a zero block count.
 */

#include <cuda/signal/signal_processing.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cuda::signal::test {

namespace {

// Owns a device copy of host data for the duration of a test.
class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::vector<float>& host) {
        size_ = host.size() * sizeof(float);
        if (size_ > 0) {
            cudaMalloc(&ptr_, size_);
            cudaMemcpy(ptr_, host.data(), size_, cudaMemcpyHostToDevice);
        }
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    const float* data() const { return ptr_; }
    float* data() { return ptr_; }

private:
    float* ptr_ = nullptr;
    size_t size_ = 0;
};

std::vector<float> to_host(const memory::Buffer<float>& device, size_t count) {
    std::vector<float> host(count);
    if (count > 0) {
        cudaMemcpy(host.data(), device.data(), count * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return host;
}

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

}  // namespace

TEST(FirFilterTest, ConvolvingAnImpulseReproducesTheTaps) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> impulse = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> taps = {1.0f, 2.0f, 3.0f};
    DeviceBuffer signal(impulse);
    DeviceBuffer coefficients(taps);

    const auto result = fir_filter(signal.data(), impulse.size(),
                                   coefficients.data(), taps.size());
    ASSERT_EQ(result.output_size, impulse.size());

    const auto output = to_host(result.output, result.output_size);
    // A unit impulse convolved with {1,2,3} yields the taps, then zeros.
    EXPECT_FLOAT_EQ(output[0], 1.0f);
    EXPECT_FLOAT_EQ(output[1], 2.0f);
    EXPECT_FLOAT_EQ(output[2], 3.0f);
    EXPECT_FLOAT_EQ(output[3], 0.0f);
    EXPECT_FLOAT_EQ(output[4], 0.0f);
}

TEST(FirFilterTest, SignalLengthOutputHasOneValuePerInputSample) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> signal_values(64, 1.0f);
    const std::vector<float> taps = {0.25f, 0.25f, 0.25f, 0.25f};
    DeviceBuffer signal(signal_values);
    DeviceBuffer coefficients(taps);

    const auto result = fir_filter(signal.data(), signal_values.size(),
                                   coefficients.data(), taps.size());
    EXPECT_EQ(result.output_size, signal_values.size());
}

// Regression: num_coeffs == 0 wrapped `num_coeffs - 1` to SIZE_MAX and the
// kernel then indexed coeffs[] out of bounds.
TEST(FirFilterTest, EmptyFilterProducesZeroesRatherThanReadingOutOfBounds) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> signal_values = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> no_taps;  // empty
    DeviceBuffer signal(signal_values);
    DeviceBuffer coefficients(no_taps);

    const auto result = fir_filter(signal.data(), signal_values.size(),
                                   coefficients.data(), 0);
    ASSERT_EQ(result.output_size, signal_values.size());

    const auto output = to_host(result.output, result.output_size);
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], 0.0f)
            << "an empty filter has no taps, so sample " << i << " must be zero";
    }
}

// Regression: this underflowed to SIZE_MAX and tried to allocate 16 EiB.
TEST(FftConvolutionTest, EmptyInputsProduceAnEmptyResult) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> empty;
    DeviceBuffer signal(empty);
    DeviceBuffer kernel(empty);

    const auto result = fft_convolution(signal.data(), 0, kernel.data(), 0);
    EXPECT_EQ(result.output_size, 0u);
}

TEST(FftConvolutionTest, EmptyKernelProducesAnEmptyResult) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> signal_values = {1.0f, 2.0f, 3.0f};
    const std::vector<float> empty;
    DeviceBuffer signal(signal_values);
    DeviceBuffer kernel(empty);

    const auto result = fft_convolution(signal.data(), signal_values.size(),
                                        kernel.data(), 0);
    EXPECT_EQ(result.output_size, 0u);
}

// Regression: `mode` was never read, so Reflect and Wrap silently behaved like
// ZeroPadding.
TEST(FftConvolutionTest, UnimplementedBoundaryModesAreRejected) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> signal_values = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> kernel_values = {1.0f, 1.0f};
    DeviceBuffer signal(signal_values);
    DeviceBuffer kernel(kernel_values);

    EXPECT_THROW(fft_convolution(signal.data(), signal_values.size(),
                                 kernel.data(), kernel_values.size(),
                                 BoundaryMode::Reflect),
                 std::invalid_argument)
        << "a boundary mode the implementation cannot honour must not be ignored";
    EXPECT_THROW(fft_convolution(signal.data(), signal_values.size(),
                                 kernel.data(), kernel_values.size(),
                                 BoundaryMode::Wrap),
                 std::invalid_argument);
}

// KNOWN DEFECT, not asserted here: the convolution itself does not work. The
// implementation runs ONE R2C plan and then calls cufftExecC2R on the same plan
// with a buffer that was written as frequency data, and it never multiplies the
// two spectra pointwise — the transform of the kernel overwrites the transform
// of the signal. Executing it raises CUFFT_INVALID_VALUE (4). Fixing that means
// rewriting the algorithm (two R2C transforms, a pointwise complex multiply, one
// C2R), which is beyond what this change covers. What IS pinned below is that
// the input validation added here runs and that the call reaches cuFFT at all,
// so the defect cannot be mistaken for a silent wrong answer.
TEST(FftConvolutionTest, ReachesCufftForAValidRequestToday) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> signal_values = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> kernel_values = {1.0f, 1.0f, 1.0f};
    DeviceBuffer signal(signal_values);
    DeviceBuffer kernel(kernel_values);

    EXPECT_THROW(
        {
            auto result = fft_convolution(signal.data(), signal_values.size(),
                                          kernel.data(), kernel_values.size(),
                                          BoundaryMode::ZeroPadding);
            (void)result;
        },
        std::runtime_error)
        << "the convolution currently fails inside cuFFT; see the comment above";
    (void)cudaGetLastError();
}

TEST(HaarWaveletTest, ForwardCountsTheLevelsOfAPowerOfTwoLength) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    DeviceBuffer signal(values);

    const auto result = haar_wavelet_forward(signal.data(), values.size());
    EXPECT_EQ(result.levels, 3u) << "8 samples halve down to 1 in three steps";
}

TEST(HaarWaveletTest, ZeroLengthIsHandled) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> empty;
    DeviceBuffer signal(empty);

    const auto result = haar_wavelet_forward(signal.data(), 0);
    EXPECT_EQ(result.levels, 0u);
}

// Regression: a level count larger than the size allows drove curr_n to zero and
// launched the reconstruct kernel with a zero grid.
TEST(HaarWaveletTest, InverseRejectsALevelCountThatCannotFitTheSize) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> values(4, 1.0f);
    DeviceBuffer coefficients_source(values);

    WaveletResult bogus{};
    bogus.coefficients = memory::Buffer<float>(4);
    cudaMemcpy(bogus.coefficients.data(), coefficients_source.data(),
               4 * sizeof(float), cudaMemcpyDeviceToDevice);
    bogus.levels = 5;  // 4 samples cannot be decomposed five times

    EXPECT_THROW(haar_wavelet_inverse(bogus, 4), std::invalid_argument);
}

// KNOWN DEFECT, not asserted here: forward then inverse does not reconstruct the
// input, because haar_wavelet_forward overwrites coefficients[0..curr_n/2) with
// the approximation at every level but never writes that level's DETAIL
// coefficients anywhere — it computes them into a temporary and drops it. The
// inverse therefore has nothing to add back. Pinning the current output would
// enshrine wrong numbers, and fixing it changes the coefficient layout, which is
// a separate decision. The test records that the round trip is lossy today.
TEST(HaarWaveletTest, RoundTripIsCurrentlyLossy) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    DeviceBuffer signal(values);

    const auto forward = haar_wavelet_forward(signal.data(), values.size());
    ASSERT_EQ(forward.levels, 2u);

    const auto reconstructed = haar_wavelet_inverse(forward, values.size());
    const auto output = to_host(reconstructed, values.size());
    ASSERT_EQ(output.size(), values.size());

    bool exact = true;
    for (size_t i = 0; i < values.size(); ++i) {
        if (std::abs(output[i] - values[i]) > 1e-4f) {
            exact = false;
        }
    }
    EXPECT_FALSE(exact)
        << "the round trip now reconstructs exactly — update this test and the "
           "comment above, the detail-coefficient defect appears to be fixed";
}

}  // namespace cuda::signal::test
