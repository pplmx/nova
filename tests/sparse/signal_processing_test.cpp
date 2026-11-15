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


// The convolution used to raise CUFFT_INVALID_VALUE: it ran one R2C plan and
// then called cufftExecC2R on that same plan with frequency data, never
// multiplying the two spectra. It now does the full transform pipeline.
TEST(FftConvolutionTest, MatchesDirectConvolution) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> signal_values = {1.0f, 2.0f, 3.0f};
    const std::vector<float> kernel_values = {1.0f, 1.0f};
    DeviceBuffer signal(signal_values);
    DeviceBuffer kernel(kernel_values);

    const auto result = fft_convolution(signal.data(), signal_values.size(),
                                        kernel.data(), kernel_values.size(),
                                        BoundaryMode::ZeroPadding);
    ASSERT_EQ(result.output_size, signal_values.size() + kernel_values.size() - 1);

    const auto output = to_host(result.output, result.output_size);
    const std::vector<float> expected = {1.0f, 3.0f, 5.0f, 3.0f};
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-5f) << "sample " << i;
    }
}

TEST(FftConvolutionTest, AgreesWithDirectConvolutionAcrossShapes) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    struct Case {
        std::vector<float> signal;
        std::vector<float> kernel;
        const char* label;
    };
    const std::vector<Case> cases = {
        {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, "uniform 5x3"},
        {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {0.5f}, "scale 8x1"},
        {{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
         {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
         "impulse 9x5"},
        {{-1.0f, -2.0f, -3.0f}, {1.0f, -1.0f}, "negatives 3x2"},
        {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
          13.0f},
         {1.0f, 2.0f},
         "13x2 non-power-of-two"},
    };

    for (const Case& test_case : cases) {
        DeviceBuffer signal(test_case.signal);
        DeviceBuffer kernel(test_case.kernel);

        const auto result = fft_convolution(signal.data(), test_case.signal.size(),
                                            kernel.data(), test_case.kernel.size(),
                                            BoundaryMode::ZeroPadding);

        std::vector<float> reference(test_case.signal.size() + test_case.kernel.size() - 1,
                                     0.0f);
        for (size_t i = 0; i < test_case.signal.size(); ++i) {
            for (size_t j = 0; j < test_case.kernel.size(); ++j) {
                reference[i + j] += test_case.signal[i] * test_case.kernel[j];
            }
        }

        ASSERT_EQ(result.output_size, reference.size()) << test_case.label;
        const auto output = to_host(result.output, result.output_size);
        for (size_t i = 0; i < reference.size(); ++i) {
            EXPECT_NEAR(output[i], reference[i], 1e-3f)
                << test_case.label << " sample " << i;
        }
    }
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


// The forward transform used to compute each level's detail coefficients into a
// temporary and never store them, and the inverse rebuilt the full-length region
// first, overwriting detail the shallower levels still needed. Both are fixed,
// so the round trip is now exact.
TEST(HaarWaveletTest, ForwardThenInverseReconstructsTheInput) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    for (const size_t n : {size_t{4}, size_t{8}, size_t{16}, size_t{32}}) {
        std::vector<float> values(n);
        for (size_t i = 0; i < n; ++i) {
            values[i] = static_cast<float>(i + 1);
        }
        DeviceBuffer signal(values);

        const auto forward = haar_wavelet_forward(signal.data(), values.size());
        const auto reconstructed = haar_wavelet_inverse(forward, values.size());
        const auto output = to_host(reconstructed, values.size());

        ASSERT_EQ(output.size(), values.size()) << "n=" << n;
        for (size_t i = 0; i < values.size(); ++i) {
            EXPECT_NEAR(output[i], values[i], 1e-4f) << "n=" << n << " sample " << i;
        }
    }
}

// A constant signal has zero detail, so it would hide a broken reconstruction.
// The alternating signal exercises the detail coefficients at every level.
TEST(HaarWaveletTest, AlternatingSignalSurvivesTheRoundTrip) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> values = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
    DeviceBuffer signal(values);

    const auto forward = haar_wavelet_forward(signal.data(), values.size());
    const auto reconstructed = haar_wavelet_inverse(forward, values.size());
    const auto output = to_host(reconstructed, values.size());

    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(output[i], values[i], 1e-4f) << "sample " << i;
    }
}

}  // namespace cuda::signal::test
