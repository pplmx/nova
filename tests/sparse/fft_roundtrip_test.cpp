/**
 * @file fft_roundtrip_test.cpp
 * @brief Numerical correctness for the forward/inverse FFT pair
 *
 * The existing FFT suites cover plan construction, enum values and configuration
 * defaults: fft_inverse_test.cpp checks that FFTPlan objects are truthy, and
 * fft_accuracy_test.cpp checks type traits and Direction/TransformType enumerators.
 * Neither ever runs a transform, so nothing in the tree verified that the forward and
 * inverse implementations agree on an answer.
 *
 * That gap matters because this module has been wrong before: fft_convolution once
 * failed with CUFFT_INVALID_VALUE for every input, which a plan-property test cannot
 * detect. The cases below run real transforms and compare against values computed in
 * host code.
 *
 * Scaling convention: cuFFT is unnormalised, so inverse(forward(x)) returns x * N.
 * Every expectation below accounts for that rather than assuming a normalised
 * transform.
 */

#include <cuda/fft/fft.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <cmath>
#include <vector>

namespace cuda::fft::test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

/// Owns a device allocation and frees it at scope exit.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        cudaMalloc(&ptr_, (count > 0 ? count : 1) * sizeof(T));
        cudaMemset(ptr_, 0, (count > 0 ? count : 1) * sizeof(T));
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    void upload(const std::vector<T>& host) {
        cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    std::vector<T> download() const {
        std::vector<T> host(count_, T{});
        cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost);
        return host;
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

}  // namespace

TEST(FFTRoundTripTest, InverseUndoesForwardUpToTheScaleFactor) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    constexpr size_t n = 64;
    std::vector<float> input(n);
    for (size_t i = 0; i < n; ++i) {
        // A sum of two sinusoids, so a wrong direction or a mirrored frequency is
        // visible rather than accidentally symmetric.
        input[i] = std::sin(2.0 * M_PI * 3.0 * static_cast<double>(i) / n) +
                   0.5f * std::cos(2.0 * M_PI * 11.0 * static_cast<double>(i) / n);
    }

    DeviceBuffer<float> d_input(n);
    DeviceBuffer<cuComplex> d_spectrum(n);
    DeviceBuffer<float> d_output(n);
    d_input.upload(input);

    FFTPlan plan(n, Direction::Forward, TransformType::RealToComplex);
    ASSERT_TRUE(static_cast<bool>(plan)) << "the plan could not be created";

    plan.forward(d_input.data(), d_spectrum.data());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    FFTPlan inverse_plan(n, Direction::Inverse, TransformType::ComplexToReal);
    ASSERT_TRUE(static_cast<bool>(inverse_plan));
    inverse_plan.inverse(d_spectrum.data(), d_output.data());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto result = d_output.download();
    // cuFFT is unnormalised: the round trip returns input * n.
    const float scale = static_cast<float>(n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(result[i] / scale, input[i], 1e-4f)
            << "element " << i << " did not survive the round trip";
    }
}

TEST(FFTRoundTripTest, AConstantSignalHasAllEnergyInTheDcBin) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The transform of a constant c has a DC bin of c * N and zeros everywhere else.
    // This pins the forward direction against an analytically known spectrum, which a
    // round-trip alone cannot do: an implementation that consistently mislabels the
    // bins would still invert correctly.
    constexpr size_t n = 32;
    constexpr float value = 3.0f;
    const std::vector<float> input(n, value);

    DeviceBuffer<float> d_input(n);
    DeviceBuffer<cuComplex> d_spectrum(n);
    d_input.upload(input);

    FFTPlan plan(n, Direction::Forward, TransformType::RealToComplex);
    ASSERT_TRUE(static_cast<bool>(plan));
    plan.forward(d_input.data(), d_spectrum.data());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto spectrum = d_spectrum.download();
    EXPECT_NEAR(spectrum[0].x, value * static_cast<float>(n), 1e-3f)
        << "the DC bin should hold the sum of the samples";
    EXPECT_NEAR(spectrum[0].y, 0.0f, 1e-3f) << "a real input has no imaginary DC part";

    for (size_t i = 1; i < n / 2; ++i) {
        EXPECT_NEAR(spectrum[i].x, 0.0f, 1e-3f) << "bin " << i << " should be empty";
        EXPECT_NEAR(spectrum[i].y, 0.0f, 1e-3f) << "bin " << i << " should be empty";
    }
}

TEST(FFTRoundTripTest, APureToneLandsInItsOwnBin) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A signal of exactly k cycles across the window has all its energy in bin k (and
    // its conjugate counterpart), with magnitude N/2 per side.
    constexpr size_t n = 64;
    constexpr size_t k = 5;
    std::vector<float> input(n);
    for (size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(
            std::cos(2.0 * M_PI * static_cast<double>(k) * static_cast<double>(i) / n));
    }

    DeviceBuffer<float> d_input(n);
    DeviceBuffer<cuComplex> d_spectrum(n);
    d_input.upload(input);

    FFTPlan plan(n, Direction::Forward, TransformType::RealToComplex);
    ASSERT_TRUE(static_cast<bool>(plan));
    plan.forward(d_input.data(), d_spectrum.data());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto spectrum = d_spectrum.download();
    const float expected = static_cast<float>(n) / 2.0f;

    EXPECT_NEAR(spectrum[k].x, expected, 1e-3f)
        << "bin " << k << " should carry the tone's energy";

    // Bins away from k must be empty.
    for (size_t i = 0; i < n / 2; ++i) {
        if (i == k) {
            continue;
        }
        const float magnitude =
            std::sqrt(spectrum[i].x * spectrum[i].x + spectrum[i].y * spectrum[i].y);
        EXPECT_LT(magnitude, 1e-2f) << "bin " << i << " should be empty";
    }
}

TEST(FFTRoundTripTest, ZeroInputProducesAZeroSpectrum) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The simplest input exercises the whole pipeline without any rounding argument.
    constexpr size_t n = 16;
    const std::vector<float> input(n, 0.0f);

    DeviceBuffer<float> d_input(n);
    DeviceBuffer<cuComplex> d_spectrum(n);
    d_input.upload(input);

    FFTPlan plan(n, Direction::Forward, TransformType::RealToComplex);
    ASSERT_TRUE(static_cast<bool>(plan));
    plan.forward(d_input.data(), d_spectrum.data());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "the forward transform reported an error";

    for (const auto& bin : d_spectrum.download()) {
        EXPECT_FLOAT_EQ(bin.x, 0.0f);
        EXPECT_FLOAT_EQ(bin.y, 0.0f);
    }
}

TEST(FFTRoundTripTest, RoundTripHoldsForSeveralPowerOfTwoSizes) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Small and large windows, so a size-dependent plan or launch bug shows up.
    for (const size_t n : {size_t{8}, size_t{128}, size_t{1024}}) {
        std::vector<float> input(n);
        for (size_t i = 0; i < n; ++i) {
            input[i] = static_cast<float>(i % 17) - 8.0f;
        }

        DeviceBuffer<float> d_input(n);
        DeviceBuffer<cuComplex> d_spectrum(n);
        DeviceBuffer<float> d_output(n);
        d_input.upload(input);

        FFTPlan forward_plan(n, Direction::Forward, TransformType::RealToComplex);
        ASSERT_TRUE(static_cast<bool>(forward_plan)) << "size " << n;
        forward_plan.forward(d_input.data(), d_spectrum.data());

        FFTPlan inverse_plan(n, Direction::Inverse, TransformType::ComplexToReal);
        ASSERT_TRUE(static_cast<bool>(inverse_plan)) << "size " << n;
        inverse_plan.inverse(d_spectrum.data(), d_output.data());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "size " << n;

        const auto result = d_output.download();
        const float scale = static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            ASSERT_NEAR(result[i] / scale, input[i], 1e-3f)
                << "size " << n << ", element " << i;
        }
    }
}

}  // namespace cuda::fft::test
