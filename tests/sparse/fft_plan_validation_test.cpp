/**
 * @file fft_plan_validation_test.cpp
 * @brief Contracts for FFTPlan construction and cuFFT error reporting
 *
 * src/cuda/fft/fft.cu builds cuFFT plans. Five of its sixteen functions had any
 * test reference, and two defects made failures hard to act on:
 *
 * - Nothing validated the transform dimensions. A size above INT_MAX narrowed to
 *   an implementation-defined int before reaching cuFFT, so the plan described a
 *   different length than the caller asked for; a zero size reached cuFFT and
 *   came back as a generic failure. Both now throw std::invalid_argument naming
 *   the offending value.
 *
 * - CUFFT_CHECK reported only the numeric code: "cuFFT error: 8". That code is
 *   CUFFT_INVALID_SIZE, which is exactly the piece of information the caller
 *   needs, and the project already had a Result enum listing every code. A
 *   result_name() helper now maps each code, and the message carries both the
 *   name and the number.
 */

#include <cuda/fft/fft.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
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

/// Runs `call` and returns the exception message, or an empty string if it threw
/// nothing. Keeps the catch out of the test body so EXPECT_THROW style macros do
/// not choke on commas in the constructor arguments.
template <typename Callable>
std::string message_from(Callable&& call) {
    try {
        call();
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

}  // namespace

// Regression: a zero size reached cuFFT and came back as an unlabelled failure.
TEST(FftPlanValidationTest, ZeroSizeIsRejectedWithAClearMessage) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::string message = message_from([] {
        FFTPlan plan(0, Direction::Forward, TransformType::RealToComplex);
    });

    EXPECT_FALSE(message.empty()) << "a zero-length transform must not be built";
    EXPECT_NE(message.find("positive"), std::string::npos)
        << "the message should say the size must be positive: " << message;
}

// Regression: the size narrowed to int without a check.
TEST(FftPlanValidationTest, SizeAboveIntMaxIsRejected) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const size_t too_large = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
    const std::string message = message_from([too_large] {
        FFTPlan plan(too_large, Direction::Forward, TransformType::RealToComplex);
    });

    EXPECT_FALSE(message.empty())
        << "narrowing to int would silently change the transform length";
    EXPECT_NE(message.find("exceeds"), std::string::npos)
        << "the message should name the limit it exceeded: " << message;
}

TEST(FftPlanValidationTest, Zero2dDimensionIsRejected) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::string message = message_from([] {
        FFTPlan plan(0, 4, Direction::Forward, TransformType::RealToComplex);
    });
    EXPECT_FALSE(message.empty());
    EXPECT_NE(message.find("2D"), std::string::npos)
        << "the message should say which constructor failed: " << message;
}

TEST(FftPlanValidationTest, Zero3dDimensionIsRejected) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::string message = message_from([] {
        FFTPlan plan(2, 4, 0, Direction::Forward, TransformType::RealToComplex);
    });
    EXPECT_FALSE(message.empty());
    EXPECT_NE(message.find("3D"), std::string::npos) << message;
}

// Regression: the macro printed the bare code.
TEST(FftPlanValidationTest, ResultCodesHaveReadableNames) {
    EXPECT_STREQ(result_name(CUFFT_SUCCESS), "CUFFT_SUCCESS");
    EXPECT_STREQ(result_name(CUFFT_INVALID_SIZE), "CUFFT_INVALID_SIZE");
    EXPECT_STREQ(result_name(CUFFT_ALLOC_FAILED), "CUFFT_ALLOC_FAILED");
    EXPECT_STREQ(result_name(CUFFT_INVALID_VALUE), "CUFFT_INVALID_VALUE");
    EXPECT_STREQ(result_name(CUFFT_NOT_SUPPORTED), "CUFFT_NOT_SUPPORTED");

    // An unmapped code still yields a usable string rather than a null pointer.
    const auto unknown = static_cast<cufftResult>(9999);
    ASSERT_NE(result_name(unknown), nullptr);
    EXPECT_STREQ(result_name(unknown), "CUFFT_UNKNOWN_RESULT");
}

TEST(FftPlanValidationTest, EveryResultEnumeratorHasADistinctName) {
    const cufftResult codes[] = {
        CUFFT_SUCCESS,
        CUFFT_INVALID_PLAN,
        CUFFT_ALLOC_FAILED,
        CUFFT_INVALID_TYPE,
        CUFFT_INVALID_VALUE,
        CUFFT_INTERNAL_ERROR,
        CUFFT_EXEC_FAILED,
        CUFFT_SETUP_FAILED,
        CUFFT_INVALID_SIZE,
        CUFFT_UNALIGNED_DATA,
        CUFFT_INCOMPLETE_PARAMETER_LIST,
        CUFFT_INVALID_DEVICE,
        CUFFT_PARSE_ERROR,
        CUFFT_NO_WORKSPACE,
        CUFFT_NOT_IMPLEMENTED,
        CUFFT_LICENSE_ERROR,
        CUFFT_NOT_SUPPORTED,
    };

    for (size_t i = 0; i < std::size(codes); ++i) {
        for (size_t j = i + 1; j < std::size(codes); ++j) {
            EXPECT_STRNE(result_name(codes[i]), result_name(codes[j]))
                << "codes " << static_cast<int>(codes[i]) << " and "
                << static_cast<int>(codes[j]) << " share a name";
        }
    }
}

TEST(FftPlanValidationTest, ValidSizeBuildsAPlanOfThatSize) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    FFTPlan plan(16, Direction::Forward, TransformType::RealToComplex);
    EXPECT_EQ(plan.size(), 16u);

    FFTPlan square(4, 8, Direction::Forward, TransformType::RealToComplex);
    EXPECT_EQ(square.size(), 32u) << "a 2D plan reports nx * ny";

    FFTPlan cube(2, 3, 4, Direction::Forward, TransformType::RealToComplex);
    EXPECT_EQ(cube.size(), 24u) << "a 3D plan reports nx * ny * nz";
}

TEST(FftPlanValidationTest, RoundTripsARealSignal) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    constexpr size_t n = 8;
    std::vector<float> host_input(n, 0.0f);
    host_input[0] = 1.0f;  // an impulse has a flat spectrum

    float* device_input = nullptr;
    cuComplex* device_spectrum = nullptr;
    float* device_output = nullptr;
    ASSERT_EQ(cudaMalloc(&device_input, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_spectrum, n * sizeof(cuComplex)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_output, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_input, host_input.data(), n * sizeof(float),
                         cudaMemcpyHostToDevice), cudaSuccess);

    {
        FFTPlan forward(n, Direction::Forward, TransformType::RealToComplex);
        FFTPlan inverse(n, Direction::Inverse, TransformType::ComplexToReal);

        ASSERT_NO_THROW(forward.forward(device_input, device_spectrum, nullptr));
        // cuFFT's inverse is unnormalised, so the round trip scales by n.
        ASSERT_NO_THROW(inverse.inverse(device_spectrum, device_output, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    std::vector<float> host_output(n, 0.0f);
    ASSERT_EQ(cudaMemcpy(host_output.data(), device_output, n * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(host_output[i] / static_cast<float>(n), host_input[i], 1e-4f)
            << "element " << i;
    }

    cudaFree(device_input);
    cudaFree(device_spectrum);
    cudaFree(device_output);
}

TEST(FftPlanValidationTest, IsPowerOfTwoExcludesZero) {
    // Used by callers to decide whether a fast path applies; zero must not count.
    EXPECT_FALSE(is_power_of_two(0));
    EXPECT_TRUE(is_power_of_two(1));
    EXPECT_TRUE(is_power_of_two(2));
    EXPECT_TRUE(is_power_of_two(1024));
    EXPECT_FALSE(is_power_of_two(3));
    EXPECT_FALSE(is_power_of_two(1000));
}

}  // namespace cuda::fft::test
