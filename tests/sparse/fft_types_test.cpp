/**
 * @file fft_types_test.cpp
 * @brief Contracts for the FFT type definitions
 *
 * include/cuda/fft/fft_types.h holds the enums and helpers every FFT caller
 * uses. It had no tests, and two defects lived in it:
 *
 * - is_power_of_two(0) returned true. 0 & (0-1) wraps to 0, so the bare bit
 *   trick reports zero as a power of two; callers sizing a plan or a buffer from
 *   it then asked for a zero-length transform.
 *
 * - Result::NotSupported was defined as CUFFT_UNALIGNED_DATA (0x9), while the
 *   real CUFFT_NOT_SUPPORTED is 0x10. A genuine "this configuration is not
 *   supported" answer was therefore reported as "the data is unaligned", and
 *   nine real cufftResult codes had no enumerator at all.
 */

#include <cuda/fft/fft_types.h>

#include <gtest/gtest.h>

#include <array>
#include <type_traits>

namespace cuda::fft::test {

    TEST(IsPowerOfTwoTest, ZeroIsNotAPowerOfTwo) {
        // Regression: this returned true and made a zero-sized FFT look valid.
        EXPECT_FALSE(is_power_of_two(0));
    }

    TEST(IsPowerOfTwoTest, OneAndTheDoublingSequenceAre) {
        EXPECT_TRUE(is_power_of_two(1));
        EXPECT_TRUE(is_power_of_two(2));
        EXPECT_TRUE(is_power_of_two(4));
        EXPECT_TRUE(is_power_of_two(512));
        EXPECT_TRUE(is_power_of_two(size_t {1} << 40));
    }

    TEST(IsPowerOfTwoTest, NeighboursOfAPowerAreNot) {
        EXPECT_FALSE(is_power_of_two(3));
        EXPECT_FALSE(is_power_of_two(5));
        EXPECT_FALSE(is_power_of_two(6));
        EXPECT_FALSE(is_power_of_two(7));
        EXPECT_FALSE(is_power_of_two(1023));
        EXPECT_FALSE(is_power_of_two(1025));
    }

    TEST(IsPowerOfTwoTest, IsUsableInAConstantExpression) {
        static_assert(is_power_of_two(64));
        static_assert(!is_power_of_two(0));
        static_assert(!is_power_of_two(48));
        SUCCEED();
    }

    // Regression: NotSupported pointed at the unaligned-data code.
    TEST(ResultTest, NotSupportedIsItsOwnCode) {
        EXPECT_EQ(static_cast<int>(Result::NotSupported), static_cast<int>(CUFFT_NOT_SUPPORTED));
        EXPECT_NE(static_cast<int>(Result::NotSupported), static_cast<int>(CUFFT_UNALIGNED_DATA)) << "a supported/unsupported answer is not an alignment complaint";
    }

    TEST(ResultTest, EveryMemberMatchesItsCufftCode) {
        // Pinning each member to the real constant is what stops the table drifting
        // again; the previous bug was a single mis-wired entry.
        EXPECT_EQ(static_cast<int>(Result::Success), static_cast<int>(CUFFT_SUCCESS));
        EXPECT_EQ(static_cast<int>(Result::InvalidPlan), static_cast<int>(CUFFT_INVALID_PLAN));
        EXPECT_EQ(static_cast<int>(Result::AllocFailed), static_cast<int>(CUFFT_ALLOC_FAILED));
        EXPECT_EQ(static_cast<int>(Result::InvalidType), static_cast<int>(CUFFT_INVALID_TYPE));
        EXPECT_EQ(static_cast<int>(Result::InvalidValue), static_cast<int>(CUFFT_INVALID_VALUE));
        EXPECT_EQ(static_cast<int>(Result::InternalError), static_cast<int>(CUFFT_INTERNAL_ERROR));
        EXPECT_EQ(static_cast<int>(Result::ExecFailed), static_cast<int>(CUFFT_EXEC_FAILED));
        EXPECT_EQ(static_cast<int>(Result::SetupFailed), static_cast<int>(CUFFT_SETUP_FAILED));
        EXPECT_EQ(static_cast<int>(Result::InvalidSize), static_cast<int>(CUFFT_INVALID_SIZE));
        EXPECT_EQ(static_cast<int>(Result::UnalignedData), static_cast<int>(CUFFT_UNALIGNED_DATA));
        EXPECT_EQ(static_cast<int>(Result::IncompleteParameterList), static_cast<int>(CUFFT_INCOMPLETE_PARAMETER_LIST));
        EXPECT_EQ(static_cast<int>(Result::InvalidDevice), static_cast<int>(CUFFT_INVALID_DEVICE));
        EXPECT_EQ(static_cast<int>(Result::ParseError), static_cast<int>(CUFFT_PARSE_ERROR));
        EXPECT_EQ(static_cast<int>(Result::NoWorkSpace), static_cast<int>(CUFFT_NO_WORKSPACE));
        EXPECT_EQ(static_cast<int>(Result::NotImplemented), static_cast<int>(CUFFT_NOT_IMPLEMENTED));
        EXPECT_EQ(static_cast<int>(Result::LicenseError), static_cast<int>(CUFFT_LICENSE_ERROR));
    }

    TEST(ResultTest, MembersAreDistinct) {
        // A duplicated value would make two different failures indistinguishable.
        constexpr std::array<Result, 17> all = {
            Result::Success,
            Result::InvalidPlan,
            Result::AllocFailed,
            Result::InvalidType,
            Result::InvalidValue,
            Result::InternalError,
            Result::ExecFailed,
            Result::SetupFailed,
            Result::InvalidSize,
            Result::UnalignedData,
            Result::IncompleteParameterList,
            Result::InvalidDevice,
            Result::ParseError,
            Result::NoWorkSpace,
            Result::NotImplemented,
            Result::LicenseError,
            Result::NotSupported,
        };
        for (size_t i = 0; i < all.size(); ++i) {
            for (size_t j = i + 1; j < all.size(); ++j) {
                EXPECT_NE(static_cast<int>(all[i]), static_cast<int>(all[j])) << "members " << i << " and " << j << " share a value";
            }
        }
    }

    TEST(DirectionTest, MatchesCufftConventions) {
        // cuFFT defines forward as -1 and inverse as +1; the enum must not swap them.
        EXPECT_EQ(static_cast<int>(Direction::Forward), CUFFT_FORWARD);
        EXPECT_EQ(static_cast<int>(Direction::Inverse), CUFFT_INVERSE);
        EXPECT_EQ(static_cast<int>(Direction::Forward), -1);
        EXPECT_EQ(static_cast<int>(Direction::Inverse), 1);
    }

    TEST(TransformTypeTest, MatchesCufftTransformKinds) {
        EXPECT_EQ(static_cast<int>(TransformType::RealToComplex), CUFFT_R2C);
        EXPECT_EQ(static_cast<int>(TransformType::DoubleRealToComplex), CUFFT_D2Z);
        EXPECT_EQ(static_cast<int>(TransformType::ComplexToReal), CUFFT_C2R);
        EXPECT_EQ(static_cast<int>(TransformType::ComplexToComplex), CUFFT_C2C);
    }

    TEST(ComplexTraitsTest, MapToCufftTypes) {
        static_assert(std::is_same_v<ComplexT<float>, cuComplex>);
        static_assert(std::is_same_v<ComplexT<double>, cuDoubleComplex>);
        SUCCEED();
    }

    TEST(FFTConfigTest, DefaultsDescribeAOneElementForwardTransform) {
        const FFTConfig config;

        EXPECT_EQ(config.size, 0u);
        EXPECT_EQ(config.direction, Direction::Forward);
        EXPECT_EQ(config.batch, 1) << "a batch of zero would transform nothing";
        EXPECT_EQ(config.stream_id, 0);
        EXPECT_EQ(config.type, TransformType::RealToComplex);
    }

}  // namespace cuda::fft::test
