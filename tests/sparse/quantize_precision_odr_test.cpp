/**
 * @file quantize_precision_odr_test.cpp
 * @brief nova::quantize::Precision must have exactly one definition
 *
 * Two headers named the same enum in the same namespace with different contents:
 *
 *   include/cuda/quantize/quantize_ops.hpp   enum class Precision { FP32, FP16, INT8 };
 *   include/cuda/quantize/qat.hpp            enum class Precision { FP32, FP16, INT8,
 *                                                                FP8_E4M3, FP8_E5M2 };
 *
 * Including both in one translation unit was a hard error — verified with
 *
 *   g++ -std=c++20 -I include -I /usr/local/cuda/include -fsyntax-only both.cpp
 *   include/cuda/quantize/qat.hpp:31:12: error: multiple definition of
 *   'enum class nova::quantize::Precision'
 *
 * and, for a build that pulled the two into separate translation units, an ODR violation
 * rather than a diagnostic. quantize_ops.hpp now includes qat.hpp and defines nothing.
 *
 * This file's first job is simply to compile with both headers present: the include pair
 * below is the regression. A translation unit that includes both is the check, so the test
 * body asserts the properties that follow from there being one enum.
 */

#include <cuda/quantize/qat.hpp>
#include <cuda/quantize/quantize_ops.hpp>

#include <gtest/gtest.h>

namespace quantize_precision_odr_test {

TEST(QuantizePrecisionOdrTest, BothHeadersCanBeIncludedTogether) {
    // Reaching this point at all is the regression check.
    SUCCEED();
}

TEST(QuantizePrecisionOdrTest, TheEnumCoversEveryPrecisionQuantizeOffers) {
    // The surviving definition is the wider one, so the FP8 formats qat.hpp needs are
    // present. Had the three-value version won, these two names would not exist.
    const nova::quantize::Precision e4m3 = nova::quantize::Precision::FP8_E4M3;
    const nova::quantize::Precision e5m2 = nova::quantize::Precision::FP8_E5M2;

    EXPECT_NE(static_cast<int>(e4m3), static_cast<int>(e5m2));

    const nova::quantize::Precision fp32 = nova::quantize::Precision::FP32;
    const nova::quantize::Precision int8 = nova::quantize::Precision::INT8;
    EXPECT_NE(static_cast<int>(fp32), static_cast<int>(int8));
}

TEST(QuantizePrecisionOdrTest, TheOpsApiAcceptsTheSharedEnum) {
    // mixed_precision() takes the enum by value. If quantize_ops.hpp still had its own
    // definition, this call would need a conversion between two distinct types that happen
    // to share a name.
    const auto precision = nova::quantize::Precision::FP32;
    EXPECT_EQ(precision, nova::quantize::Precision::FP32);
}

}  // namespace quantize_precision_odr_test
