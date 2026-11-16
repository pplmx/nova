/**
 * @file fp_determinism_verdict_test.cpp
 * @brief Contracts for what a determinism check actually establishes
 *
 * src/testing/fp_determinism.cpp compares floating-point outputs across iterations.
 * Its early return claimed a verdict it never reached:
 *
 *     DeterminismResult result;
 *     result.deterministic = true;
 *     ...
 *     if (iterations < 2 || data == nullptr || size == 0) {
 *         return result;                       // deterministic == true
 *     }
 *
 * Measured before the fix:
 *
 *     null data, iterations = 16 -> deterministic = true
 *     single iteration           -> deterministic = true
 *
 * In both cases nothing was compared, yet a caller reading the flag would conclude
 * the computation had been shown to be reproducible. The struct had no way to
 * express "not checked", so the false positive was indistinguishable from a pass.
 *
 * A `verified` flag now records whether a comparison happened, and the unverifiable
 * cases report deterministic = false so the check cannot be mistaken for approval.
 */

#include <cuda/testing/fp_determinism.h>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace cuda::testing::test {

namespace {

using cuda::testing::FPDeterminismControl;

FPDeterminismControl::DeterminismResult check(const std::vector<double>& values,
                                              size_t iterations) {
    return FPDeterminismControl::instance().verify_determinism(
        values.empty() ? nullptr : values.data(), values.size(), iterations);
}

}  // namespace

// Regression: this reported deterministic = true having compared nothing.
TEST(FpDeterminismVerdictTest, NullDataIsNotACertificateOfDeterminism) {
    const auto result = check({}, 16);

    EXPECT_FALSE(result.verified) << "no buffer was inspected";
    EXPECT_FALSE(result.deterministic)
        << "an unperformed check must not read as a passing one";
    EXPECT_EQ(result.iteration_count, 16u) << "the request is still echoed";
    EXPECT_DOUBLE_EQ(result.max_difference, 0.0);
}

// Regression: a single iteration also claimed determinism.
TEST(FpDeterminismVerdictTest, ASingleIterationCannotEstablishDeterminism) {
    const std::vector<double> values = {1.0};

    const auto result = check(values, 1);

    EXPECT_FALSE(result.verified)
        << "one sample has nothing to compare against";
    EXPECT_FALSE(result.deterministic);
}

TEST(FpDeterminismVerdictTest, IdenticalValuesAreVerifiedDeterministic) {
    const std::vector<double> values = {2.5, 2.5, 2.5, 2.5};

    const auto result = check(values, 4);

    EXPECT_TRUE(result.verified) << "four samples were compared";
    EXPECT_TRUE(result.deterministic);
    EXPECT_DOUBLE_EQ(result.max_difference, 0.0);
    EXPECT_DOUBLE_EQ(result.mean_difference, 0.0);
}

TEST(FpDeterminismVerdictTest, DifferingValuesAreVerifiedNondeterministic) {
    // Each element is compared against the first, and the loop starts at index 1,
    // so the differences are 0.5 and 0.5 over two samples: max 0.5, mean 0.5.
    const std::vector<double> values = {1.0, 1.5, 1.5};

    const auto result = check(values, 3);

    EXPECT_TRUE(result.verified);
    EXPECT_FALSE(result.deterministic);
    EXPECT_NEAR(result.max_difference, 0.5, 1e-12);
    EXPECT_NEAR(result.mean_difference, 0.5, 1e-12)
        << "the mean spans iterations - 1 differences, not all iterations";
}

TEST(FpDeterminismVerdictTest, VerifiedImpliesDeterministicAndCountedTogether) {
    // The two flags must never contradict each other: a verified pass means a
    // comparison happened and it matched.
    const std::vector<std::vector<double>> cases = {
        {1.0, 1.0},        // identical
        {1.0, 2.0},        // differing
        {0.0, 0.0, 0.0},   // identical, three samples
        {3.0, 3.0, 4.0},   // differing at the end
    };

    for (const auto& values : cases) {
        const auto result = check(values, values.size());
        ASSERT_TRUE(result.verified)
            << "a list of " << values.size() << " samples is comparable";
        EXPECT_EQ(result.deterministic, result.max_difference == 0.0)
            << "the verdict must follow the measured difference";
    }
}

TEST(FpDeterminismVerdictTest, ExactBitEqualityIsTheCriterion) {
    // Floating-point determinism here means bit-identical, not "close enough": a
    // difference at the last bit counts.
    std::vector<double> values = {1.0, 1.0};
    values[1] = std::nextafter(1.0, 2.0);

    const auto result = check(values, 2);

    ASSERT_TRUE(result.verified);
    EXPECT_FALSE(result.deterministic)
        << "a one-ulp difference is still a difference";
    EXPECT_GT(result.max_difference, 0.0);
}

TEST(FpDeterminismVerdictTest, SizeZeroWithNonNullPointerIsAlsoUnverified) {
    // size == 0 is its own early-return condition, independent of the null check.
    double dummy = 1.0;
    const auto result = FPDeterminismControl::instance().verify_determinism(
        static_cast<const void*>(&dummy), 0, 8);

    EXPECT_FALSE(result.verified);
    EXPECT_FALSE(result.deterministic);
}

}  // namespace cuda::testing::test
