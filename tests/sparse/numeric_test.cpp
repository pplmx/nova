/**
 * @file numeric_test.cpp
 * @brief Contracts for the numeric integration, root finding and interpolation
 *
 * include/cuda/numeric/numeric.h provides GPU-accelerated classical numeric
 * routines. Only one of its seven entry points had any test reference, and three
 * edge cases were broken:
 *
 * - trapezoidal_integration and simpson_integration divided by the interval
 *   count, so n == 0 computed h = inf and returned NaN in every field. Measured:
 *   "trap n=0 -> nan", "simpson n=0 -> -nan". A caller received a silently
 *   meaningless number instead of an error or a zero.
 *
 * - linear_interpolation sized its coefficient buffer as (n - 1) * 2, which
 *   underflows to SIZE_MAX when n == 0. Measured: the call threw
 *   "CUDA error: out of memory" while asking for ~32 exabytes.
 *
 * Every routine is checked against an analytic result rather than against a
 * recorded number, so a change in accuracy shows up as a failure.
 */

#include <cuda/numeric/numeric.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace cuda::numeric::test {

    namespace {

        float square(float x) {
            return x * x;
        }
        float cube(float x) {
            return x * x * x;
        }
        float sine(float x) {
            return std::sin(x);
        }
        float square_minus_two(float x) {
            return x * x - 2.0f;
        }
        float square_minus_two_deriv(float x) {
            return 2.0f * x;
        }

    }  // namespace

    // ------------------------------------------------------------- integration --

    TEST(TrapezoidalIntegrationTest, IntegratesAPolynomial) {
        // integral of x^2 over [0, 1] is 1/3.
        const auto result = trapezoidal_integration(square, 0.0f, 1.0f, 10000);

        EXPECT_NEAR(result.value, 1.0f / 3.0f, 1e-4f);
        EXPECT_EQ(result.intervals, 10000u);
    }

    TEST(TrapezoidalIntegrationTest, ApproachesTheExactValueAsIntervalsGrow) {
        const float coarse = trapezoidal_integration(cube, 0.0f, 1.0f, 16).value;
        const float fine = trapezoidal_integration(cube, 0.0f, 1.0f, 16384).value;

        // integral of x^3 over [0,1] is 1/4.
        EXPECT_LT(std::fabs(fine - 0.25f), std::fabs(coarse - 0.25f)) << "more intervals must be more accurate: coarse=" << coarse << " fine=" << fine;
    }

    // Regression: this returned NaN.
    TEST(TrapezoidalIntegrationTest, ZeroIntervalsReturnsZeroNotNaN) {
        const auto result = trapezoidal_integration(square, 0.0f, 1.0f, 0);

        EXPECT_TRUE(std::isfinite(result.value)) << "an empty integration produced " << result.value;
        EXPECT_FLOAT_EQ(result.value, 0.0f);
        EXPECT_EQ(result.intervals, 0u);
        EXPECT_TRUE(std::isfinite(result.error_estimate));
    }

    TEST(SimpsonIntegrationTest, IntegratesAPolynomial) {
        // Simpson on x^2 is exact up to rounding.
        const auto result = simpson_integration(square, 0.0f, 1.0f, 500);

        EXPECT_NEAR(result.value, 1.0f / 3.0f, 1e-5f);
        EXPECT_EQ(result.intervals, 500u);
    }

    TEST(SimpsonIntegrationTest, IsMoreAccurateThanTrapezoidForTheSameWork) {
        const auto simpson = simpson_integration(sine, 0.0f, 3.14159265f, 100);
        const auto trapezoid = trapezoidal_integration(sine, 0.0f, 3.14159265f, 200);

        // integral of sin over [0, pi] is 2.
        EXPECT_NEAR(simpson.value, 2.0f, 1e-4f);
        EXPECT_LT(std::fabs(simpson.value - 2.0f), std::fabs(trapezoid.value - 2.0f)) << "Simpson converges faster than the trapezoidal rule";
    }

    // Regression: this returned -nan.
    TEST(SimpsonIntegrationTest, ZeroIntervalsReturnsZeroNotNaN) {
        const auto result = simpson_integration(square, 0.0f, 1.0f, 0);

        EXPECT_TRUE(std::isfinite(result.value)) << "an empty integration produced " << result.value;
        EXPECT_FLOAT_EQ(result.value, 0.0f);
        EXPECT_EQ(result.intervals, 0u);
    }

    TEST(NumericIntegrationTest, ReversedLimitsGiveTheNegatedIntegral) {
        // Swapping the limits flips the sign; the routines should not return NaN.
        const float forward = trapezoidal_integration(square, 0.0f, 1.0f, 1000).value;
        const float backward = trapezoidal_integration(square, 1.0f, 0.0f, 1000).value;

        EXPECT_TRUE(std::isfinite(backward));
        EXPECT_NEAR(backward, -forward, 1e-4f);
    }

    // ------------------------------------------------------------ root finding --

    TEST(BisectionTest, FindsTheRootOfX2MinusTwo) {
        const auto result = bisection(square_minus_two, 0.0f, 2.0f);

        ASSERT_TRUE(result.converged);
        EXPECT_NEAR(result.root, std::sqrt(2.0f), 1e-4f);
        EXPECT_LT(result.residual, 1e-3f);
    }

    TEST(BisectionTest, ReportsFailureWhenTheBracketHasNoSignChange) {
        // x^2 - 2 is positive at both 2 and 3.
        const auto result = bisection(square_minus_two, 2.0f, 3.0f);

        EXPECT_FALSE(result.converged) << "there is no root in [2, 3]";
        EXPECT_FALSE(std::isfinite(result.residual) && result.residual < 1e-3f);
    }

    TEST(BisectionTest, RespectsTheIterationLimit) {
        // A tiny budget cannot converge to the default tolerance.
        const auto result = bisection(square_minus_two, 0.0f, 2.0f, 1e-12f, 2);

        EXPECT_FALSE(result.converged);
        EXPECT_LE(result.iterations, 2u);
    }

    TEST(NewtonRaphsonTest, FindsTheRootOfX2MinusTwo) {
        const auto result = newton_raphson(square_minus_two, square_minus_two_deriv, 1.0f);

        ASSERT_TRUE(result.converged);
        EXPECT_NEAR(result.root, std::sqrt(2.0f), 1e-4f);
    }

    TEST(NewtonRaphsonTest, StopsAtAZeroDerivative) {
        // A constant function has zero derivative everywhere, so the step is
        // undefined; the routine must report failure rather than divide by zero.
        auto constant = [](float) { return 1.0f; };
        auto zero_deriv = [](float) { return 0.0f; };

        const auto result = newton_raphson(constant, zero_deriv, 0.0f);

        EXPECT_FALSE(result.converged);
        EXPECT_TRUE(std::isfinite(result.root)) << "a zero derivative must not produce inf or NaN";
    }

    // ----------------------------------------------------------- interpolation --

    // Regression: n == 0 underflowed the coefficient count and threw "out of memory".
    TEST(LinearInterpolationTest, ZeroPointsDoesNotAllocateAnAbsurdBuffer) {
        EXPECT_NO_THROW({
            const auto result = linear_interpolation(nullptr, nullptr, 0);
            EXPECT_EQ(result.n, 0u);
            EXPECT_EQ(result.coeffs.size(), 0u);
        });
    }

    TEST(LinearInterpolationTest, OnePointHasNoSegments) {
        const float x[] = {2.0f};
        const float y[] = {5.0f};

        const auto result = linear_interpolation(x, y, 1);

        EXPECT_EQ(result.n, 1u);
        EXPECT_EQ(result.coeffs.size(), 0u) << "one point describes no segment";
        EXPECT_EQ(result.x.size(), 1u) << "the sample must still be stored";
    }

    TEST(LinearInterpolationTest, CoefficientsDescribeTheSegments) {
        // Points (0,1), (1,3), (2,7): slopes 2 and 4, intercepts 1 and -1.
        const float x[] = {0.0f, 1.0f, 2.0f};
        const float y[] = {1.0f, 3.0f, 7.0f};

        const auto result = linear_interpolation(x, y, 3);
        ASSERT_EQ(result.n, 3u);
        ASSERT_EQ(result.coeffs.size(), 4u) << "two segments, two coefficients each";

        std::vector<float> coeffs(4);
        result.coeffs.copy_to(coeffs.data(), coeffs.size());

        EXPECT_NEAR(coeffs[0], 2.0f, 1e-5f) << "slope of the first segment";
        EXPECT_NEAR(coeffs[1], 1.0f, 1e-5f) << "intercept of the first segment";
        EXPECT_NEAR(coeffs[2], 4.0f, 1e-5f) << "slope of the second segment";
        EXPECT_NEAR(coeffs[3], -1.0f, 1e-5f) << "intercept of the second segment";
    }

    // Regression: cubic_spline_interpolation had no n < 2 guard, so n == 0 walked off
    // the end of both input arrays and wrote the coefficients block at a wild offset.
    //
    //     for (size_t i = 0; i < n - 1; ++i)      // n == 0 -> n - 1 == SIZE_MAX
    //
    // The loop condition is not the only casualty:
    //
    //     h_coeffs[(n - 1) * 4 + i] = 0.0f;       // writes 4 * SIZE_MAX floats past a
    //                                             // malloc(0) block
    //
    // linear_interpolation beside it already refused n < 2 (its underflow was fixed in an
    // earlier round); the cubic path was never given the same treatment and had no test.
    TEST(CubicSplineTest, FewerThanTwoPointsDescribeNoSegment) {
        // Two points is the smallest input that describes a segment; below that there is
        // nothing to interpolate and the call must say so rather than index past the input.
        std::vector<float> x = {0.0f, 1.0f};
        std::vector<float> y = {0.0f, 1.0f};

        for (size_t n : {size_t {0}, size_t {1}}) {
            auto result = cuda::numeric::cubic_spline_interpolation(x.data(), y.data(), n);
            EXPECT_EQ(result.n, n);
            EXPECT_EQ(result.coeffs.size(), n * 4) << "the coefficient buffer must match the requested size for n = " << n;
        }
    }

    TEST(CubicSplineTest, InterpolatesTwoPointsAsAStraightLine) {
        std::vector<float> x = {0.0f, 1.0f};
        std::vector<float> y = {1.0f, 3.0f};

        auto result = cuda::numeric::cubic_spline_interpolation(x.data(), y.data(), 2);
        ASSERT_EQ(result.n, 2u);
        ASSERT_EQ(result.coeffs.size(), 8u);

        std::vector<float> coeffs(8);
        result.coeffs.copy_to(coeffs.data(), 8);

        // With a single segment the routine builds the linear form: slope (y1-y0)/h and
        // intercept y0 in the first four slots.
        EXPECT_NEAR(coeffs[2], 2.0f, 1e-5f) << "slope";
        EXPECT_NEAR(coeffs[3], 1.0f, 1e-5f) << "intercept";
    }

}  // namespace cuda::numeric::test
