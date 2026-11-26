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
#include <cuda/sparse/solver_workspace.hpp>

#include <gtest/gtest.h>
#include <stdexcept>

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

    // The segment slope divides by (x[i+1] - x[i]). A repeated or decreasing knot makes that zero or
    // negative, so the slope becomes inf and the intercept y - slope*x becomes inf - inf = NaN --
    // both written into the coefficient buffer and evaluated by every later sample, with no error
    // reported. cubic_spline_interpolation in the same file rejects exactly this input
    // ("a repeated or decreasing knot makes the segment width zero or negative and the system is not
    // solvable"); the linear form had no such check.
    TEST(LinearInterpolationTest, RejectsNonIncreasingKnots) {
        const float repeated_x[] = {0.0f, 0.0f, 1.0f};
        const float repeated_y[] = {0.0f, 1.0f, 2.0f};
        EXPECT_THROW((void)linear_interpolation(repeated_x, repeated_y, 3), std::invalid_argument) << "a zero-width segment divides by zero";

        const float decreasing_x[] = {0.0f, 5.0f, 2.0f};
        const float decreasing_y[] = {0.0f, 1.0f, 2.0f};
        EXPECT_THROW((void)linear_interpolation(decreasing_x, decreasing_y, 3), std::invalid_argument) << "a negative segment inverts the slope";
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

    // Regression: SolverWorkspace::resize() published size_ before resizing the containers,
    // and accepted negative sizes.
    //
    //     void resize(int size) {
    //         ...
    //         size_ = size;          // first
    //         r_.resize(size);       // then, and this can throw
    //
    // std::vector::resize takes size_t, so a negative `size` converts to a value near
    // SIZE_MAX and the call throws length_error or attempts an exabyte allocation. Either
    // way size_ had already been assigned: the object then reported the new size while the
    // vectors still held the old one, and the `size == size_` early return meant no later
    // resize() repaired it. Measured on the old ordering: a throwing resize left size_ at
    // the requested value and every container empty.
    TEST(SolverWorkspaceTest, ResizeIsAllOrNothing) {
        nova::sparse::SolverWorkspace<double> ws(4);
        ASSERT_EQ(ws.size(), 4);
        ASSERT_EQ(ws.r().size(), 4u);

        // A negative size is caller error, and it must not leave the object altered.
        EXPECT_THROW(ws.resize(-1), std::invalid_argument);
        EXPECT_EQ(ws.size(), 4) << "a refused resize must not change the reported size";
        EXPECT_EQ(ws.r().size(), 4u) << "and must not change the containers";
        EXPECT_EQ(ws.t().size(), 4u);

        // A normal resize still works and keeps everything in step.
        ws.resize(8);
        EXPECT_EQ(ws.size(), 8);
        for (const auto* v : {&ws.r(), &ws.p(), &ws.Ap(), &ws.r_tilde(), &ws.p_hat(), &ws.s(), &ws.t()}) {
            EXPECT_EQ(v->size(), 8u) << "every workspace vector tracks size()";
        }
    }

    TEST(SolverWorkspaceTest, ResetZeroesEveryBuffer) {
        nova::sparse::SolverWorkspace<double> ws(3);
        std::fill(ws.r().begin(), ws.r().end(), 5.0);
        std::fill(ws.t().begin(), ws.t().end(), 7.0);

        ws.reset();

        for (double v : ws.r()) {
            EXPECT_EQ(v, 0.0);
        }
        for (double v : ws.t()) {
            EXPECT_EQ(v, 0.0);
        }
    }

    // ---------------------------------------------------------------------------
    // cubic_spline_interpolation was not a spline: it wrote the linear form and left
    // the cubic and quadratic slots at zero, so every segment was a straight line.
    // Measured on sin over [0, pi] with five knots, midpoint error 7.033e-02, against
    // 1.064e-03 for the natural cubic spline it now computes.
    // ---------------------------------------------------------------------------

    namespace {

        /// Evaluate segment i of a coefficient buffer at local t.
        float eval_segment(const std::vector<float>& coeffs, size_t i, float t) {
            return coeffs[i * 4 + 0] * t * t * t + coeffs[i * 4 + 1] * t * t + coeffs[i * 4 + 2] * t + coeffs[i * 4 + 3];
        }

    }  // namespace

    // The defining property: the quadratic and cubic coefficients are not all zero, and the
    // result is substantially more accurate than piecewise linear on the same knots.
    TEST(CubicSplineTest, IsActuallyCubicAndBeatsPiecewiseLinear) {
        std::vector<float> x = {0.0f, static_cast<float>(M_PI) / 4.0f, static_cast<float>(M_PI) / 2.0f, static_cast<float>(3.0 * M_PI) / 4.0f, static_cast<float>(M_PI)};
        std::vector<float> y;
        for (float v : x) {
            y.push_back(std::sin(v));
        }

        auto result = cuda::numeric::cubic_spline_interpolation(x.data(), y.data(), x.size());
        std::vector<float> coeffs(result.coeffs.size());
        result.coeffs.copy_to(coeffs.data(), coeffs.size());

        // Before the fix every one of these was exactly 0.
        int nonzero = 0;
        for (size_t i = 0; i + 1 < x.size(); ++i) {
            if (coeffs[i * 4 + 0] != 0.0f || coeffs[i * 4 + 1] != 0.0f) {
                ++nonzero;
            }
        }
        EXPECT_EQ(nonzero, static_cast<int>(x.size() - 1)) << "every segment must carry a non-zero curvature term";

        double max_midpoint_error = 0.0;
        for (size_t i = 0; i + 1 < x.size(); ++i) {
            const float xm = 0.5f * (x[i] + x[i + 1]);
            const float t = xm - x[i];
            max_midpoint_error = std::max(max_midpoint_error, std::fabs(static_cast<double>(eval_segment(coeffs, i, t)) - std::sin(xm)));
        }
        EXPECT_LT(max_midpoint_error, 2e-3) << "a natural spline on these knots is ~1.1e-3; the linear version was 7.0e-2";
    }

    // A spline must pass through every knot it was given, at both ends of each segment.
    TEST(CubicSplineTest, PassesThroughEveryKnot) {
        std::vector<float> x = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
        std::vector<float> y = {0.0f, 0.25f, 1.0f, 2.25f, 4.0f};  // y = x^2

        auto result = cuda::numeric::cubic_spline_interpolation(x.data(), y.data(), x.size());
        std::vector<float> coeffs(result.coeffs.size());
        result.coeffs.copy_to(coeffs.data(), coeffs.size());

        for (size_t i = 0; i + 1 < x.size(); ++i) {
            const float h = x[i + 1] - x[i];
            EXPECT_NEAR(eval_segment(coeffs, i, 0.0f), y[i], 1e-4f) << "start of segment " << i;
            EXPECT_NEAR(eval_segment(coeffs, i, h), y[i + 1], 1e-4f) << "end of segment " << i;
        }
    }

    // The natural end condition: second derivative zero at the first and last knot.
    // S''(t) = 6*a*t + 2*b, so at t = 0 it is 2*b.
    TEST(CubicSplineTest, HasZeroSecondDerivativeAtBothEnds) {
        std::vector<float> x = {0.0f, 1.0f, 2.0f, 3.0f};
        std::vector<float> y = {0.0f, 1.0f, 8.0f, 27.0f};

        auto result = cuda::numeric::cubic_spline_interpolation(x.data(), y.data(), x.size());
        std::vector<float> coeffs(result.coeffs.size());
        result.coeffs.copy_to(coeffs.data(), coeffs.size());

        EXPECT_NEAR(coeffs[1], 0.0f, 1e-5f) << "S'' at the first knot must be 0 for a natural spline";

        const size_t last = x.size() - 2;
        const float h_last = x[last + 1] - x[last];
        const float second_deriv_end = 6.0f * coeffs[last * 4 + 0] * h_last + 2.0f * coeffs[last * 4 + 1];
        EXPECT_NEAR(second_deriv_end, 0.0f, 1e-4f) << "S'' at the last knot must be 0 for a natural spline";
    }

    // A natural cubic spline does NOT reproduce an arbitrary cubic exactly, because it forces
    // the second derivative to zero at the ends and x^3 has curvature 6x there. The error is
    // therefore a property of the end condition, not of this implementation: measured against
    // an independent reference solve, a natural spline on x^3 over [0, 2] with these knots
    // reaches 1.37e-01 at the far segment. This case pins that the implementation agrees with
    // that reference -- a much larger disagreement would mean the system is being solved wrong.
    TEST(CubicSplineTest, MatchesTheNaturalSplineReferenceOnACubic) {
        std::vector<float> x = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
        std::vector<float> y;
        for (float v : x) {
            y.push_back(v * v * v);
        }

        auto result = cuda::numeric::cubic_spline_interpolation(x.data(), y.data(), x.size());
        std::vector<float> coeffs(result.coeffs.size());
        result.coeffs.copy_to(coeffs.data(), coeffs.size());

        // The reference values below come from an independent Thomas-algorithm solve of the
        // same natural-spline system, not from this implementation.
        const float expected[4][3] = {
            {-0.000140f, 0.012277f, 0.049805f},
            {0.249581f, 0.431920f, 0.679548f},
            {1.404157f, 1.916295f, 2.564035f},
            {4.364258f, 5.496652f, 6.724470f},
        };
        for (size_t i = 0; i + 1 < x.size(); ++i) {
            const float h = x[i + 1] - x[i];
            for (int k = 1; k < 4; ++k) {
                const float t = h * static_cast<float>(k) / 4.0f;
                EXPECT_NEAR(eval_segment(coeffs, i, t), expected[i][k - 1], 1e-4f) << "segment " << i << " at t=" << t;
            }
        }
    }

    // A repeated knot makes a segment of zero width, so the system has no solution. It must be
    // reported rather than divided by zero.
    TEST(CubicSplineTest, RejectsNonIncreasingKnots) {
        std::vector<float> x_same = {0.0f, 1.0f, 1.0f, 2.0f};
        std::vector<float> y = {0.0f, 1.0f, 2.0f, 3.0f};
        EXPECT_THROW(cuda::numeric::cubic_spline_interpolation(x_same.data(), y.data(), 4), std::invalid_argument);

        std::vector<float> x_down = {0.0f, 1.0f, 0.5f};
        std::vector<float> y_down = {0.0f, 1.0f, 2.0f};
        EXPECT_THROW(cuda::numeric::cubic_spline_interpolation(x_down.data(), y_down.data(), 3), std::invalid_argument);
    }

    // ---------------------------------------------------------------------------
    // bisection could not find a root that sits on an endpoint of the bracket.
    // ---------------------------------------------------------------------------

    // Regression: f(x) = x over [0, 1] has its root at a = 0. The midpoint loop only tests
    // interior points, so the interval narrowed towards 0 from above and the routine returned
    // 1.0 -- the wrong end of the bracket -- while reporting converged = true.
    TEST(BisectionTest, FindsARootThatSitsOnTheLowerEndpoint) {
        auto result = cuda::numeric::bisection([](float x) { return x; }, 0.0f, 1.0f);

        EXPECT_TRUE(result.converged);
        EXPECT_NEAR(result.root, 0.0f, 1e-6f) << "the root is at the lower endpoint, not the upper one";
        EXPECT_NEAR(result.residual, 0.0f, 1e-6f);
    }

    TEST(BisectionTest, FindsARootThatSitsOnTheUpperEndpoint) {
        auto result = cuda::numeric::bisection([](float x) { return x - 1.0f; }, 0.0f, 1.0f);

        EXPECT_TRUE(result.converged);
        EXPECT_NEAR(result.root, 1.0f, 1e-6f);
        EXPECT_NEAR(result.residual, 0.0f, 1e-6f);
    }

    // An interval with no sign change must be reported, and the residual must not carry a
    // stale value from a previous evaluation.
    TEST(BisectionTest, ReportsAnIntervalThatDoesNotBracketARoot) {
        auto result = cuda::numeric::bisection([](float x) { return x * x + 1.0f; }, -1.0f, 1.0f);

        EXPECT_FALSE(result.converged);
        EXPECT_EQ(result.iterations, 0u);
        EXPECT_TRUE(std::isinf(result.residual)) << "no evaluation was performed, so the residual is not a number";
    }

    // newton_raphson used to report |dx| (the last Newton step) in `residual`, while bisection
    // reported |f(root)|. The two agree to first order near a simple root, so a value like
    // 4.2e-08 looked reasonable either way; this case requires the documented meaning by
    // comparing the field against an independent evaluation of the function at the answer.
    TEST(NewtonRaphsonTest, ResidualIsTheFunctionValueAtTheRoot) {
        auto result = cuda::numeric::newton_raphson(square_minus_two, square_minus_two_deriv, 1.0f);

        ASSERT_TRUE(result.converged);
        EXPECT_NEAR(result.root, std::sqrt(2.0f), 1e-5f);

        const float evaluated = square_minus_two(result.root);
        EXPECT_NEAR(result.residual, std::abs(evaluated), 1e-6f) << "residual must be |f(root)|, not the step size that produced it";
    }

    // A vanishing derivative stops the iteration; the reported residual is still the truth about
    // the point it stopped at, not a stale or invented number.
    TEST(NewtonRaphsonTest, ReportsTheResidualWhenTheDerivativeVanishes) {
        auto result = cuda::numeric::newton_raphson([](float) { return 1.0f; }, [](float) { return 0.0f; }, 0.5f);

        EXPECT_FALSE(result.converged);
        EXPECT_NEAR(result.residual, 1.0f, 1e-6f) << "f is 1 everywhere, so that is the residual wherever it stopped";
    }

    // The ordinary interior case must still work.
    TEST(BisectionTest, FindsAnInteriorRoot) {
        auto result = cuda::numeric::bisection(square_minus_two, 0.0f, 2.0f);
        EXPECT_TRUE(result.converged);
        EXPECT_NEAR(result.root, std::sqrt(2.0f), 1e-5f);
    }

}  // namespace cuda::numeric::test
