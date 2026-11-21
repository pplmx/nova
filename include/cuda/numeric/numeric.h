#pragma once

/**
 * @file numeric.h
 * @brief GPU numerical methods: Monte Carlo, integration, root finding, interpolation
 * @author Nova CUDA Library
 * @version 2.3
 */

#include <cuda_runtime.h>
#include <curand.h>
#include <cstddef>

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::numeric {

    struct MonteCarloResult {
        float mean;
        float variance;
        float std_error;
        size_t samples;
        bool converged;
    };

    struct IntegrationResult {
        float value;
        float error_estimate;
        size_t intervals;
        bool converged;
    };

    struct RootFindingResult {
        float root;
        size_t iterations;
        /// |f(root)| at the reported root, for both root finders. Earlier newton_raphson
        /// reported the size of its last Newton step instead, which agrees near a simple
        /// root and diverges elsewhere.
        float residual;
        bool converged;
    };

    struct InterpolationResult {
        memory::Buffer<float> x;
        memory::Buffer<float> y;
        memory::Buffer<float> coeffs;
        size_t n;
    };

    MonteCarloResult monte_carlo_integration(float (*func)(float x), float a, float b, size_t samples, float tolerance = 1e-5f);
    IntegrationResult trapezoidal_integration(float (*func)(float x), float a, float b, size_t n);
    IntegrationResult simpson_integration(float (*func)(float x), float a, float b, size_t n);
    RootFindingResult bisection(float (*func)(float x), float a, float b, float tolerance = 1e-6f, size_t max_iter = 100);
    RootFindingResult newton_raphson(float (*func)(float x), float (*deriv)(float x), float x0, float tolerance = 1e-6f, size_t max_iter = 100);
    /**
     * @brief Piecewise-linear interpolation through the given samples
     *
     * coeffs holds two floats per segment (slope, intercept) on the local variable
     * t = xv - x[i], so segment i is `coeffs[2i] * t + coeffs[2i+1]`.
     */
    InterpolationResult linear_interpolation(const float* x, const float* y, size_t n);

    /**
     * @brief Natural cubic spline through the given samples
     *
     * Continuously differentiable with S'' = 0 at the first and last knot (the "natural" end
     * condition). coeffs holds four floats per segment on the local variable
     * t = xv - x[i], so segment i is
     *
     *     coeffs[4i+0] * t^3 + coeffs[4i+1] * t^2 + coeffs[4i+2] * t + coeffs[4i+3]
     *
     * This layout is unchanged, but what fills it is not: the previous implementation put
     * the linear form in the last two slots and left the cubic and quadratic at zero, so
     * every segment was a straight line. Measured on sin over [0, pi] with five knots, the
     * midpoint error went from 7.033e-02 to 1.064e-03 with the real spline.
     *
     * @param x Knot positions; must be strictly increasing. A repeated or decreasing knot is
     *        rejected with std::invalid_argument rather than solved for.
     * @param y Values at the knots
     * @param n Number of knots; fewer than two produces an empty coefficient buffer
     */
    InterpolationResult cubic_spline_interpolation(const float* x, const float* y, size_t n);

}  // namespace cuda::numeric
