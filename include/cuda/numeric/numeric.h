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
InterpolationResult linear_interpolation(const float* x, const float* y, size_t n);
InterpolationResult cubic_spline_interpolation(const float* x, const float* y, size_t n);

}  // namespace cuda::numeric
