#include "cuda/numeric/numeric.h"

#include <cmath>
#include <cstdlib>

#include "cuda/device/error.h"
#include "cuda/algo/reduce.h"

namespace cuda::numeric {

namespace {

__global__ void monte_carlo_kernel(float* results, size_t n, float a, float b, curandState* states) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState state = states[idx];
    float u = curand_uniform(&state);
    float x = a + u * (b - a);
    results[idx] = func(x);
    states[idx] = state;
}

__global__ void antithetic_variate_kernel(float* results, size_t n, float a, float b, curandState* states) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState state = states[idx];
    float u = curand_uniform(&state);

    float x1 = a + u * (b - a);
    float x2 = a + (1.0f - u) * (b - a);

    results[idx] = (func(x1) + func(x2)) * 0.5f;
    states[idx] = state;
}

__global__ void trap_kernel(float* results, size_t n, float h, float a, float (*func)(float)) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = a + idx * h;
    results[idx] = func(x);
}

__global__ void simpson_kernel(float* f0, float* f1, float* f2, size_t n, float h, float a, float (*func)(float)) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x0 = a + (2 * idx) * h;
    float x1 = x0 + h;
    float x2 = x0 + 2 * h;

    f0[idx] = func(x0);
    f1[idx] = func(x1);
    f2[idx] = func(x2);
}

float func_wrapper(float x) {
    return x * x;
}

curandGenerator_t get_curand_generator() {
    static curandGenerator_t gen = [] {
        curandGenerator_t g;
        curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(g, 1234ULL);
        return g;
    }();
    return gen;
}

}  // namespace

MonteCarloResult monte_carlo_integration(float (*func)(float), float a, float b, size_t samples, float tolerance) {
    MonteCarloResult result;
    result.samples = samples;
    result.converged = false;

    memory::Buffer<float> results(samples);
    memory::Buffer<float> d_func(samples);

    curandGenerator_t gen = get_curand_generator();
    float* h_samples = static_cast<float*>(malloc(samples * sizeof(float)));
    float* h_func = static_cast<float*>(malloc(samples * sizeof(float)));

    for (size_t i = 0; i < samples; ++i) {
        h_samples[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    CUDA_CHECK(cudaMemcpy(d_func.data(), h_samples, samples * sizeof(float), cudaMemcpyHostToDevice));

    float mean = cuda::algo::reduce_sum(d_func.data(), samples) / samples;
    float variance = 0.0f;

    for (size_t i = 0; i < samples; ++i) {
        float diff = h_func[i] - mean;
        variance += diff * diff;
    }
    variance /= samples;
    result.mean = mean * (b - a);
    result.variance = variance;
    result.std_error = std::sqrt(variance / samples);
    result.converged = result.std_error < tolerance;

    free(h_samples);
    free(h_func);

    return result;
}

IntegrationResult trapezoidal_integration(float (*func)(float), float a, float b, size_t n) {
    IntegrationResult result;
    float h = (b - a) / n;

    memory::Buffer<float> values(n + 1);
    float* h_values = static_cast<float*>(malloc((n + 1) * sizeof(float)));

    for (size_t i = 0; i <= n; ++i) {
        float x = a + i * h;
        h_values[i] = func(x);
    }

    CUDA_CHECK(cudaMemcpy(values.data(), h_values, (n + 1) * sizeof(float), cudaMemcpyHostToDevice));

    float sum = cuda::algo::reduce_sum(values.data(), n + 1);
    result.value = h * (sum - 0.5f * (h_values[0] + h_values[n]));
    result.intervals = n;
    result.error_estimate = std::abs(h * h * (b - a) / 12.0f);
    result.converged = result.error_estimate < 1e-5f;

    free(h_values);
    return result;
}

IntegrationResult simpson_integration(float (*func)(float), float a, float b, size_t n) {
    IntegrationResult result;
    float h = (b - a) / (2 * n);

    memory::Buffer<float> f0(n), f1(n), f2(n);
    float* h_f0 = static_cast<float*>(malloc(n * sizeof(float)));
    float* h_f1 = static_cast<float*>(malloc(n * sizeof(float)));
    float* h_f2 = static_cast<float*>(malloc(n * sizeof(float)));

    for (size_t i = 0; i < n; ++i) {
        float x0 = a + (2 * i) * h;
        float x1 = x0 + h;
        float x2 = x0 + 2 * h;
        h_f0[i] = func(x0);
        h_f1[i] = func(x1);
        h_f2[i] = func(x2);
    }

    CUDA_CHECK(cudaMemcpy(f0.data(), h_f0, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(f1.data(), h_f1, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(f2.data(), h_f2, n * sizeof(float), cudaMemcpyHostToDevice));

    float sum0 = cuda::algo::reduce_sum(f0.data(), n);
    float sum1 = cuda::algo::reduce_sum(f1.data(), n);
    float sum2 = cuda::algo::reduce_sum(f2.data(), n);

    result.value = h / 3.0f * (sum0 + 4.0f * sum1 + sum2);
    result.intervals = n;
    result.error_estimate = std::abs(h * h * h * h * (b - a) / 180.0f);
    result.converged = result.error_estimate < 1e-5f;

    free(h_f0);
    free(h_f1);
    free(h_f2);

    return result;
}

RootFindingResult bisection(float (*func)(float), float a, float b, float tolerance, size_t max_iter) {
    RootFindingResult result;

    float fa = func(a);
    float fb = func(b);

    if (fa * fb > 0) {
        result.converged = false;
        result.root = 0;
        result.iterations = 0;
        result.residual = INFINITY;
        return result;
    }

    for (result.iterations = 0; result.iterations < max_iter; ++result.iterations) {
        float c = (a + b) * 0.5f;
        float fc = func(c);

        result.residual = std::abs(fc);

        if (result.residual < tolerance || (b - a) * 0.5f < tolerance) {
            result.root = c;
            result.converged = true;
            return result;
        }

        if (fa * fc < 0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    result.root = (a + b) * 0.5f;
    result.converged = false;
    return result;
}

RootFindingResult newton_raphson(float (*func)(float), float (*deriv)(float), float x0, float tolerance, size_t max_iter) {
    RootFindingResult result;
    result.root = x0;

    for (result.iterations = 0; result.iterations < max_iter; ++result.iterations) {
        float fx = func(result.root);
        float dfx = deriv(result.root);

        if (std::abs(dfx) < 1e-12) {
            result.converged = false;
            result.residual = std::abs(fx);
            return result;
        }

        float dx = fx / dfx;
        result.root -= dx;
        result.residual = std::abs(dx);

        if (result.residual < tolerance) {
            result.converged = true;
            return result;
        }
    }

    result.converged = false;
    return result;
}

InterpolationResult linear_interpolation(const float* x, const float* y, size_t n) {
    InterpolationResult result;
    result.n = n;
    result.x = memory::Buffer<float>(n);
    result.y = memory::Buffer<float>(n);
    result.coeffs = memory::Buffer<float>((n - 1) * 2);

    CUDA_CHECK(cudaMemcpy(result.x.data(), x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.y.data(), y, n * sizeof(float), cudaMemcpyHostToDevice));

    float* h_coeffs = static_cast<float*>(malloc((n - 1) * 2 * sizeof(float)));
    for (size_t i = 0; i < n - 1; ++i) {
        h_coeffs[i * 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        h_coeffs[i * 2 + 1] = y[i] - h_coeffs[i * 2] * x[i];
    }

    CUDA_CHECK(cudaMemcpy(result.coeffs.data(), h_coeffs, (n - 1) * 2 * sizeof(float), cudaMemcpyHostToDevice));
    free(h_coeffs);

    return result;
}

InterpolationResult cubic_spline_interpolation(const float* x, const float* y, size_t n) {
    InterpolationResult result;
    result.n = n;
    result.x = memory::Buffer<float>(n);
    result.y = memory::Buffer<float>(n);
    result.coeffs = memory::Buffer<float>(n * 4);

    CUDA_CHECK(cudaMemcpy(result.x.data(), x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.y.data(), y, n * sizeof(float), cudaMemcpyHostToDevice));

    float* h_coeffs = static_cast<float*>(malloc(n * 4 * sizeof(float)));

    for (size_t i = 0; i < n - 1; ++i) {
        float h_i = x[i + 1] - x[i];
        h_coeffs[i * 4 + 0] = 0.0f;
        h_coeffs[i * 4 + 1] = 0.0f;
        h_coeffs[i * 4 + 2] = (y[i + 1] - y[i]) / h_i;
        h_coeffs[i * 4 + 3] = y[i];
    }
    for (size_t i = 0; i < 4; ++i) {
        h_coeffs[(n - 1) * 4 + i] = 0.0f;
    }

    CUDA_CHECK(cudaMemcpy(result.coeffs.data(), h_coeffs, n * 4 * sizeof(float), cudaMemcpyHostToDevice));
    free(h_coeffs);

    return result;
}

}  // namespace cuda::numeric
