#include "cuda/numeric/numeric.h"

#include <cmath>
#include <cstdlib>
#include <vector>

#include "cuda/device/error.h"
#include "cuda/algo/reduce.h"

namespace cuda::numeric {

    // Three device kernels used to live here: trap_kernel, simpson_kernel and
    // linear_interp_kernel. None of them was ever launched -- every entry point in this file
    // computes on the host and then uses cudaMemcpy plus a device reduction -- and the first
    // two could not have worked: both took `float (*func)(float)`, a HOST function pointer,
    // and called it from device code. That compiles and launches, then fails at run time with
    // cudaErrorInvalidProgramCounter (718), measured. Evaluating the integrand on the host and
    // reducing on the device is the correct shape, and is what the live code does.

    MonteCarloResult monte_carlo_integration(float (*func)(float), float a, float b, size_t samples, float tolerance) {
        MonteCarloResult result;
        result.samples = samples;
        result.converged = false;
        result.mean = 0.0f;
        result.variance = 0.0f;
        result.std_error = 0.0f;

        if (samples == 0) {
            return result;
        }

        // The old implementation filled h_func with uniform rand()/RAND_MAX in
        // [0,1) and never called func, so every integrand "integrated" to
        // ~(b-a)/2 (mean of uniforms times the interval width). Draw uniform
        // samples in [a,b], evaluate the integrand on them, and estimate
        // mean(func) * (b-a).
        memory::Buffer<float> d_func(samples);
        std::vector<float> h_func(samples);

        for (size_t i = 0; i < samples; ++i) {
            const float u = a + (b - a) * (static_cast<float>(rand()) / RAND_MAX);
            h_func[i] = func(u);
        }

        d_func.copy_from(h_func.data(), samples);

        const float sum = cuda::algo::reduce_sum(d_func.data(), samples);
        const float mean = sum / static_cast<float>(samples);
        result.mean = mean * (b - a);

        float variance = 0.0f;
        for (size_t i = 0; i < samples; ++i) {
            const float diff = h_func[i] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(samples);
        result.variance = variance;
        result.std_error = std::sqrt(variance / static_cast<float>(samples));
        result.converged = result.std_error < tolerance;

        return result;
    }

    IntegrationResult trapezoidal_integration(float (*func)(float), float a, float b, size_t n) {
        IntegrationResult result;

        // h = (b - a) / n divides by the interval count, so n == 0 produced inf and
        // propagated NaN through every field. An empty integration range is zero.
        if (n == 0) {
            result.value = 0.0f;
            result.intervals = 0;
            result.error_estimate = 0.0f;
            result.converged = true;
            return result;
        }

        const float h = (b - a) / static_cast<float>(n);

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

        // Same division by the interval count as the trapezoidal rule.
        if (n == 0) {
            result.value = 0.0f;
            result.intervals = 0;
            result.error_estimate = 0.0f;
            result.converged = true;
            return result;
        }

        const float h = (b - a) / (2.0f * static_cast<float>(n));

        memory::Buffer<float> f0(n), f1(n), f2(n);
        // std::vector rather than three bare malloc()s: those were never checked for null, and the
        // three CUDA_CHECK copies below can throw, which left all three arrays with no owner. This
        // file includes no Thrust or libcu++ header, so unqualified std:: is ::std:: here (see
        // docs/include-order.md).
        std::vector<float> h_f0(n), h_f1(n), h_f2(n);

        for (size_t i = 0; i < n; ++i) {
            float x0 = a + (2 * i) * h;
            float x1 = x0 + h;
            float x2 = x0 + 2 * h;
            h_f0[i] = func(x0);
            h_f1[i] = func(x1);
            h_f2[i] = func(x2);
        }

        CUDA_CHECK(cudaMemcpy(f0.data(), h_f0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(f1.data(), h_f1.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(f2.data(), h_f2.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        float sum0 = cuda::algo::reduce_sum(f0.data(), n);
        float sum1 = cuda::algo::reduce_sum(f1.data(), n);
        float sum2 = cuda::algo::reduce_sum(f2.data(), n);

        result.value = h / 3.0f * (sum0 + 4.0f * sum1 + sum2);
        result.intervals = n;
        result.error_estimate = std::abs(h * h * h * h * (b - a) / 180.0f);
        result.converged = result.error_estimate < 1e-5f;

        return result;
    }

    RootFindingResult bisection(float (*func)(float), float a, float b, float tolerance, size_t max_iter) {
        RootFindingResult result;
        result.root = 0;
        result.iterations = 0;
        result.residual = INFINITY;
        result.converged = false;

        float fa = func(a);
        float fb = func(b);

        // A root AT an endpoint is the answer for that endpoint, and the midpoint loop
        // cannot find it: it only ever tests interior points, so for f(x) = x over [0, 1]
        // the interval narrows towards 0 from above and the routine returned 1.0 -- the
        // wrong end -- while reporting converged = true. Measured before this guard:
        // root = 1.000e+00.
        //
        // The bracket test below also uses `fa * fb > 0` to reject, which an endpoint root
        // satisfies as zero, so this has to come first.
        if (fa == 0.0f) {
            result.root = a;
            result.residual = 0.0f;
            result.converged = true;
            return result;
        }
        if (fb == 0.0f) {
            result.root = b;
            result.residual = 0.0f;
            result.converged = true;
            return result;
        }

        if (fa * fb > 0) {
            // No sign change, so the interval does not bracket a root. residual stays
            // INFINITY rather than being left at whatever the last evaluation produced.
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
        result.residual = INFINITY;
        result.converged = false;

        // `residual` means |f(x)| at the reported root, matching bisection() next door.
        // It used to hold |dx| -- the size of the last Newton step, not how far the function
        // is from zero there. The two agree to first order near a simple root, which is why
        // it went unnoticed, but they diverge where it matters: at a flat root |f| is much
        // smaller than |dx| suggests, and for a scale-free statement about the answer the
        // caller needs |f|. The step size is still what the loop tests for convergence,
        // because that is the quantity that shrinks quadratically.
        for (result.iterations = 0; result.iterations < max_iter; ++result.iterations) {
            float fx = func(result.root);
            float dfx = deriv(result.root);

            if (std::abs(dfx) < 1e-12) {
                // A vanishing derivative means the step is not defined. Report the residual
                // that is actually true at the point we stopped at.
                result.converged = false;
                result.residual = std::abs(fx);
                return result;
            }

            float dx = fx / dfx;

            if (std::abs(dx) < tolerance) {
                result.root -= dx;
                result.residual = std::abs(func(result.root));
                result.converged = true;
                return result;
            }

            result.root -= dx;
        }

        result.residual = std::abs(func(result.root));
        result.converged = false;
        return result;
    }

    InterpolationResult linear_interpolation(const float* x, const float* y, size_t n) {
        InterpolationResult result;

        // The coefficient count is (n - 1) * 2, which underflows to SIZE_MAX for
        // n == 0 and asked the allocator for 32 exabytes ("out of memory"). Fewer
        // than two points describe no segment at all.
        if (n < 2) {
            result.n = n;
            result.x = memory::Buffer<float>(n);
            result.y = memory::Buffer<float>(n);
            result.coeffs = memory::Buffer<float>(0);
            if (n > 0) {
                CUDA_CHECK(cudaMemcpy(result.x.data(), x, n * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(result.y.data(), y, n * sizeof(float), cudaMemcpyHostToDevice));
            }
            return result;
        }

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

        // Fewer than two points describe no segment. Without this the loop below runs
        // `i < n - 1` with n == 0, i.e. i < SIZE_MAX, reading x[1..] past the end of the
        // caller's arrays, and the closing write targets h_coeffs[4 * SIZE_MAX + i] on a
        // malloc(0) block. Measured before the guard: the process dumped core.
        //
        // linear_interpolation next door already returns an empty coefficient buffer for
        // n < 2; this mirrors it.
        if (n < 2) {
            if (n > 0) {
                CUDA_CHECK(cudaMemcpy(result.x.data(), x, n * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(result.y.data(), y, n * sizeof(float), cudaMemcpyHostToDevice));
            }
            return result;
        }

        CUDA_CHECK(cudaMemcpy(result.x.data(), x, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(result.y.data(), y, n * sizeof(float), cudaMemcpyHostToDevice));

        // A NATURAL cubic spline: continuous second derivative with S'' = 0 at both ends.
        //
        // What was here before was not a spline at all. It wrote the linear form and left
        // the cubic and quadratic slots at zero, so every segment was a straight line:
        // measured on sin over [0, pi] with five knots, every segment had coefficient
        // slots 0 and 1 exactly 0, and the midpoint error was 7.033e-02. A natural spline
        // on the same knots is 1.064e-03 -- 66x better -- and that is what the name
        // promises.
        //
        // Solve for the second derivatives M at each knot from the standard tridiagonal
        // system, then write each segment on the local variable t in [0, h_i]:
        //
        //     S_i(t) = d + c*t + b*t^2 + a*t^3
        //     a = (M[i+1] - M[i]) / (6 h_i)
        //     b = M[i] / 2
        //     c = (y[i+1] - y[i]) / h_i - h_i (2 M[i] + M[i+1]) / 6
        //     d = y[i]
        //
        // The coefficient layout is unchanged (four floats per segment), so a caller that
        // already evaluates a*t^3 + b*t^2 + c*t + d keeps working -- it just gets a cubic
        // where it used to get a line.
        std::vector<float> h_i(n - 1);
        bool strictly_increasing = true;
        for (size_t i = 0; i < n - 1; ++i) {
            h_i[i] = x[i + 1] - x[i];
            if (!(h_i[i] > 0.0f)) {
                strictly_increasing = false;
            }
        }
        if (!strictly_increasing) {
            throw std::invalid_argument(
                "cubic_spline_interpolation: the x values must be strictly increasing; a repeated or decreasing knot makes the segment width "
                "zero or negative and the system is not solvable");
        }

        std::vector<float> M(n, 0.0f);
        const size_t interior = n - 2;  // number of unknowns, M[0] and M[n-1] are 0
        if (interior > 0) {
            // Tridiagonal system, solved with the Thomas algorithm in place.
            std::vector<float> diag(interior), upper(interior), lower(interior), rhs(interior);
            for (size_t k = 0; k < interior; ++k) {
                const size_t i = k + 1;  // knot index, 1..n-2
                lower[k] = h_i[i - 1];
                diag[k] = 2.0f * (h_i[i - 1] + h_i[i]);
                upper[k] = h_i[i];
                rhs[k] = 6.0f * ((y[i + 1] - y[i]) / h_i[i] - (y[i] - y[i - 1]) / h_i[i - 1]);
            }
            for (size_t k = 1; k < interior; ++k) {
                const float w = lower[k] / diag[k - 1];
                diag[k] -= w * upper[k - 1];
                rhs[k] -= w * rhs[k - 1];
            }
            std::vector<float> solution(interior, 0.0f);
            solution[interior - 1] = rhs[interior - 1] / diag[interior - 1];
            for (size_t k = interior - 1; k-- > 0;) {
                solution[k] = (rhs[k] - upper[k] * solution[k + 1]) / diag[k];
            }
            for (size_t k = 0; k < interior; ++k) {
                M[k + 1] = solution[k];
            }
        }

        float* h_coeffs = static_cast<float*>(malloc(n * 4 * sizeof(float)));
        for (size_t i = 0; i < n - 1; ++i) {
            h_coeffs[i * 4 + 0] = (M[i + 1] - M[i]) / (6.0f * h_i[i]);
            h_coeffs[i * 4 + 1] = M[i] / 2.0f;
            h_coeffs[i * 4 + 2] = (y[i + 1] - y[i]) / h_i[i] - h_i[i] * (2.0f * M[i] + M[i + 1]) / 6.0f;
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
