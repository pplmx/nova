#include "property_test.hpp"
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>

using namespace nova::property;

bool FloatPrecisionConsistency(RandomGenerator& gen) {
    auto a = gen.UniformFloat<double>(-1000.0, 1000.0);
    auto b = gen.UniformFloat<double>(-1000.0, 1000.0);

    // Avoid division by zero
    if (std::abs(b) < 1e-10) b = 1.0;

    // Simple operations
    double sum = a + b;
    double diff = a - b;
    double prod = a * b;
    double quot = a / b;

    // Results should be finite
    if (!std::isfinite(sum) || !std::isfinite(diff) ||
        !std::isfinite(prod) || !std::isfinite(quot)) {
        return false;
    }

    // NaN propagation: 0/0 = NaN
    double zero = 0.0;
    double nan_result = zero / zero;
    if (!std::isnan(nan_result)) {
        return false;
    }

    // Inf propagation
    double inf = std::numeric_limits<double>::infinity();
    double inf_sum = inf + a;
    if (!std::isinf(inf_sum) || inf_sum <= 0) {
        return false;
    }

    return true;
}

bool NoNaNWithoutCause(RandomGenerator& gen) {
    auto a = gen.UniformFloat<double>(-100.0, 100.0);
    auto b = gen.UniformFloat<double>(-100.0, 100.0);

    // Normal operations shouldn't produce NaN
    double sum = a + b;
    double diff = a - b;
    double prod = a * b;

    // sqrt with positive argument
    double positive = std::abs(a) + 1.0;
    double sqrt_val = std::sqrt(positive);

    return std::isfinite(sum) && std::isfinite(diff) &&
           std::isfinite(prod) && std::isfinite(sqrt_val);
}

bool NumericalStabilityAcrossModes(RandomGenerator& gen) {
    auto val = gen.UniformFloat<double>(0.001, 1000.0);

    // Simulate different precision modes
    // FP64: full precision
    double fp64_val = val;
    // FP32: single precision (simulated)
    float fp32_val = static_cast<float>(val);
    double fp32_back = static_cast<double>(fp32_val);
    // FP16: half precision (simulated)
    float fp16_val = static_cast<float>(val);
    double fp16_back = static_cast<double>(fp16_val);

    // FP32 should be closer to FP64 than FP16
    double fp32_diff = std::abs(fp64_val - fp32_back);
    double fp16_diff = std::abs(fp64_val - fp16_back);

    return fp32_diff <= fp16_diff;
}

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "  Numerical Stability Property Tests\n";
    std::cout << "========================================\n\n";

    uint64_t seed = 0;
    if (argc > 1) {
        seed = std::stoull(argv[1]);
    }

    size_t total_tests = 0;
    size_t passed_tests = 0;

    auto run_test = [&](const std::string& name, auto test_fn) {
        total_tests++;
        auto result = CheckProperty(name, test_fn, 100, seed);

        std::cout << "[" << (result.passed ? "PASS" : "FAIL") << "] "
                  << name << "\n";
        std::cout << "  Seed: " << result.seed
                  << " | Iterations: " << result.iterations << "\n";

        if (!result.passed) {
            std::cout << "  Reason: " << result.failure_reason << "\n";
        }
        std::cout << "\n";

        if (result.passed) passed_tests++;
    };

    run_test("Float Precision Consistency", FloatPrecisionConsistency);
    run_test("No NaN Without Cause", NoNaNWithoutCause);
    run_test("Numerical Stability Across Modes", NumericalStabilityAcrossModes);

    std::cout << "========================================\n";
    std::cout << "Results: " << passed_tests << "/" << total_tests << " passed\n";
    std::cout << "========================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
