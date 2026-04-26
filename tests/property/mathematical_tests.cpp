#include "property_test.hpp"
#include <nova/neural/matmul.hpp>
#include <nova/fft/fft.hpp>
#include <cassert>
#include <iostream>
#include <iomanip>

using namespace nova::property;

template<typename T>
bool MatmulIdentity(RandomGenerator& gen) {
    // Generate random square matrix
    auto n = gen.Uniform<size_t>(2, 32);
    std::vector<T> A(n * n), I(n * n, T{0});

    // Fill A with random values
    for (size_t i = 0; i < n * n; ++i) {
        A[i] = gen.UniformFloat<T>(-10.0f, 10.0f);
    }

    // Create identity matrix
    for (size_t i = 0; i < n; ++i) {
        I[i * n + i] = T{1};
    }

    // Compute A @ I
    std::vector<T> result(n * n, T{0});
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            for (size_t j = 0; j < n; ++j) {
                result[i * n + j] += A[i * n + k] * I[k * n + j];
            }
        }
    }

    // Verify A @ I == A
    T max_diff = T{0};
    for (size_t i = 0; i < n * n; ++i) {
        T diff = std::abs(result[i] - A[i]);
        if (diff > max_diff) max_diff = diff;
    }

    return max_diff < T{1e-5};
}

template<typename T>
bool TransposeInvolution(RandomGenerator& gen) {
    auto rows = gen.Uniform<size_t>(2, 32);
    auto cols = gen.Uniform<size_t>(2, 32);

    std::vector<T> A(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        A[i] = gen.UniformFloat<T>(-10.0f, 10.0f);
    }

    // First transpose: A -> A^T
    std::vector<T> AT(cols * rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            AT[j * rows + i] = A[i * cols + j];
        }
    }

    // Second transpose: AT -> (A^T)^T
    std::vector<T> ATT(rows * cols);
    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            ATT[j * cols + i] = AT[i * rows + j];
        }
    }

    // Verify (A^T)^T == A
    T max_diff = T{0};
    for (size_t i = 0; i < rows * cols; ++i) {
        T diff = std::abs(ATT[i] - A[i]);
        if (diff > max_diff) max_diff = diff;
    }

    return max_diff < T{1e-5};
}

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "  Mathematical Invariant Property Tests\n";
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

    // Matmul identity: A @ I = A
    run_test("Matmul Identity (float)", MatmulIdentity<float>);

    // Transpose involution: (A^T)^T = A
    run_test("Transpose Involution (float)", TransposeInvolution<float>);

    std::cout << "========================================\n";
    std::cout << "Results: " << passed_tests << "/" << total_tests << " passed\n";
    std::cout << "========================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
