#include "property_test.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include <iostream>

using namespace nova::property;

bool SortProducesSortedOutput(RandomGenerator& gen) {
    auto n = gen.Uniform<size_t>(1, 1000);
    std::vector<double> data(n);

    for (size_t i = 0; i < n; ++i) {
        data[i] = gen.UniformFloat<double>(-1000.0, 1000.0);
    }

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    for (size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i] < sorted[i - 1]) {
            return false;
        }
    }
    return true;
}

bool ReduceIsAssociative(RandomGenerator& gen) {
    auto n = gen.Uniform<size_t>(2, 100);
    std::vector<double> data(n);

    for (size_t i = 0; i < n; ++i) {
        data[i] = gen.UniformFloat<double>(-100.0, 100.0);
    }

    // Left-to-right reduction
    double left_to_right = data[0];
    for (size_t i = 1; i < n; ++i) {
        left_to_right = left_to_right + data[i];
    }

    // Right-to-left reduction
    double right_to_left = data[n - 1];
    for (size_t i = n - 2; i != static_cast<size_t>(-1); --i) {
        right_to_left = data[i] + right_to_left;
    }

    // Associative: left_to_right == right_to_left
    return std::abs(left_to_right - right_to_left) < 1e-10;
}

bool InclusiveScanPrefixSum(RandomGenerator& gen) {
    auto n = gen.Uniform<size_t>(1, 100);
    std::vector<double> data(n), result(n);

    for (size_t i = 0; i < n; ++i) {
        data[i] = gen.UniformFloat<double>(-10.0, 10.0);
    }

    double expected = 0;
    for (size_t i = 0; i < n; ++i) {
        expected += data[i];
        result[i] = expected;
    }

    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return std::abs(result[n - 1] - sum) < 1e-10;
}

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "  Algorithmic Correctness Property Tests\n";
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

    run_test("Sort Produces Sorted Output", SortProducesSortedOutput);
    run_test("Reduce Is Associative", ReduceIsAssociative);
    run_test("Scan Produces Correct Prefix Sum", InclusiveScanPrefixSum);

    std::cout << "========================================\n";
    std::cout << "Results: " << passed_tests << "/" << total_tests << " passed\n";
    std::cout << "========================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
