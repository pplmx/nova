/**
 * @file quantize_benchmark_test.cpp
 * @brief Contracts for the quantisation precision comparison
 *
 * include/cuda/quantize/benchmark.hpp declares run_precision_comparison() and
 * src/cuda/quantize/benchmark.cpp implements every other benchmark entry point
 * but not that one. Any caller failed to link with:
 *
 *   undefined reference to
 *   nova::quantize::benchmark::run_precision_comparison(float const*, ...,
 *                                                       int, int, int)
 *
 * It is implemented now: it quantises the inputs to each precision's grid, runs
 * the product, and reports the error against the supplied float reference. The
 * tests below pin that contract.
 */

#include <cuda/quantize/benchmark.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace nova::quantize::benchmark::test {

    namespace {

        /// Reference float matmul for the small shapes these tests use.
        std::vector<float> reference_matmul(const std::vector<float>& a, const std::vector<float>& b, int m, int k, int n) {
            std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; ++p) {
                        sum += a[static_cast<size_t>(i) * k + p] * b[static_cast<size_t>(p) * n + j];
                    }
                    c[static_cast<size_t>(i) * n + j] = sum;
                }
            }
            return c;
        }

    }  // namespace

    // Regression: linking this function failed before it was implemented.
    TEST(RunPrecisionComparisonTest, IsLinkableAndReturnsOneResultPerPrecision) {
        const int m = 2, k = 2, n = 2;
        const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> b = {1.0f, 0.0f, 0.0f, 1.0f};
        const std::vector<float> c_ref = reference_matmul(a, b, m, k, n);

        const auto results = run_precision_comparison(a.data(), b.data(), c_ref.data(), m, k, n);

        ASSERT_EQ(results.size(), 2u) << "one result per supported precision";
        // Names are descriptive rather than exact, so identify the case by its
        // precision token appearing anywhere in the label.
        EXPECT_NE(results[0].name.find("fp8"), std::string::npos) << results[0].name;
        EXPECT_NE(results[1].name.find("int8"), std::string::npos) << results[1].name;
        EXPECT_NE(results[0].name, results[1].name) << "the cases must be distinguishable";
    }

    TEST(RunPrecisionComparisonTest, EveryResultDescribesTheShape) {
        const int m = 3, k = 4, n = 2;
        std::vector<float> a(static_cast<size_t>(m) * k, 1.0f);
        std::vector<float> b(static_cast<size_t>(k) * n, 1.0f);
        const std::vector<float> c_ref = reference_matmul(a, b, m, k, n);

        const auto results = run_precision_comparison(a.data(), b.data(), c_ref.data(), m, k, n);
        ASSERT_FALSE(results.empty());

        for (const auto& result : results) {
            EXPECT_EQ(result.batch_size, m * n) << "batch_size reports the output element count";
            EXPECT_GE(result.num_samples, 1);
            EXPECT_FALSE(result.name.empty());
        }
    }

    TEST(RunPrecisionComparisonTest, ValuesExactInBothGridsHaveNoError) {
        // 1.0 and 0/1 are exactly representable in fp8 E4M3 and int8, so an
        // identity multiplication must reproduce the reference exactly.
        const int m = 2, k = 2, n = 2;
        const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> b = {1.0f, 0.0f, 0.0f, 1.0f};
        const std::vector<float> c_ref = reference_matmul(a, b, m, k, n);

        const auto results = run_precision_comparison(a.data(), b.data(), c_ref.data(), m, k, n);
        ASSERT_EQ(results.size(), 2u);

        for (const auto& result : results) {
            EXPECT_NEAR(result.relative_error, 0.0f, 1e-5f) << result.name << " should reproduce exactly representable values";
        }
    }

    TEST(RunPrecisionComparisonTest, ErrorGrowsWhenValuesDoNotFitTheGrid) {
        // A large magnitude saturates fp8 E4M3 (max ~448), so the reported error
        // must be clearly non-zero.
        const int m = 1, k = 2, n = 1;
        const std::vector<float> a = {1000.0f, 1000.0f};
        const std::vector<float> b = {1.0f, 1.0f};
        const std::vector<float> c_ref = reference_matmul(a, b, m, k, n);

        const auto results = run_precision_comparison(a.data(), b.data(), c_ref.data(), m, k, n);
        ASSERT_EQ(results.size(), 2u);

        // results[0] is the fp8 case: 1000 saturates, so the product cannot match.
        EXPECT_GT(results[0].relative_error, 0.0f) << "values beyond fp8's range must show up as error";
    }

    TEST(RunPrecisionComparisonTest, RelativeErrorIsNeverNegative) {
        const int m = 2, k = 3, n = 2;
        std::vector<float> a = {0.5f, -1.5f, 2.25f, -3.0f, 4.0f, 0.125f};
        std::vector<float> b = {1.0f, -1.0f, 0.5f, 2.0f, -2.0f, 0.25f};
        const std::vector<float> c_ref = reference_matmul(a, b, m, k, n);

        const auto results = run_precision_comparison(a.data(), b.data(), c_ref.data(), m, k, n);
        ASSERT_FALSE(results.empty());
        for (const auto& result : results) {
            EXPECT_GE(result.relative_error, 0.0f);
            EXPECT_GE(result.accuracy_loss, 0.0f);
            EXPECT_TRUE(std::isfinite(result.relative_error));
        }
    }

    TEST(RunPrecisionComparisonTest, NullOrEmptyInputReturnsNoResults) {
        // A degenerate request must produce an empty vector rather than crash or
        // allocate a zero-length reference.
        EXPECT_TRUE(run_precision_comparison(nullptr, nullptr, nullptr, 0, 0, 0).empty());

        std::vector<float> a = {1.0f};
        EXPECT_TRUE(run_precision_comparison(a.data(), nullptr, a.data(), 1, 1, 1).empty());
        EXPECT_TRUE(run_precision_comparison(a.data(), a.data(), a.data(), 0, 1, 1).empty());
        EXPECT_TRUE(run_precision_comparison(a.data(), a.data(), a.data(), 1, 0, 1).empty());
        EXPECT_TRUE(run_precision_comparison(a.data(), a.data(), a.data(), 1, 1, 0).empty());
    }

    TEST(RunPrecisionComparisonTest, ResultsAreSerialisable) {
        // The header documents save_json/print as the reporting path; the fields the
        // comparison fills must be enough to produce a report row.
        const int m = 1, k = 1, n = 1;
        const std::vector<float> a = {1.0f};
        const std::vector<float> b = {1.0f};
        const std::vector<float> c_ref = {1.0f};

        const auto results = run_precision_comparison(a.data(), b.data(), c_ref.data(), m, k, n);
        ASSERT_FALSE(results.empty());

        std::ostringstream stream;
        results.front().save_json(stream);
        const std::string json = stream.str();

        EXPECT_NE(json.find("\"name\""), std::string::npos);
        EXPECT_NE(json.find("\"relative_error\""), std::string::npos);
    }

}  // namespace nova::quantize::benchmark::test
