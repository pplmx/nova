/**
 * @file cache_analyzer_honesty_test.cpp
 * @brief The cache analyzer must not report hit rates it did not measure
 *
 * src/cuda/performance/bandwidth/cache_analyzer.cpp has no CUPTI integration: nothing in
 * the file links against it or calls into it. Its availability probe nonetheless returned
 * true unconditionally, so analyze() returned
 *
 *     l1_hit_rate = 0.85, l2_hit_rate = 0.95, texture_hit_rate = 0.99
 *
 * — literals, not measurements — and analyze_kernel() picked its numbers by
 * substring-matching the kernel name against "shared" and "global". A kernel called
 * "shared_matmul" reported 0.95/0.98, "global_scan" reported 0.70/0.85, and anything else
 * reported 0.85/0.92. All of them looked like data.
 *
 * The existing CacheAnalyzerTest cases could not catch this:
 *
 *     EXPECT_TRUE(available || !available);          // tautology
 *     if (metrics.available) { ...check range... }   // skipped when not available
 *
 * The analyzer now reports available = false with an error message. These cases pin that
 * so a real implementation has to change them deliberately.
 */

#include <cuda/performance/bandwidth/cache_analyzer.h>

#include <gtest/gtest.h>

#include <string>

namespace cuda::performance::bandwidth::test {

    // Regression: this reported all cache counters as available while measuring nothing.
    TEST(CacheAnalyzerHonestyTest, AvailabilityIsNotClaimedWithoutCUPTI) {
        CacheAnalyzer analyzer;

        EXPECT_FALSE(analyzer.is_available()) << "no CUPTI code exists in this build, so cache counters are not available";
        EXPECT_FALSE(analyzer.get_error_message().empty()) << "unavailability must come with a reason a caller can act on";
    }

    // Regression: analyze() returned 0.85 / 0.95 / 0.99 as if measured.
    TEST(CacheAnalyzerHonestyTest, AnalyzeDoesNotInventHitRates) {
        CacheAnalyzer analyzer;

        const auto metrics = analyzer.analyze(0);

        EXPECT_FALSE(metrics.available);
        EXPECT_FALSE(metrics.error_message.empty());
    }

    // Regression: analyze_kernel() picked different fake numbers from the kernel name.
    TEST(CacheAnalyzerHonestyTest, KernelNameDoesNotChangeInventedNumbers) {
        CacheAnalyzer analyzer;

        const auto shared = analyzer.analyze_kernel("shared_matmul", 0);
        const auto global = analyzer.analyze_kernel("global_scan", 0);
        const auto plain = analyzer.analyze_kernel("matmul_kernel", 0);

        for (const auto& metrics : {shared, global, plain}) {
            EXPECT_FALSE(metrics.available) << "none of these kernels was profiled";
            EXPECT_FALSE(metrics.error_message.empty());
        }
    }

    TEST(CacheAnalyzerHonestyTest, AnUnavailableResultSaysWhy) {
        // The message is the only guidance a caller gets, so it should point at the missing
        // dependency rather than being a generic failure.
        CacheAnalyzer analyzer;
        const auto metrics = analyzer.analyze(0);

        EXPECT_NE(metrics.error_message.find("CUPTI"), std::string::npos) << "the error should name the missing dependency: " << metrics.error_message;
    }

}  // namespace cuda::performance::bandwidth::test
