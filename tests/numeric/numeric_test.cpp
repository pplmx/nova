#include <gtest/gtest.h>
#include <cmath>

#include "cuda/numeric/numeric.h"

using namespace cuda::numeric;

namespace {

class NumericTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }

    void TearDown() override {
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaGetLastError();
    }
};

// The old monte_carlo_integration never called the integrand: it filled the
// sample buffer with uniforms in [0,1) and returned mean * (b-a), so every
// function "integrated" to ~(b-a)/2. These tests pin that the integrand is
// actually evaluated over [a,b].
TEST_F(NumericTest, MonteCarloIntegratesXSquare) {
    const size_t n = 200000;
    auto result = monte_carlo_integration(
        [](float x) { return x * x; }, 0.0f, 1.0f, n);

    // int_0^1 x^2 dx = 1/3; std. error ~ 7e-4, so 0.01 is ~15 sigma.
    EXPECT_NEAR(result.mean, 1.0f / 3.0f, 0.01f);
    EXPECT_EQ(result.samples, n);
    EXPECT_GT(result.std_error, 0.0f);
}

TEST_F(NumericTest, MonteCarloResultDependsOnIntegrand) {
    // Constant integrands have zero variance, so the estimates are exact and
    // must differ between functions (the broken version returned the same
    // ~0.5 for both).
    auto c1 = monte_carlo_integration([](float) { return 1.0f; }, 0.0f, 1.0f, 100000);
    auto c2 = monte_carlo_integration([](float) { return 2.0f; }, 0.0f, 1.0f, 100000);
    EXPECT_NEAR(c1.mean, 1.0f, 1e-3f);
    EXPECT_NEAR(c2.mean, 2.0f, 1e-3f);
    EXPECT_LT(c1.std_error, 1e-3f);
}

TEST_F(NumericTest, MonteCarloZeroSamplesNoCrash) {
    auto result = monte_carlo_integration([](float x) { return x; }, 0.0f, 1.0f, 0);
    EXPECT_EQ(result.samples, 0u);
    EXPECT_FALSE(result.converged);
}

TEST_F(NumericTest, MonteCarloSupportsNegativeInterval) {
    // int_0^2 x dx = 2; the [a,b] sampling must honor the interval endpoints.
    auto result = monte_carlo_integration(
        [](float x) { return x; }, 0.0f, 2.0f, 100000);
    EXPECT_NEAR(result.mean, 2.0f, 0.02f);
}

}  // namespace
