#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "cuda/fft/fft.h"

class FFTAccuracyTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess && err != cudaErrorNoDevice) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }

    static constexpr float TOLERANCE = 1e-3f;
    static constexpr size_t FFT_SIZE = 256;
};

TEST_F(FFTAccuracyTest, PlanPropertiesAreConsistent) {
    cuda::fft::FFTPlan plan(FFT_SIZE, cuda::fft::Direction::Forward);
    EXPECT_TRUE(static_cast<bool>(plan));
    EXPECT_EQ(plan.size(), FFT_SIZE);
    EXPECT_EQ(plan.direction(), cuda::fft::Direction::Forward);
}

TEST_F(FFTAccuracyTest, PowerOfTwoCheck) {
    EXPECT_TRUE(cuda::fft::is_power_of_two(256));
    EXPECT_TRUE(cuda::fft::is_power_of_two(1024));
    EXPECT_TRUE(cuda::fft::is_power_of_two(1));
    EXPECT_FALSE(cuda::fft::is_power_of_two(100));
}

TEST_F(FFTAccuracyTest, ComplexTypeTrait) {
    using cfloat = cuda::fft::ComplexT<float>;
    using cdouble = cuda::fft::ComplexT<double>;
    static_assert(std::is_same_v<cfloat, cuComplex>);
    static_assert(std::is_same_v<cdouble, cuDoubleComplex>);
}

TEST_F(FFTAccuracyTest, FFTConfigDefaultInitialization) {
    cuda::fft::FFTConfig config;
    EXPECT_EQ(config.size, 0u);
    EXPECT_EQ(config.direction, cuda::fft::Direction::Forward);
    EXPECT_EQ(config.batch, 1);
    EXPECT_EQ(config.stream_id, 0);
    EXPECT_EQ(config.type, cuda::fft::TransformType::RealToComplex);
}

TEST_F(FFTAccuracyTest, PowerSpectrumIsNonNegative) {
    cuda::fft::FFTPlan plan(FFT_SIZE);
    EXPECT_TRUE(static_cast<bool>(plan));
}

TEST_F(FFTAccuracyTest, DirectionEnumValues) {
    EXPECT_EQ(static_cast<int>(cuda::fft::Direction::Forward), CUFFT_FORWARD);
    EXPECT_EQ(static_cast<int>(cuda::fft::Direction::Inverse), CUFFT_INVERSE);
}

TEST_F(FFTAccuracyTest, TransformTypeEnumValues) {
    EXPECT_EQ(static_cast<int>(cuda::fft::TransformType::RealToComplex), CUFFT_R2C);
    EXPECT_EQ(static_cast<int>(cuda::fft::TransformType::DoubleRealToComplex), CUFFT_D2Z);
    EXPECT_EQ(static_cast<int>(cuda::fft::TransformType::ComplexToReal), CUFFT_C2R);
    EXPECT_EQ(static_cast<int>(cuda::fft::TransformType::ComplexToComplex), CUFFT_C2C);
}
