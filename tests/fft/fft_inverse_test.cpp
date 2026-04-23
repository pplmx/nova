#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "cuda/fft/fft.h"
#include "cuda/stream/stream.h"

class FFTInverseTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }

    static constexpr size_t FFT_SIZE = 256;

    float* allocate_device(size_t n) {
        float* ptr;
        cudaMalloc(&ptr, n * sizeof(float));
        return ptr;
    }

    cuComplex* allocate_complex_device(size_t n) {
        cuComplex* ptr;
        cudaMalloc(&ptr, n * sizeof(cuComplex));
        return ptr;
    }

    void free_device(void* ptr) {
        cudaFree(ptr);
    }

    void copy_to_device(float* dst, const float* src, size_t n) {
        cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice);
    }

    void copy_from_device(float* dst, const float* src, size_t n) {
        cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void copy_from_complex(cuComplex* dst, const cuComplex* src, size_t n) {
        cudaMemcpy(dst, src, n * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    }
};

TEST_F(FFTInverseTest, ForwardTransformPlanCreation) {
    cuda::fft::FFTPlan plan(FFT_SIZE);
    EXPECT_TRUE(static_cast<bool>(plan));
}

TEST_F(FFTInverseTest, PlanWithDifferentTransformTypes) {
    cuda::fft::FFTPlan r2c_plan(FFT_SIZE, cuda::fft::Direction::Forward, cuda::fft::TransformType::RealToComplex);
    cuda::fft::FFTPlan c2r_plan(FFT_SIZE, cuda::fft::Direction::Inverse, cuda::fft::TransformType::ComplexToReal);
    cuda::fft::FFTPlan c2c_plan(FFT_SIZE, cuda::fft::Direction::Forward, cuda::fft::TransformType::ComplexToComplex);

    EXPECT_TRUE(static_cast<bool>(r2c_plan));
    EXPECT_TRUE(static_cast<bool>(c2r_plan));
    EXPECT_TRUE(static_cast<bool>(c2c_plan));
}

TEST_F(FFTInverseTest, MultiplePlansCreated) {
    for (int i = 0; i < 3; ++i) {
        cuda::fft::FFTPlan plan(128);
        EXPECT_TRUE(static_cast<bool>(plan));
    }
    SUCCEED();
}

TEST_F(FFTInverseTest, TransformTypeQuery) {
    cuda::fft::FFTPlan plan(256, cuda::fft::Direction::Forward, cuda::fft::TransformType::ComplexToComplex);
    EXPECT_EQ(plan.type(), cuda::fft::TransformType::ComplexToComplex);
    EXPECT_EQ(plan.direction(), cuda::fft::Direction::Forward);
}
