/**
 * @file declared_but_unimplemented_test.cpp
 * @brief Public API that the library declares must be linkable
 *
 * Round 121 fixed QuantizationBenchmark::benchmark_dequantization, which was declared in the
 * header and never defined. The scan that found it compared declarations against definitions
 * textually and produced 97 candidates, almost all parameter names and macros. This file
 * takes the opposite approach for the cases it can reach: it names the functions and lets the
 * linker decide. A declaration without a definition makes this translation unit fail to link,
 * which is a verdict no scan can fake.
 *
 * Two declarations in the tree have no definition and are deliberately not referenced here,
 * because referencing them would be a link error that says nothing new:
 *
 *   BeamSearchManager::expand_beams   private, never called; documented at its declaration
 *   sha256 helpers in checkpoint code  provided by OpenSSL, not by this library
 *
 * What this file covers is the arithmetic and memory surface that a caller reaches through
 * the public headers.
 */

#include <cuda/quantize/benchmark.hpp>
#include <cuda/quantize/int8_kernels.hpp>
#include <cuda/quantize/fp8_kernels.hpp>
#include <cuda/memory/buffer.h>
#include <cuda/memory/unique_ptr.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace declared_but_unimplemented_test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

}  // namespace

// Each case below takes the address of a free function. If any of them lost its definition,
// the test binary would fail to link rather than fail at run time — which is the point.
TEST(DeclaredButUnimplementedTest, TheInt8KernelSurfaceIsLinkable) {
    // Overload-free, so plain address-of is enough and the linker is the arbiter.
    auto quantize = &nova::quantize::cuda::quantize_f32_to_int8;
    auto dequantize = &nova::quantize::cuda::dequantize_int8_to_f32;

    EXPECT_TRUE(quantize != nullptr);
    EXPECT_TRUE(dequantize != nullptr);
}

TEST(DeclaredButUnimplementedTest, TheBenchmarkSurfaceIsLinkable) {
    // The member fixed in round 121, taken by address so linking proves the definition.
    using Member = nova::quantize::benchmark::BenchmarkResult
        (nova::quantize::benchmark::QuantizationBenchmark::*)(
            const std::vector<int8_t>&, nova::quantize::Precision);
    const Member fn =
        &nova::quantize::benchmark::QuantizationBenchmark::benchmark_dequantization;
    EXPECT_TRUE(fn != nullptr);
}

TEST(DeclaredButUnimplementedTest, TheMemorySurfaceIsLinkable) {
    // Buffer and unique_ptr are header-implemented; naming their members makes the compiler
    // instantiate them, which is where a missing body would surface.
    auto release = &cuda::memory::Buffer<float>::release;
    EXPECT_TRUE(release != nullptr);

    auto reset = &cuda::memory::unique_ptr<float>::reset;
    EXPECT_TRUE(reset != nullptr);
}

TEST(DeclaredButUnimplementedTest, TheQuantizationEntryPointsActuallyRun) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Linking is one thing; the definitions doing something is another. One round trip
    // through the pair above keeps both properties covered.
    constexpr size_t kCount = 64;
    float* d_input = nullptr;
    int8_t* d_quantized = nullptr;
    float* d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, kCount * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_quantized, kCount * sizeof(int8_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, kCount * sizeof(float)), cudaSuccess);

    std::vector<float> host(kCount, 0.5F);
    ASSERT_EQ(cudaMemcpy(d_input, host.data(), kCount * sizeof(float),
                         cudaMemcpyHostToDevice), cudaSuccess);

    nova::quantize::cuda::QuantizationParams params(0.1F);
    nova::quantize::cuda::quantize_f32_to_int8(d_input, d_quantized, kCount, params, 0);
    nova::quantize::cuda::dequantize_int8_to_f32(d_quantized, d_output, kCount, params, 0);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);

    std::vector<float> result(kCount, -1.0F);
    ASSERT_EQ(cudaMemcpy(result.data(), d_output, kCount * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_NEAR(result[0], 0.5F, 0.15F)
        << "0.5 through int8 and back should land near 0.5";

    (void)cudaFree(d_input);
    (void)cudaFree(d_quantized);
    (void)cudaFree(d_output);
}

}  // namespace declared_but_unimplemented_test
