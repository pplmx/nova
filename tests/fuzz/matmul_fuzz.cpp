// NOT A FUZZ TARGET FOR THIS PROJECT - do not wire it to a libFuzzer driver.
//
// The body generates two matrices, multiplies them with its own triple loop, then checks that
// loop's output for NaN/Inf. It calls nothing from nova: no cuda::neural::matmul, no cuBLAS
// path, not even a memory::Buffer. Fuzzing it would fuzz the arithmetic in this file.
//
// Left in place rather than deleted because removing files is the maintainer's call. See
// tests/fuzz/README_STATUS.md. If this file is meant to exist, it needs rewriting to call the
// library the way algorithm_fuzz.cpp calls reduce_sum.

#include "fuzz_utils.hpp"
#include <vector>
#include <cmath>

namespace nova {
namespace fuzz {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    FuzzedDataProvider provider(data, size);

    // Generate matrix dimensions with reasonable limits
    auto M = provider.ConsumeIntegralInRange<size_t>(1, 256);
    auto K = provider.ConsumeIntegralInRange<size_t>(1, 256);
    auto N = provider.ConsumeIntegralInRange<size_t>(1, 256);

    // Limit total elements to prevent OOM
    size_t max_elements = 256 * 256;
    if (M * K > max_elements || K * N > max_elements || M * N > max_elements) {
        return 0; // Skip this input
    }

    // Generate precision mode (0=FP32, 1=FP16, 2=FP64)
    auto precision = provider.ConsumeIntegralInRange<int>(0, 2);

    // Create input matrices
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);

    // Fill with fuzzed data
    for (size_t i = 0; i < M * K; ++i) {
        A[i] = provider.ConsumeFloatingPoint<float>();
    }
    for (size_t i = 0; i < K * N; ++i) {
        B[i] = provider.ConsumeFloatingPoint<float>();
    }

    // Simplified matmul for fuzzing (full precision)
    // This tests numerical stability across different input ranges
    try {
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[m * K + k] * B[k * N + n];
                }
                C[m * N + n] = sum;
            }
        }

        // Check for NaN/Inf
        for (size_t i = 0; i < M * N; ++i) {
            if (std::isnan(C[i]) || std::isinf(C[i])) {
                // Log but don't crash - NaN/Inf are valid float values
            }
        }
    } catch (...) {
        return 1; // Crash on error
    }

    return 0;
}

} // namespace fuzz
} // namespace nova
