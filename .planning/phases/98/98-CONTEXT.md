# Phase 98 Context — FP8 Foundation

**Phase:** 98
**Milestone:** v2.12 Advanced Quantization
**Goal:** FP8 type support and CUDA kernels for E4M3/E5M2 formats
**Started:** 2026-05-03

---

## Domain Scope

### What is FP8?

FP8 (8-bit floating point) is a floating-point format using only 8 bits:
- **E4M3:** 4-bit exponent, 3-bit mantissa (11 values: 1 sign + 4 exp + 3 mantissa)
  - Range: ~240 distinct positive values
  - Used for: Weights, activations with limited dynamic range
- **E5M2:** 5-bit exponent, 2-bit mantissa (12 values: 1 sign + 5 exp + 2 mantissa)
  - Range: ~240 values, wider dynamic range
  - Used for: Gradients, high dynamic range data

### IEEE 754-like semantics
- E4M3: Max normal = 240.0, Min normal = 2^-6 = 0.015625
- E5M2: Max normal = 57344.0, Min normal = 2^-14 ≈ 6.1e-5
- Special values: 0, Inf, NaN, subnormal

---

## Existing Code

### quantize_tensor.hpp (existing)
```cpp
struct alignas(2) float16 {
    uint16_t value;
    float16(float f);        // FP32 → FP16
    operator float() const;  // FP16 → FP32
};

template<typename T>
class QuantizedTensor {
    QuantizationMetadata metadata_;
    std::vector<T> data_;
    static std::optional<QuantizedTensor<int8_t>> FromFloat(const float* data, size_t size, float scale);
    std::vector<float> ToFloat() const;
};
```

### quantize_ops.hpp (existing)
```cpp
enum class Precision { FP32, FP16, INT8 };
class QuantizedMatmul {
    static void forward(const QuantizedInt8& a, const QuantizedInt8& b, QuantizedInt8& output, int m, int k, int n);
    static std::vector<float> mixed_precision(const float* a, const int8_t* b, const float* scale_b, int m, int k, int n, Precision output_precision);
};
```

---

## Implementation Plan

### 1. FP8 Type Definitions

```cpp
// include/cuda/quantize/fp8_types.hpp
namespace nova::quantize {

struct alignas(1) FP8E4M3 {
    uint8_t value;
    FP8E4M3() = default;
    explicit FP8E4M3(float f);
    operator float() const;
    static constexpr uint8_t POS_INF = 0x7C;
    static constexpr uint8_t NEG_INF = 0xFC;
    static constexpr uint8_t NAN = 0x7D;
    static constexpr uint8_t NEG_ZERO = 0x80;
};

struct alignas(1) FP8E5M2 {
    uint8_t value;
    FP8E5M2() = default;
    explicit FP8E5M2(float f);
    operator float() const;
    // Similar special values
};

} // namespace
```

### 2. Conversion Logic

E4M3 conversion:
1. Extract sign bit
2. Get absolute value
3. If abs > 240.0, clamp to INF
4. If abs < 0.015625 and abs != 0, subnormal
5. Otherwise: compute exp, mantissa, reconstruct

E5M2 conversion (similar but wider range)

### 3. CUDA Kernels

```cpp
// src/cuda/quantize/fp8_kernels.cu
namespace nova::quantize::cuda {

template<typename SrcT, typename DstT>
__global__ void quantize_kernel(const SrcT* src, DstT* dst, size_t n, float scale);

template<typename DstT, typename SrcT>
__global__ void dequantize_kernel(const DstT* src, SrcT* dst, size_t n, float scale);

} // namespace
```

### 4. FP8 GEMM

```cpp
// include/cuda/quantize/fp8_gemm.hpp
namespace nova::quantize {

class FP8GEMM {
public:
    static void forward(const FP8E4M3* a, const FP8E4M3* b, float* output,
                        int m, int k, int n,
                        const float* scale_a, const float* scale_b);
};

} // namespace
```

---

## Dependencies

- Existing: `include/cuda/quantize/quantize_tensor.hpp`
- Existing: `include/cuda/linalg/matmul.hpp`
- New: None (self-contained)

---

## Testing Strategy

1. **Type conversion tests:** All corner cases (NaN, Inf, subnormals, overflow)
2. **Roundtrip tests:** FP32 → FP8 → FP32, verify error bounds
3. **Kernel tests:** CUDA kernel output vs CPU reference
4. **GEMM tests:** vs cuBLAS FP32 reference, tolerance 0.1%

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| FP8 numerical instability | Clamp to max/min, test edge cases |
| Performance not meeting targets | Profile and optimize vectorization |
| Compatibility issues | Graceful fallback if intrinsics unavailable |

---

## Files to Create/Modify

### New files:
- `include/cuda/quantize/fp8_types.hpp`
- `src/cuda/quantize/fp8_types.cpp`
- `include/cuda/quantize/fp8_kernels.cuh`
- `src/cuda/quantize/fp8_kernels.cu`
- `include/cuda/quantize/fp8_gemm.hpp`
- `src/cuda/quantize/fp8_gemm.cu`
- `tests/quantize/fp8_types_test.cpp`
- `tests/quantize/fp8_gemm_test.cpp`

### Modifications:
- `CMakeLists.txt` — Add fp8 sources
- `tests/CMakeLists.txt` — Add fp8 tests
- `include/cuda/quantize/quantize.hpp` — Export FP8 types

---

*Context created: 2026-05-03*
*Phase 98: FP8 Foundation*
