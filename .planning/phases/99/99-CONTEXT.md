# Phase 99 Context — CUDA Quantization Kernels

**Phase:** 99
**Milestone:** v2.12 Advanced Quantization
**Goal:** CUDA kernels for INT8/FP16 quantization with vectorization
**Started:** 2026-05-03

---

## Domain Scope

### Existing Code

The existing quantize_tensor.hpp has CPU-only quantization:

```cpp
template<>
std::optional<QuantizedTensor<int8_t>> QuantizedTensor<int8_t>::FromFloat(
    const float* data, size_t size, float scale) {
    // CPU implementation - iterate and quantize
    for (size_t i = 0; i < size; ++i) {
        float normalized = data[i] / scale;
        quantized[i] = static_cast<int8_t>(std::round(normalized));
    }
}
```

### Requirements

- **KERN-01:** CUDA kernels for INT8/FP16 quantization (move from CPU)
- **KERN-02:** Vectorized load/store for quantization (128-bit wide)

---

## Implementation Plan

### 1. INT8 Quantization Kernels

```cpp
// include/cuda/quantize/int8_kernels.hpp
namespace nova::quantize::cuda {

void quantize_f32_to_int8(
    const float* src, int8_t* dst, size_t n,
    float scale, float zero_point,
    cudaStream_t stream = 0
);

void dequantize_int8_to_f32(
    const int8_t* src, float* dst, size_t n,
    float scale, float zero_point,
    cudaStream_t stream = 0
);
}
```

### 2. Vectorized Load/Store

```cpp
// Vectorized using float4 (128-bit)
template<typename T>
__global__ void quantize_vectorized_kernel(
    const float4* src, T* dst, size_t n4, float scale) {
    // Process 4 elements at a time
    float4 f4 = src[blockIdx.x];
    dst[idx*4+0] = T(f4.x / scale);
    dst[idx*4+1] = T(f4.y / scale);
    // ...
}
```

### 3. Calibration Histogram (for Phase 100)

```cpp
// Shared memory histogram for calibration
__global__ void build_histogram_kernel(
    const float* data, uint32_t* histogram,
    size_t n, float min_val, float max_val, int num_bins) {
    // Atomic add to histogram bins
}
```

---

## Files to Create

### New files:
- `include/cuda/quantize/int8_kernels.hpp`
- `src/cuda/quantize/int8_kernels.cu`
- `include/cuda/quantize/calibration_kernels.hpp`
- `src/cuda/quantize/calibration_kernels.cu`
- `tests/quantize/int8_kernels_test.cpp`

### Modifications:
- `CMakeLists.txt` — Add to QUANTIZE_SOURCES
- `tests/CMakeLists.txt` — Add int8 test

---

## Performance Targets

- >10x speedup over CPU for 1GB tensors
- Vectorized load/store (128-bit)
- Shared memory for histogram accumulation
- Async copy overlap with compute

---

*Context created: 2026-05-03*
*Phase 99: CUDA Quantization Kernels*
