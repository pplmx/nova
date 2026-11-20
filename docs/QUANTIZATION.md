# Quantization Guide

Nova provides INT8 and FP8 quantization support for accelerated inference with reduced memory and improved throughput.

## Quantization Modes

| Mode           | Precision | Speedup     | Use Case                  |
| -------------- | --------- | ----------- | ------------------------- |
| **FP16**       | 16-bit    | 1x baseline | Good accuracy             |
| **INT8**       | 8-bit     | 2-4x faster | General inference         |
| **FP8 (E4M3)** | 8-bit     | 4-8x faster | Activations, small values |
| **FP8 (E5M2)** | 8-bit     | 4-8x faster | Weights, large values     |

## Calibration

Calibration determines optimal quantization scales from real data.

### Min-Max Calibration

```cpp
#include "cuda/quantize/calibrator.hpp"

nova::quantize::MinMaxCalibrator calibrator;

// Collect statistics from representative data
for (const auto& batch : calibration_data) {
    calibrator.collect_data(batch.data(), batch.size());
}

// Compute scale
auto result = calibrator.compute();
std::cout << "Scale: " << result.scale << "\n";
std::cout << "Range: [" << result.min_val << ", " << result.max_val << "]\n";
```

### Percentile Calibration

More robust to outliers:

```cpp
// There is no PercentileCalibrator: HistogramCalibrator takes the percentile as a
// constructor argument. calibrate() returns the result; it does not accumulate first.
nova::quantize::HistogramCalibrator calibrator(
    /*num_bins=*/2048, /*percentile=*/99.99f, /*symmetric=*/true);
auto result = calibrator.calibrate(data.data(), data.size(), /*stream=*/0);
```

The calibrators this tree provides are `MinMaxCalibrator`, `HistogramCalibrator`,
`MSECalibrator` and `PerChannelCalibrator`, all deriving from `Calibrator`.

## INT8 Quantization

### Quantizing Activations

```cpp
#include "cuda/quantize/int8_kernels.hpp"
#include "cuda/quantize/quantize_tensor.hpp"

// Create quantization parameters
nova::quantize::cuda::QuantizationParams params{
    .scale = calibration_result.scale,
    .zero_point = calibration_result.zero_point,
    .symmetric = false
};

// Quantize FP32 to INT8
std::vector<int8_t> int8_data(size);
quantize_f32_to_int8(
    float_data.data(),
    int8_data.data(),
    size,
    params,
    stream
);
```

### Quantized GEMM

```cpp
#include "cuda/quantize/quantize_ops.hpp"

// Quantized matrix multiply with INT8 accumulation
nova::quantize::QuantizedInt8 a_quant, b_quant;
// ... quantization setup ...

nova::quantize::QuantizedMatmul::forward(
    a_quant, b_quant, c_int32, m, n, k,
    scale_a, scale_b, scale_c
);
```

## FP8 Quantization (H100/H200)

FP8 requires NVIDIA H100 or newer GPUs with Tensor Core support.

### FP8 Types

```cpp
#include "cuda/quantize/fp8_types.hpp"

// E4M3: Better for activations (larger dynamic range)
using FP8E4M3 = nova::quantize::FP8E4M3;

// E5M2: Better for weights (more exponent range)
using FP8E5M2 = nova::quantize::FP8E5M2;
```

### FP8 GEMM

```cpp
#include "cuda/quantize/fp8_gemm.hpp"

nova::quantize::FP8GEMM::Config config{
    .scale_a = scale_activation,
    .scale_b = scale_weights,
    .scale_out = scale_output
};

FP8GEMM gemm;
gemm.configure(config);
gemm.forward(fp8_a, fp8_b, fp8_c, m, n, k);
```

## Quantization-Aware Training (QAT)

QAT simulates quantization effects during training for better accuracy.

```cpp
#include "cuda/quantize/qat.hpp"

// Fake quantization simulation. FakeQuantize takes a Config; there is no
// QuantScheme enum and no num_bits/quant_scheme aggregate initialiser.
nova::quantize::FakeQuantize::Config cfg;
cfg.precision = nova::quantize::Precision::INT8;
cfg.quant_bits = 8;
nova::quantize::FakeQuantize fq(cfg);
```

## Best Practices

1. **Use representative calibration data**: Include edge cases from your distribution

2. **Per-channel vs per-tensor**: Per-channel often better for weights, per-tensor for activations

3. **FP8 for modern GPUs**: H100/H200 provide native FP8 Tensor Cores

4. **Validate accuracy**: Always measure accuracy degradation after quantization

5. **Use QAT for difficult models**: End-to-end quantization may degrade accuracy significantly

## Performance Comparison

| Configuration | Throughput | Memory | Accuracy |
| ------------- | ---------- | ------ | -------- |
| FP32          | 1x         | 1x     | 100%     |
| FP16          | 1.5x       | 0.5x   | ~100%    |
| INT8          | 3x         | 0.25x  | 98-100%  |
| FP8           | 4-8x       | 0.25x  | 97-100%  |

## See Also

- [FP8 Types Reference](../include/cuda/quantize/fp8_types.hpp)
- [Calibration API](../include/cuda/quantize/calibrator.hpp)
- [Quantization-Aware Training](../include/cuda/quantize/qat.hpp)
