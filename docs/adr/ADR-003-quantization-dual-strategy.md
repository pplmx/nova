# ADR-003: INT8 + FP8 Dual Quantization Strategy

**Date:** 2026-07-25
**Status:** Accepted
**Context version:** v2.11

## Context

Nova targets both general inference (where INT8 is the industry
standard) and modern datacenter GPUs (H100/H200) where FP8 hardware
support provides 4-8× speedup over FP16 for specific tensor operations.

Supporting only INT8 would leave H100/H200 performance on the table.
Supporting only FP8 would exclude consumer GPUs and older datacenter
hardware (A100, V100) that lack native FP8 tensor cores.

## Decision

Implement both quantization paths as equal citizens:

| Mode | Precision | Use case |
|------|-----------|----------|
| INT8 | 8-bit integer | General inference, all GPUs |
| FP8 E4M3 | 8-bit float (4 exp, 3 mantissa) | Activations, small values |
| FP8 E5M2 | 8-bit float (5 exp, 2 mantissa) | Weights, large values |

Quantization is calibration-driven: `MinMaxCalibrator` collects
statistics from representative data to determine per-tensor scales.

Quantization-aware training (QAT) is supported as an opt-in path for
accuracy-critical deployments.

## Consequences

- **Positive**: Hardware-adaptive — code paths select INT8 or FP8 at
  compile time based on `__CUDA_ARCH__` and runtime capability checks.
  Consumer GPUs use INT8; H100/H200 use FP8 for supported operations.
- **Positive**: The calibration interface is shared across INT8 and FP8
  — callers don't need to know which path will be used.
- **Negative**: Dual-path maintenance burden. INT8 and FP8 kernels have
  different numeric behavior (saturation vs rounding), requiring
  separate test suites.
- **Negative**: FP8 E4M3/E5M2 format selection is delegated to the
  caller — incorrect format selection produces silently degraded
  accuracy. Mitigated by documentation and clear naming conventions.

## Alternatives considered

- **INT8 only**: Rejected — leaves 4-8× performance on H100/H200
  unused.
- **FP8 only**: Rejected — excludes all hardware without native FP8
  tensor cores.
- **Automatic format selection**: Rejected — the optimal format depends
  on the tensor's value distribution (activations vs weights), which
  the library cannot determine without calibration data that crosses
  abstraction boundaries. Caller knows their data best.

## See also

- `docs/QUANTIZATION.md` — usage guide with code examples
- `docs/PERFORMANCE_TOOLING.md` — NVBlox metrics for quantized kernel profiling
