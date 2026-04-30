---
phase: 81
plan: 02
type: summary
wave: 1
status: complete
files_modified:
  - include/cuda/sparse/roofline.hpp
---

# Phase 81, Plan 02: Roofline JSON Export

## Status: Complete

### Files Modified

**include/cuda/sparse/roofline.hpp:**
- Added `ClassificationConfidence` struct with confidence percentage
- Added `classify_with_confidence()` method for detailed classification
- Added `bound_to_string()` helper for JSON export
- Added `to_json()` method for single kernel metrics
- Added `to_json_device_info()` method for device peaks
- Added `RooflineAnalysis` class for multi-kernel comparison with JSON export

### JSON Export Schema

```json
{
  "metadata": {
    "device_name": "...",
    "compute_capability": "8.0",
    "precision": "FP32",
    "timestamp": "2026-05-01T00:00:00Z"
  },
  "device_peaks": {
    "peak_gflops": 10000,
    "peak_bandwidth_gbps": 1000,
    "fp64_peak_gflops": 5000,
    "fp32_peak_gflops": 10000,
    "fp16_peak_gflops": 50000
  },
  "kernels": [{
    "name": "spmv_csr",
    "arithmetic_intensity": 0.09,
    "achieved_gflops": 500,
    "performance_bound": "MEMORY_BOUND",
    "efficiency_percent": 5.0
  }]
}
```

### Classification Confidence

- ratio = (AI × bandwidth) / peak_compute
- ratio < 0.85: MEMORY_BOUND with (1-ratio)×100% confidence
- ratio > 1.15: COMPUTE_BOUND with 50+(ratio-1)×50% confidence
- else: BALANCED with 100-|ratio-1|×200% confidence

### Verification

SpMV AI: 0.093 FLOPs/byte
Bandwidth ceiling: 93 GFLOPS
Classification confidence: 99.07%
Classification: MEMORY_BOUND

---
*Summary generated: 2026-05-01*
