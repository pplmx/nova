# Phase 100 Context — Calibration Infrastructure

**Phase:** 100
**Milestone:** v2.12 Advanced Quantization
**Goal:** Production calibration utilities with histogram and percentile methods
**Started:** 2026-05-03

---

## Domain Scope

### What's Already Built (Phase 99)

- `build_histogram()` — GPU histogram from float data
- `compute_minmax()` — GPU min/max reduction
- Basic quantization kernels

### What Phase 100 Adds

- **CAL-01:** HistogramCalibrator with percentile selection
- **CAL-02:** MinMax and MSE calibration methods
- **CAL-03:** Per-channel calibration with channel-wise scales

---

## Implementation Plan

### 1. Calibration Base Class

```cpp
// include/cuda/quantize/calibrator.hpp
class Calibrator {
public:
    virtual ~Calibrator() = default;
    virtual QuantizationParams calibrate(const float* data, size_t n) = 0;
    virtual void save_cache(const std::string& path) const = 0;
    virtual void load_cache(const std::string& path) = 0;
};
```

### 2. HistogramCalibrator

Uses histogram to find optimal scale based on percentile.

```cpp
class HistogramCalibrator : public Calibrator {
public:
    HistogramCalibrator(int num_bins = 2048, float percentile = 99.99f);
    QuantizationParams calibrate(const float* data, size_t n) override;
    void save_cache(const std::string& path) const override;
    void load_cache(const std::string& path) override;

private:
    int num_bins_;
    float percentile_;
    std::vector<uint32_t> histogram_;
    float min_val_, max_val_;
};
```

### 3. MinMaxCalibrator

Simple min/max-based scaling.

```cpp
class MinMaxCalibrator : public Calibrator {
public:
    MinMaxCalibrator();
    QuantizationParams calibrate(const float* data, size_t n) override;

private:
    float min_val_, max_val_;
};
```

### 4. MSECalibrator

Optimizes scale to minimize mean squared error.

### 5. PerChannelCalibrator

Computes per-channel scales for activations.

---

## Files to Create

### New files:
- `include/cuda/quantize/calibrator.hpp`
- `tests/quantize/calibrator_test.cpp`

---

*Context created: 2026-05-03*
*Phase 100: Calibration Infrastructure*
