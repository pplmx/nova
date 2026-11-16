#pragma once

#include <array>
#include <cuda_runtime.h>
#include <string>

namespace cuda::testing {

enum class DeterminismLevel {
    NotGuaranteed,
    RunToRun,
    GpuToGpu,
};

class FPDeterminismControl {
public:
    FPDeterminismControl() = default;

    static FPDeterminismControl& instance();

    void set_level(DeterminismLevel level);
    DeterminismLevel level() const { return level_; }

    const char* level_name() const;
    const char* level_description() const;

    /// Records that the caller wants flush-to-zero behaviour. This is a REQUEST, not an
    /// effect: FTZ is a compilation mode (-ftz=true) or an instruction-level setting in
    /// device code, and no CUDA runtime call enables it for a running kernel. The flag
    /// below is the only thing this changes; kernels compiled without FTZ keep their
    /// denormal behaviour.
    void enable_flush_to_zero();
    void disable_flush_to_zero();

    /// Records that the caller wants denormals-are-zero behaviour. Same caveat as
    /// enable_flush_to_zero: DAZ is not switchable at runtime.
    void enable_denormals_are_zero();
    void disable_denormals_are_zero();

    /// Whether enable_flush_to_zero() was requested. This does NOT mean the device is
    /// flushing denormals.
    bool is_flush_to_zero_enabled() const { return ftz_enabled_; }
    /// Whether enable_denormals_are_zero() was requested. This does NOT mean the device
    /// is treating denormals as zero.
    bool is_denormals_are_zero_enabled() const { return daz_enabled_; }

    void reset();

    struct DeterminismResult {
        /// True only when at least two iterations were actually compared and every
        /// one matched exactly. An empty buffer or a single iteration has nothing to
        /// compare, so it cannot establish determinism.
        bool deterministic;
        /// True when a comparison really happened. Without it a caller could not
        /// tell "verified identical" from "never looked": the previous version
        /// reported deterministic = true for null data and for iterations == 1,
        /// having compared nothing at all.
        bool verified;
        size_t iteration_count;
        double max_difference;
        double mean_difference;
    };

    DeterminismResult verify_determinism(const void* data, size_t size, size_t iterations);

private:
    FPDeterminismControl(const FPDeterminismControl&) = delete;
    FPDeterminismControl& operator=(const FPDeterminismControl&) = delete;

    DeterminismLevel level_ = DeterminismLevel::NotGuaranteed;
    bool ftz_enabled_ = false;
    bool daz_enabled_ = false;
};

const char* determinism_level_name(DeterminismLevel level);
const char* determinism_level_description(DeterminismLevel level);

class ScopedDeterminism {
public:
    explicit ScopedDeterminism(DeterminismLevel level);
    ~ScopedDeterminism();

    ScopedDeterminism(const ScopedDeterminism&) = delete;
    ScopedDeterminism& operator=(const ScopedDeterminism&) = delete;

private:
    DeterminismLevel previous_level_;
    bool previous_ftz_;
    bool previous_daz_;
};

}  // namespace cuda::testing
