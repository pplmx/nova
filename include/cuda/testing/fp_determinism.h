#pragma once

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

    void enable_flush_to_zero();
    void disable_flush_to_zero();

    void enable_denormals_are_zero();
    void disable_denormals_are_zero();

    bool is_flush_to_zero_enabled() const { return ftz_enabled_; }
    bool is_denormals_are_zero_enabled() const { return daz_enabled_; }

    void reset();

    struct DeterminismResult {
        bool deterministic;
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
