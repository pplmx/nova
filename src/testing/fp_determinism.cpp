#include "cuda/testing/fp_determinism.h"

#include <cuda_runtime.h>
#include <cmath>

namespace cuda::testing {

static constexpr std::array<const char*, 3> LEVEL_NAMES = {
    "not_guaranteed",
    "run_to_run",
    "gpu_to_gpu",
};

static constexpr std::array<const char*, 3> LEVEL_DESCRIPTIONS = {
    "No determinism guarantees between runs",
    "Results are reproducible across runs on the same GPU",
    "Results are reproducible across different GPUs",
};

FPDeterminismControl& FPDeterminismControl::instance() {
    static FPDeterminismControl instance;
    return instance;
}

void FPDeterminismControl::set_level(DeterminismLevel level) {
    level_ = level;
}

const char* FPDeterminismControl::level_name() const {
    return LEVEL_NAMES[static_cast<size_t>(level_)];
}

const char* FPDeterminismControl::level_description() const {
    return LEVEL_DESCRIPTIONS[static_cast<size_t>(level_)];
}

void FPDeterminismControl::enable_flush_to_zero() {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    ftz_enabled_ = true;
}

void FPDeterminismControl::disable_flush_to_zero() {
    ftz_enabled_ = false;
}

void FPDeterminismControl::enable_denormals_are_zero() {
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    daz_enabled_ = true;
}

void FPDeterminismControl::disable_denormals_are_zero() {
    daz_enabled_ = false;
}

void FPDeterminismControl::reset() {
    level_ = DeterminismLevel::NotGuaranteed;
    ftz_enabled_ = false;
    daz_enabled_ = false;
}

FPDeterminismControl::DeterminismResult
FPDeterminismControl::verify_determinism(const void* data, size_t size, size_t iterations) {
    DeterminismResult result;
    result.deterministic = true;
    result.iteration_count = iterations;
    result.max_difference = 0.0;
    result.mean_difference = 0.0;

    if (iterations < 2 || data == nullptr || size == 0) {
        return result;
    }

    const double* values = static_cast<const double*>(data);

    double max_diff = 0.0;
    double sum_diff = 0.0;

    for (size_t i = 1; i < iterations; ++i) {
        double diff = std::fabs(values[i] - values[0]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
    }

    result.max_difference = max_diff;
    result.mean_difference = sum_diff / static_cast<double>(iterations - 1);
    result.deterministic = (max_diff == 0.0);

    return result;
}

const char* determinism_level_name(DeterminismLevel level) {
    return LEVEL_NAMES[static_cast<size_t>(level)];
}

const char* determinism_level_description(DeterminismLevel level) {
    return LEVEL_DESCRIPTIONS[static_cast<size_t>(level)];
}

ScopedDeterminism::ScopedDeterminism(DeterminismLevel level)
    : previous_level_(FPDeterminismControl::instance().level()),
      previous_ftz_(FPDeterminismControl::instance().is_flush_to_zero_enabled()),
      previous_daz_(FPDeterminismControl::instance().is_denormals_are_zero_enabled()) {

    FPDeterminismControl::instance().set_level(level);
}

ScopedDeterminism::~ScopedDeterminism() {
    FPDeterminismControl::instance().set_level(previous_level_);

    if (previous_ftz_) {
        FPDeterminismControl::instance().enable_flush_to_zero();
    } else {
        FPDeterminismControl::instance().disable_flush_to_zero();
    }

    if (previous_daz_) {
        FPDeterminismControl::instance().enable_denormals_are_zero();
    } else {
        FPDeterminismControl::instance().disable_denormals_are_zero();
    }
}

}  // namespace cuda::testing
