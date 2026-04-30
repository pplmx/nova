#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <string>

#include "cuda/production/error_injection.h"

namespace cuda::testing {

enum class LayerBoundary {
    Memory,
    Device,
    Algorithm,
    Stream,
    Inference,
    Production,
};

class LayerAwareErrorInjector {
public:
    LayerAwareErrorInjector() = default;

    static LayerAwareErrorInjector& instance();

    void inject_at_layer(LayerBoundary layer,
                         cuda::production::ErrorTarget target,
                         cudaError_t error);

    void inject_once_at_layer(LayerBoundary layer,
                               cuda::production::ErrorTarget target,
                               cudaError_t error);

    void inject_random_at_layer(LayerBoundary layer,
                                 cuda::production::ErrorTarget target,
                                 cudaError_t error,
                                 double probability);

    void enable_layer(LayerBoundary layer);
    void disable_layer(LayerBoundary layer);

    bool should_inject_at_layer(LayerBoundary layer,
                                 cuda::production::ErrorTarget target) const;

    cudaError_t get_error_at_layer(LayerBoundary layer,
                                    cuda::production::ErrorTarget target) const;

    size_t injection_count_at_layer(LayerBoundary layer,
                                     cuda::production::ErrorTarget target) const;

    void reset_layer(LayerBoundary layer);
    void reset_all();

    struct LayerStats {
        std::string layer_name;
        size_t total_injections;
        size_t layer_index;
    };

    std::vector<LayerStats> get_stats() const;

private:
    cuda::production::ErrorInjector injectors_[6];
    bool layer_enabled_[6] = {true, true, true, true, true, true};
};

const char* layer_boundary_name(LayerBoundary layer);
LayerBoundary parse_layer_boundary(const std::string& name);

class ScopedLayerErrorInjection {
public:
    ScopedLayerErrorInjection(LayerBoundary layer,
                               cuda::production::ErrorTarget target,
                               cudaError_t error);

    ~ScopedLayerErrorInjection();

    ScopedLayerErrorInjection(const ScopedLayerErrorInjection&) = delete;
    ScopedLayerErrorInjection& operator=(const ScopedLayerErrorInjection&) = delete;

private:
    LayerBoundary layer_;
    cuda::production::ErrorTarget target_;
};

}  // namespace cuda::testing
