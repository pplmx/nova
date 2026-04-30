#include "cuda/testing/layer_error_injection.h"

#include <array>

namespace cuda::testing {

static constexpr std::array<const char*, 6> LAYER_NAMES = {
    "Memory",
    "Device",
    "Algorithm",
    "Stream",
    "Inference",
    "Production",
};

LayerAwareErrorInjector& LayerAwareErrorInjector::instance() {
    static LayerAwareErrorInjector instance;
    return instance;
}

void LayerAwareErrorInjector::inject_at_layer(LayerBoundary layer,
                                              cuda::production::ErrorTarget target,
                                              cudaError_t error) {
    size_t idx = static_cast<size_t>(layer);
    if (idx < 6 && layer_enabled_[idx]) {
        injectors_[idx].inject_always(target, error);
    }
}

void LayerAwareErrorInjector::inject_once_at_layer(LayerBoundary layer,
                                                    cuda::production::ErrorTarget target,
                                                    cudaError_t error) {
    size_t idx = static_cast<size_t>(layer);
    if (idx < 6 && layer_enabled_[idx]) {
        injectors_[idx].inject_once(target, error);
    }
}

void LayerAwareErrorInjector::inject_random_at_layer(LayerBoundary layer,
                                                      cuda::production::ErrorTarget target,
                                                      cudaError_t error,
                                                      double probability) {
    size_t idx = static_cast<size_t>(layer);
    if (idx < 6 && layer_enabled_[idx]) {
        injectors_[idx].inject_random(target, error, probability);
    }
}

void LayerAwareErrorInjector::enable_layer(LayerBoundary layer) {
    size_t idx = static_cast<size_t>(layer);
    if (idx < 6) {
        layer_enabled_[idx] = true;
    }
}

void LayerAwareErrorInjector::disable_layer(LayerBoundary layer) {
    size_t idx = static_cast<size_t>(layer);
    if (idx < 6) {
        layer_enabled_[idx] = false;
    }
}

bool LayerAwareErrorInjector::should_inject_at_layer(LayerBoundary layer,
                                                     cuda::production::ErrorTarget target) const {
    size_t idx = static_cast<size_t>(layer);
    if (idx >= 6 || !layer_enabled_[idx]) {
        return false;
    }
    return injectors_[idx].should_inject(target);
}

cudaError_t LayerAwareErrorInjector::get_error_at_layer(LayerBoundary layer,
                                                        cuda::production::ErrorTarget target) const {
    size_t idx = static_cast<size_t>(layer);
    if (idx >= 6) {
        return cudaSuccess;
    }
    return injectors_[idx].get_error(target);
}

size_t LayerAwareErrorInjector::injection_count_at_layer(LayerBoundary layer,
                                                         cuda::production::ErrorTarget target) const {
    size_t idx = static_cast<size_t>(layer);
    if (idx >= 6) {
        return 0;
    }
    return injectors_[idx].injection_count(target);
}

void LayerAwareErrorInjector::reset_layer(LayerBoundary layer) {
    size_t idx = static_cast<size_t>(layer);
    if (idx < 6) {
        injectors_[idx].reset();
    }
}

void LayerAwareErrorInjector::reset_all() {
    for (auto& injector : injectors_) {
        injector.reset();
    }
}

std::vector<LayerAwareErrorInjector::LayerStats> LayerAwareErrorInjector::get_stats() const {
    std::vector<LayerStats> stats;

    for (size_t i = 0; i < 6; ++i) {
        LayerStats stat;
        stat.layer_name = LAYER_NAMES[i];
        stat.layer_index = i;
        stat.total_injections = injectors_[i].total_injection_count();
        stats.push_back(stat);
    }

    return stats;
}

const char* layer_boundary_name(LayerBoundary layer) {
    size_t idx = static_cast<size_t>(layer);
    if (idx < 6) {
        return LAYER_NAMES[idx];
    }
    return "Unknown";
}

LayerBoundary parse_layer_boundary(const std::string& name) {
    for (size_t i = 0; i < 6; ++i) {
        if (name == LAYER_NAMES[i]) {
            return static_cast<LayerBoundary>(i);
        }
    }
    return LayerBoundary::Memory;
}

ScopedLayerErrorInjection::ScopedLayerErrorInjection(
    LayerBoundary layer,
    cuda::production::ErrorTarget target,
    cudaError_t error)
    : layer_(layer), target_(target) {

    LayerAwareErrorInjector::instance().inject_once_at_layer(layer, target, error);
}

ScopedLayerErrorInjection::~ScopedLayerErrorInjection() {
    LayerAwareErrorInjector::instance().reset_layer(layer_);
}

}  // namespace cuda::testing
