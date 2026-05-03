#include <cuda/quantize/calibrator.hpp>
#include <cuda/quantize/fp8_types.hpp>
#include <vector>
#include <string>
#include <map>
#include <cstdint>

#ifndef NOVA_CUDA_QUANTIZE_QAT_HPP
#define NOVA_CUDA_QUANTIZE_QAT_HPP

namespace nova {
namespace quantize {

enum class Precision {
    FP32,
    FP16,
    INT8,
    FP8_E4M3,
    FP8_E5M2
};

struct LayerConfig {
    std::string name;
    Precision precision{Precision::FP32};
    float scale{1.0f};
    float zero_point{0.0f};
    bool trainable{true};

    LayerConfig() = default;
    LayerConfig(const std::string& n, Precision p)
        : name(n), precision(p) {}
};

class FakeQuantize {
public:
    struct Config {
        Precision precision{Precision::INT8};
        float scale{1.0f};
        float zero_point{0.0f};
        int quant_bits{8};
        bool per_channel{false};
    };

    FakeQuantize(const Config& config) : config_(config) {}

    Config get_config() const { return config_; }

    float forward(float x) const {
        float scale = config_.scale;
        if (scale < 1e-6f) scale = 1e-6f;

        float normalized = (x - config_.zero_point) / scale;
        float rounded = std::round(normalized);
        float clamped = std::max(-127.0f, std::min(127.0f, rounded));
        return clamped * scale + config_.zero_point;
    }

    float backward(float grad_output, float x) const {
        (void)x;
        return grad_output;
    }

    void update_scale(float new_scale) { config_.scale = new_scale; }
    void update_zero_point(float new_zp) { config_.zero_point = new_zp; }

private:
    Config config_;
};

class AMPManager {
public:
    AMPManager() = default;

    void add_layer(const std::string& name, Precision precision = Precision::FP16) {
        configs_[name] = LayerConfig(name, precision);
    }

    void set_precision(const std::string& name, Precision precision) {
        if (configs_.find(name) != configs_.end()) {
            configs_[name].precision = precision;
        }
    }

    void set_scale(const std::string& name, float scale) {
        if (configs_.find(name) != configs_.end()) {
            configs_[name].scale = scale;
        }
    }

    Precision get_precision(const std::string& name) const {
        auto it = configs_.find(name);
        return (it != configs_.end()) ? it->second.precision : Precision::FP32;
    }

    const LayerConfig& get_config(const std::string& name) const {
        static LayerConfig default_config;
        auto it = configs_.find(name);
        return (it != configs_.end()) ? it->second : default_config;
    }

    std::vector<LayerConfig> get_all_configs() const {
        std::vector<LayerConfig> result;
        for (const auto& pair : configs_) {
            result.push_back(pair.second);
        }
        return result;
    }

    void save_config(const std::string& path) const;
    void load_config(const std::string& path);

    size_t num_layers() const { return configs_.size(); }

private:
    std::map<std::string, LayerConfig> configs_;
};

class SensitivityAnalyzer {
public:
    struct LayerSensitivity {
        std::string name;
        float gradient_magnitude{0.0f};
        Precision recommended_precision{Precision::FP16};
        float confidence{0.0f};
    };

    void analyze_layer(const std::string& name, const float* activations, size_t n, const float* gradients, size_t grad_n) {
        LayerSensitivity result;
        result.name = name;

        float max_act = 0.0f;
        float max_grad = 0.0f;

        for (size_t i = 0; i < std::min(n, grad_n); ++i) {
            max_act = std::max(max_act, std::abs(activations[i]));
            max_grad = std::max(max_grad, std::abs(gradients[i]));
        }

        result.gradient_magnitude = max_act * max_grad;
        result.recommended_precision = (result.gradient_magnitude > 10.0f) ?
            Precision::FP16 : Precision::INT8;
        result.confidence = 0.7f;

        sensitivities_[name] = result;
    }

    Precision get_recommended_precision(const std::string& name) const {
        auto it = sensitivities_.find(name);
        return (it != sensitivities_.end()) ?
            it->second.recommended_precision : Precision::FP16;
    }

    const LayerSensitivity& get_sensitivity(const std::string& name) const {
        static LayerSensitivity default_sensitivity;
        auto it = sensitivities_.find(name);
        return (it != sensitivities_.end()) ? it->second : default_sensitivity;
    }

    std::vector<LayerSensitivity> get_all_sensitivities() const {
        std::vector<LayerSensitivity> result;
        for (const auto& pair : sensitivities_) {
            result.push_back(pair.second);
        }
        return result;
    }

    void auto_assign_precision(AMPManager& amp) const {
        for (const auto& pair : sensitivities_) {
            amp.set_precision(pair.first, pair.second.recommended_precision);
        }
    }

private:
    std::map<std::string, LayerSensitivity> sensitivities_;
};

} // namespace quantize
} // namespace nova

#endif // NOVA_CUDA_QUANTIZE_QAT_HPP
