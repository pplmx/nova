#ifndef NOVA_CUDA_QUANTIZE_CALIBRATOR_HPP
#define NOVA_CUDA_QUANTIZE_CALIBRATOR_HPP

#include <cuda/quantize/int8_kernels.hpp>
#include <cuda/quantize/fp8_types.hpp>
#include <string>
#include <vector>
#include <optional>
#include <cstdint>

namespace nova {
namespace quantize {

struct CalibrationResult {
    float scale;
    float zero_point;
    bool symmetric;
    float min_val;
    float max_val;

    CalibrationResult() : scale(1.0f), zero_point(0.0f), symmetric(true), min_val(0.0f), max_val(0.0f) {}
    CalibrationResult(float s, float zp, bool sym, float min_v, float max_v)
        : scale(s), zero_point(zp), symmetric(sym), min_val(min_v), max_val(max_v) {}
};

class Calibrator {
public:
    virtual ~Calibrator() = default;

    virtual CalibrationResult calibrate(
        const float* data, size_t n,
        cudaStream_t stream = 0) = 0;

    virtual void save_cache(const std::string& path) const = 0;
    virtual void load_cache(const std::string& path) = 0;

    virtual std::string name() const = 0;
};

class MinMaxCalibrator : public Calibrator {
public:
    explicit MinMaxCalibrator(bool symmetric = true);

    CalibrationResult calibrate(
        const float* data, size_t n,
        cudaStream_t stream = 0) override;

    void save_cache(const std::string& path) const override;
    void load_cache(const std::string& path) override;

    std::string name() const override { return "MinMax"; }

    float get_min() const { return min_val_; }
    float get_max() const { return max_val_; }

private:
    bool symmetric_;
    float min_val_;
    float max_val_;
};

class HistogramCalibrator : public Calibrator {
public:
    explicit HistogramCalibrator(
        int num_bins = 2048,
        float percentile = 99.99f,
        bool symmetric = true);

    CalibrationResult calibrate(
        const float* data, size_t n,
        cudaStream_t stream = 0) override;

    void save_cache(const std::string& path) const override;
    void load_cache(const std::string& path) override;

    std::string name() const override { return "Histogram"; }

    int get_num_bins() const { return num_bins_; }
    float get_percentile() const { return percentile_; }

private:
    int num_bins_;
    float percentile_;
    bool symmetric_;

    float min_val_;
    float max_val_;
    std::vector<uint32_t> histogram_;

    float find_threshold_percentile(float percentile);
};

class MSECalibrator : public Calibrator {
public:
    explicit MSECalibrator(bool symmetric = true);

    CalibrationResult calibrate(
        const float* data, size_t n,
        cudaStream_t stream = 0) override;

    void save_cache(const std::string& path) const override;
    void load_cache(const std::string& path) override;

    std::string name() const override { return "MSE"; }

private:
    bool symmetric_;
    float min_val_;
    float max_val_;
    float best_scale_;
    float best_zero_point_;
};

class PerChannelCalibrator : public Calibrator {
public:
    explicit PerChannelCalibrator(
        int channel_dim = 0,
        bool symmetric = true);

    std::vector<CalibrationResult> calibrate_per_channel(
        const float* data, const std::vector<int>& shape,
        cudaStream_t stream = 0);

    CalibrationResult calibrate(
        const float* data, size_t n,
        cudaStream_t stream = 0) override;

    void save_cache(const std::string& path) const override;
    void load_cache(const std::string& path) override;

    std::string name() const override { return "PerChannel"; }

    int get_channel_dim() const { return channel_dim_; }

private:
    int channel_dim_;
    bool symmetric_;
    std::vector<CalibrationResult> channel_results_;
};

} // namespace quantize
} // namespace nova

#endif // NOVA_CUDA_QUANTIZE_CALIBRATOR_HPP
