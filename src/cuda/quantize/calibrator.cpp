#include <cuda/quantize/calibrator.hpp>
#include <cuda/quantize/int8_kernels.hpp>
#include <cuda/quantize/fp8_types.hpp>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

namespace nova {
namespace quantize {

MinMaxCalibrator::MinMaxCalibrator(bool symmetric)
    : symmetric_(symmetric), min_val_(0.0f), max_val_(0.0f) {}

CalibrationResult MinMaxCalibrator::calibrate(
    const float* data, size_t n,
    cudaStream_t stream) {

    float d_min = 0.0f, d_max = 0.0f;
    cuda::compute_minmax(data, n, &d_min, &d_max, stream);
    cudaStreamSynchronize(stream);

    min_val_ = d_min;
    max_val_ = d_max;

    float scale, zero_point;

    if (symmetric_) {
        float abs_max = std::max(std::abs(min_val_), std::abs(max_val_));
        scale = abs_max / 127.0f;
        zero_point = 0.0f;
    } else {
        scale = (max_val_ - min_val_) / 254.0f;
        zero_point = -min_val_ / scale - 127.0f;
    }

    if (scale < 1e-6f) scale = 1e-6f;

    return CalibrationResult(scale, zero_point, symmetric_, min_val_, max_val_);
}

void MinMaxCalibrator::save_cache(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&min_val_), sizeof(float));
        file.write(reinterpret_cast<const char*>(&max_val_), sizeof(float));
    }
}

void MinMaxCalibrator::load_cache(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&min_val_), sizeof(float));
        file.read(reinterpret_cast<char*>(&max_val_), sizeof(float));
    }
}

HistogramCalibrator::HistogramCalibrator(
    int num_bins, float percentile, bool symmetric)
    : num_bins_(num_bins), percentile_(percentile), symmetric_(symmetric),
      min_val_(0.0f), max_val_(0.0f) {
    histogram_.resize(num_bins_, 0);
}

CalibrationResult HistogramCalibrator::calibrate(
    const float* data, size_t n,
    cudaStream_t stream) {

    float d_min, d_max;
    cuda::compute_minmax(data, n, &d_min, &d_max, stream);
    cudaStreamSynchronize(stream);

    min_val_ = d_min;
    max_val_ = d_max;

    uint32_t* d_hist;
    cudaMalloc(&d_hist, num_bins_ * sizeof(uint32_t));
    cudaMemset(d_hist, 0, num_bins_ * sizeof(uint32_t));

    cuda::build_histogram(data, d_hist, n, min_val_, max_val_, num_bins_, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(histogram_.data(), d_hist, num_bins_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_hist);

    float threshold = find_threshold_percentile(percentile_);

    float abs_max = std::max(std::abs(threshold), std::abs(-threshold));
    float scale = abs_max / 127.0f;
    float zero_point = 0.0f;

    if (scale < 1e-6f) scale = 1e-6f;

    return CalibrationResult(scale, zero_point, symmetric_, min_val_, max_val_);
}

float HistogramCalibrator::find_threshold_percentile(float percentile) {
    uint64_t total = 0;
    for (uint32_t count : histogram_) {
        total += count;
    }

    uint64_t target = static_cast<uint64_t>(total * percentile / 100.0f);

    float range = max_val_ - min_val_;
    float bin_width = range / num_bins_;

    uint64_t cumulative = 0;
    for (int i = 0; i < num_bins_; ++i) {
        cumulative += histogram_[i];
        if (cumulative >= target) {
            return min_val_ + (i + 0.5f) * bin_width;
        }
    }

    return max_val_;
}

void HistogramCalibrator::save_cache(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&num_bins_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&min_val_), sizeof(float));
        file.write(reinterpret_cast<const char*>(&max_val_), sizeof(float));
        file.write(reinterpret_cast<const char*>(histogram_.data()),
                   num_bins_ * sizeof(uint32_t));
    }
}

void HistogramCalibrator::load_cache(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&num_bins_), sizeof(int));
        file.read(reinterpret_cast<char*>(&min_val_), sizeof(float));
        file.read(reinterpret_cast<char*>(&max_val_), sizeof(float));
        histogram_.resize(num_bins_);
        file.read(reinterpret_cast<char*>(histogram_.data()),
                  num_bins_ * sizeof(uint32_t));
    }
}

MSECalibrator::MSECalibrator(bool symmetric)
    : symmetric_(symmetric), min_val_(0.0f), max_val_(0.0f),
      best_scale_(1.0f), best_zero_point_(0.0f) {}

CalibrationResult MSECalibrator::calibrate(
    const float* data, size_t n,
    cudaStream_t stream) {

    float d_min, d_max;
    cuda::compute_minmax(data, n, &d_min, &d_max, stream);
    cudaStreamSynchronize(stream);

    min_val_ = d_min;
    max_val_ = d_max;

    float best_mse = std::numeric_limits<float>::max();
    float best_scale = 1.0f;

    for (float candidate_scale = 0.001f; candidate_scale <= (max_val_ - min_val_); candidate_scale *= 2.0f) {
        float mse = 0.0f;
        for (size_t i = 0; i < std::min(n, static_cast<size_t>(10000)); ++i) {
            float quantized = std::round(data[i] / candidate_scale) * candidate_scale;
            float err = (data[i] - quantized) * (data[i] - quantized);
            mse += err;
        }
        mse /= std::min(n, static_cast<size_t>(10000));

        if (mse < best_mse) {
            best_mse = mse;
            best_scale = candidate_scale;
        }
    }

    best_scale_ = best_scale;
    best_zero_point_ = 0.0f;

    return CalibrationResult(best_scale_, best_zero_point_, symmetric_, min_val_, max_val_);
}

void MSECalibrator::save_cache(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&best_scale_), sizeof(float));
        file.write(reinterpret_cast<const char*>(&best_zero_point_), sizeof(float));
    }
}

void MSECalibrator::load_cache(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&best_scale_), sizeof(float));
        file.read(reinterpret_cast<char*>(&best_zero_point_), sizeof(float));
    }
}

PerChannelCalibrator::PerChannelCalibrator(int channel_dim, bool symmetric)
    : channel_dim_(channel_dim), symmetric_(symmetric) {}

CalibrationResult PerChannelCalibrator::calibrate(
    const float* data, size_t n,
    cudaStream_t stream) {

    float d_min, d_max;
    cuda::compute_minmax(data, n, &d_min, &d_max, stream);
    cudaStreamSynchronize(stream);

    float scale, zero_point;

    if (symmetric_) {
        float abs_max = std::max(std::abs(d_min), std::abs(d_max));
        scale = abs_max / 127.0f;
        zero_point = 0.0f;
    } else {
        scale = (d_max - d_min) / 254.0f;
        zero_point = -d_min / scale - 127.0f;
    }

    if (scale < 1e-6f) scale = 1e-6f;

    return CalibrationResult(scale, zero_point, symmetric_, d_min, d_max);
}

std::vector<CalibrationResult> PerChannelCalibrator::calibrate_per_channel(
    const float* data, const std::vector<int>& shape,
    cudaStream_t stream) {

    channel_results_.clear();

    int num_channels = shape[channel_dim_];
    int inner_size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int>(i) != channel_dim_) {
            inner_size *= shape[i];
        }
    }

    for (int c = 0; c < num_channels; ++c) {
        size_t offset = static_cast<size_t>(c) * inner_size;
        auto result = calibrate(data + offset, inner_size, stream);
        channel_results_.push_back(result);
    }

    return channel_results_;
}

void PerChannelCalibrator::save_cache(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        int num_channels = static_cast<int>(channel_results_.size());
        file.write(reinterpret_cast<const char*>(&num_channels), sizeof(int));
        for (const auto& result : channel_results_) {
            file.write(reinterpret_cast<const char*>(&result.scale), sizeof(float));
            file.write(reinterpret_cast<const char*>(&result.zero_point), sizeof(float));
        }
    }
}

void PerChannelCalibrator::load_cache(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        int num_channels;
        file.read(reinterpret_cast<char*>(&num_channels), sizeof(int));
        channel_results_.resize(num_channels);
        for (int i = 0; i < num_channels; ++i) {
            file.read(reinterpret_cast<char*>(&channel_results_[i].scale), sizeof(float));
            file.read(reinterpret_cast<char*>(&channel_results_[i].zero_point), sizeof(float));
        }
    }
}

} // namespace quantize
} // namespace nova
