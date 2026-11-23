#include <cuda/quantize/calibrator.hpp>
#include <cuda/quantize/int8_kernels.hpp>
#include <cuda/quantize/fp8_types.hpp>
#include "cuda/device/error.h"
#include "cuda/memory/unique_ptr.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <numeric>
#include <cuda_runtime.h>

namespace nova {
    namespace quantize {

        namespace {

            /// Pick a scale that can actually represent the calibrated data.
            ///
            /// A degenerate range -- every sample equal -- makes (max - min) exactly zero, and
            /// merely flooring that raw value at 1e-6 does not rescue the result: the ratio
            /// (val - zero_point) / scale then reaches millions, saturates at the int8 bound,
            /// and the dequantized value is not the input. Measured, 64 samples of 3.0f,
            /// asymmetric: scale 1e-6, zero_point -127, round-trip max error 2.99987 -- the
            /// value came back as 0. Sizing the step from the magnitude instead keeps the
            /// round-trip exact, and the 1e-6 floor still applies to ordinary scales.
            float stabilized_scale(float scale, float min_val, float max_val) {
                if (max_val - min_val <= 0.0f) {
                    const float magnitude = std::max(std::abs(min_val), std::abs(max_val));
                    if (!std::isfinite(magnitude)) {
                        // NaN or infinite data cannot be represented; a unit scale keeps the
                        // output finite and the caller can see the bad input in min/max.
                        return 1.0f;
                    }
                    return std::max(magnitude, 1.0f) / 127.0f;
                }
                return scale < 1e-6f ? 1e-6f : scale;
            }

        }  // namespace

        MinMaxCalibrator::MinMaxCalibrator(bool symmetric)
            : symmetric_(symmetric),
              min_val_(0.0f),
              max_val_(0.0f) {}

        CalibrationResult MinMaxCalibrator::calibrate(const float* data, size_t n, cudaStream_t stream) {
            float d_min = 0.0f, d_max = 0.0f;
            cuda::compute_minmax(data, n, &d_min, &d_max, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            min_val_ = d_min;
            max_val_ = d_max;

            float scale, zero_point;
            if (symmetric_) {
                float abs_max = std::max(std::abs(min_val_), std::abs(max_val_));
                scale = abs_max / 127.0f;
                zero_point = 0.0f;
            } else {
                scale = (max_val_ - min_val_) / 254.0f;
                zero_point = 0.0f;
            }

            // Clamp the scale BEFORE deriving the zero point. The non-symmetric branch used to
            // compute `zero_point = -min_val / scale - 127` with the raw scale and only then
            // clamp, so a degenerate range divided by zero. Measured, 64 samples:
            //
            //     symmetric=false, all zeros   -> zero_point = nan
            //     symmetric=false, constant 3  -> zero_point = -inf
            //
            // Both are exactly the cases the clamp exists for. A NaN or infinite zero point is
            // carried into every quantized value, so the calibration produced garbage instead
            // of the safe identity the symmetric path produced for the same input.
            scale = stabilized_scale(scale, min_val_, max_val_);
            if (!symmetric_) {
                zero_point = -min_val_ / scale - 127.0f;
                // Keep the result inside the representable int8 span so a caller can use it
                // without a second clamp, and never hand back a non-finite zero point: a NaN
                // min/max must not silently become a plausible-looking offset.
                if (!std::isfinite(zero_point)) {
                    zero_point = 0.0f;
                } else {
                    zero_point = std::clamp(zero_point, -127.0f, 127.0f);
                }
            }

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

        HistogramCalibrator::HistogramCalibrator(int num_bins, float percentile, bool symmetric)
            : num_bins_(num_bins),
              percentile_(percentile),
              symmetric_(symmetric),
              min_val_(0.0f),
              max_val_(0.0f) {
            // Neither parameter was validated, and both failures were unpleasant rather than
            // merely wrong. Measured:
            //   num_bins = -5      -> std::length_error from vector::_M_fill_insert, a message
            //                         that names neither the parameter nor this class
            //   num_bins = 0       -> constructed fine, then calibrate() threw out of
            //                         build_histogram on a call the caller had already made
            //   percentile = -10   -> accepted, calibrate() succeeded
            //   percentile = 150   -> accepted, calibrate() succeeded
            //   percentile = NaN   -> accepted, calibrate() succeeded
            // A percentile outside (0, 100] silently produced a threshold computed from an
            // out-of-range target. Rejecting at construction keeps the error next to the
            // argument that caused it.
            if (num_bins <= 0) {
                throw ::std::invalid_argument("HistogramCalibrator: num_bins must be positive, got " + ::std::to_string(num_bins));
            }
            if (!std::isfinite(percentile) || percentile <= 0.0f || percentile > 100.0f) {
                throw ::std::invalid_argument("HistogramCalibrator: percentile must be in (0, 100], got " + std::to_string(percentile));
            }
            histogram_.resize(num_bins_, 0);
        }

        CalibrationResult HistogramCalibrator::calibrate(const float* data, size_t n, cudaStream_t stream) {
            float d_min, d_max;
            cuda::compute_minmax(data, n, &d_min, &d_max, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            min_val_ = d_min;
            max_val_ = d_max;

            // Owned by unique_ptr so the throwing calls below release it. This was a raw pointer
            // freed at the end of the block: cudaMalloc succeeding and then cudaMemset,
            // build_histogram or the copy failing left the allocation with no owner, and the free sat
            // past every one of those throws.
            ::cuda::memory::unique_ptr<uint32_t> d_hist(num_bins_);

            CUDA_CHECK(cudaMemset(d_hist.get(), 0, num_bins_ * sizeof(uint32_t)));

            cuda::build_histogram(data, d_hist.get(), n, min_val_, max_val_, num_bins_, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaMemcpy(histogram_.data(), d_hist.get(), num_bins_ * sizeof(uint32_t), cudaMemcpyDeviceToHost));

            float threshold = find_threshold_percentile(percentile_);

            float abs_max = std::max(std::abs(threshold), std::abs(-threshold));
            float scale = abs_max / 127.0f;
            float zero_point = 0.0f;

            // A degenerate calibration range -- every sample equal -- makes (max - min) exactly
            // zero, and clamping THAT to 1e-6 does not rescue the result: the ratio
            // (val - zero_point) / scale then reaches 3e6, saturates at the int8 bound, and the
            // dequantized value is not the input. Measured, 64 samples of 3.0f, asymmetric:
            // scale 1e-6, zero_point -127, round-trip max error 2.99987 (the value came back
            // as 0). Falling back to a range that actually represents the data keeps the
            // round-trip exact; the 1e-6 floor still applies to the scale itself.
            if (max_val_ - min_val_ <= 0.0f) {
                // All samples identical (or a degenerate/invalid range). Size the step from the
                // magnitude, covering the full int8 span so zero_point can centre the value.
                const float magnitude = std::max(std::abs(min_val_), std::abs(max_val_));
                if (!std::isfinite(magnitude)) {
                    // NaN or infinite data cannot be represented at all; a unit scale keeps the
                    // output finite and lets the caller see the bad input in min_val/max_val.
                    scale = 1.0f;
                    min_val_ = std::isnan(min_val_) ? 0.0f : min_val_;
                    max_val_ = std::isnan(max_val_) ? 0.0f : max_val_;
                } else {
                    scale = std::max(magnitude, 1.0f) / 127.0f;
                }
            }

            if (scale < 1e-6f) {
                scale = 1e-6f;
            }

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
                file.write(reinterpret_cast<const char*>(histogram_.data()), num_bins_ * sizeof(uint32_t));
            }
        }

        void HistogramCalibrator::load_cache(const std::string& path) {
            std::ifstream file(path, std::ios::binary);
            if (file.is_open()) {
                file.read(reinterpret_cast<char*>(&num_bins_), sizeof(int));
                file.read(reinterpret_cast<char*>(&min_val_), sizeof(float));
                file.read(reinterpret_cast<char*>(&max_val_), sizeof(float));
                histogram_.resize(num_bins_);
                file.read(reinterpret_cast<char*>(histogram_.data()), num_bins_ * sizeof(uint32_t));
            }
        }

        MSECalibrator::MSECalibrator(bool symmetric)
            : symmetric_(symmetric),
              min_val_(0.0f),
              max_val_(0.0f),
              best_scale_(1.0f),
              best_zero_point_(0.0f) {}

        CalibrationResult MSECalibrator::calibrate(const float* data, size_t n, cudaStream_t stream) {
            float d_min, d_max;
            cuda::compute_minmax(data, n, &d_min, &d_max, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

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
            : channel_dim_(channel_dim),
              symmetric_(symmetric) {}

        CalibrationResult PerChannelCalibrator::calibrate(const float* data, size_t n, cudaStream_t stream) {
            float d_min, d_max;
            cuda::compute_minmax(data, n, &d_min, &d_max, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float scale, zero_point;
            if (symmetric_) {
                float abs_max = std::max(std::abs(d_min), std::abs(d_max));
                scale = abs_max / 127.0f;
                zero_point = 0.0f;
            } else {
                scale = (d_max - d_min) / 254.0f;
                zero_point = -d_min / scale - 127.0f;
            }

            scale = stabilized_scale(scale, d_min, d_max);
            if (!symmetric_) {
                zero_point = -d_min / scale - 127.0f;
                if (!std::isfinite(zero_point)) {
                    zero_point = 0.0f;
                } else {
                    zero_point = std::clamp(zero_point, -127.0f, 127.0f);
                }
            }

            return CalibrationResult(scale, zero_point, symmetric_, d_min, d_max);
        }

        std::vector<CalibrationResult> PerChannelCalibrator::calibrate_per_channel(const float* data, const std::vector<int>& shape, cudaStream_t stream) {
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

    }  // namespace quantize
}  // namespace nova
