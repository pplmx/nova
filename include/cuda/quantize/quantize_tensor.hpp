/**
 * @file quantize_tensor.hpp
 * @brief Tensor quantization types and utilities
 * @defgroup quantize_tensor Tensor Quantization
 * @ingroup quantize
 *
 * Provides quantization modes, metadata structures, and type definitions
 * for numerical precision reduction in deep learning workloads.
 *
 * Supported modes:
 * - PerTensor: Single scale for entire tensor
 * - PerChannel: Scale per output channel
 *
 * Example usage:
 * @code
 * QuantizationMetadata meta{
 *     .mode = QuantizationMode::PerTensor,
 *     .scale = 0.1f,
 *     .zero_point = 128,
 *     .num_bits = 8
 * };
 * @endcode
 *
 * @see calibrator.hpp For scale calibration
 * @see quantize_ops.hpp For quantization operations
 */

#ifndef NOVA_CUDA_QUANTIZE_TENSOR_HPP
#define NOVA_CUDA_QUANTIZE_TENSOR_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace nova {
    namespace quantize {

        /**
         * @brief Quantization granularity mode
         * @enum QuantizationMode
         * @ingroup quantize_tensor
         */
        enum class QuantizationMode { PerTensor, PerChannel };

        /**
         * @brief Metadata for quantized tensor
         * @struct QuantizationMetadata
         * @ingroup quantize_tensor
         */
        struct QuantizationMetadata {
            /** @brief Quantization granularity */
            QuantizationMode mode;

            /** @brief Scale factor */
            float scale;

            /** @brief Zero point for asymmetric quantization */
            float zero_point;

            /** @brief Number of bits per value */
            int num_bits;
        };

        /**
         * @brief IEEE 754 half-precision floating point
         * @struct float16
         * @ingroup quantize_tensor
         */
        struct alignas(2) float16 {
            /** @brief Raw 16-bit value */
            uint16_t value;

            float16() = default;
            explicit float16(float f);

            operator float() const;
        };

        inline float16::float16(float f) {
            const uint16_t sign = std::signbit(f) ? 0x8000 : 0;
            if (f == 0.0f) {
                value = sign;
                return;
            }

            f = std::abs(f);

            // frexp returns f == mantissa * 2^exp with mantissa in [0.5, 1), while IEEE
            // half stores 1.<frac> * 2^(exp_field - 15). Converting between the two
            // means the field is (exp - 1) + 15, not exp + 15: the previous code was
            // off by one and encoded 1.0f as 0x4000, which decodes to 2.0f.
            int exp = 0;
            float mantissa = std::frexp(f, &exp);
            const int exp_field = exp - 1 + 15;

            if (exp_field <= 0) {
                // Underflows the smallest normal (2^-14); flush to signed zero rather
                // than pretend a subnormal was produced.
                value = sign;
                return;
            }
            if (exp_field >= 31) {
                value = sign | 0x7C00;  // overflow to infinity
                return;
            }

            const float frac = (mantissa * 2.0f - 1.0f) * 1024.0f;
            value = static_cast<uint16_t>(sign | (static_cast<uint16_t>(exp_field) << 10) | (static_cast<uint16_t>(std::lround(frac)) & 0x03FF));
        }

        inline float16::operator float() const {
            if (value == 0) {
                return 0.0f;
            }

            uint16_t sign = (value & 0x8000) >> 15;
            uint16_t exp_bits = (value & 0x7C00) >> 10;
            uint16_t mantissa_bits = value & 0x03FF;

            int exp = static_cast<int>(exp_bits) - 15;
            float mantissa = 1.0f + static_cast<float>(mantissa_bits) / 1024.0f;

            float result = mantissa * std::ldexp(1.0f, exp);
            return sign ? -result : result;
        }

        template <typename T>
        class QuantizedTensor {
        public:
            QuantizedTensor() = default;

            QuantizedTensor(std::vector<T> data, QuantizationMetadata meta, std::vector<int> shape)
                : data_(std::move(data)),
                  metadata_(meta),
                  shape_(std::move(shape)) {}

            std::vector<float> ToFloat() const;

            const T* data() const { return data_.data(); }
            T* data() { return data_.data(); }

            const QuantizationMetadata& metadata() const { return metadata_; }
            const std::vector<int>& shape() const { return shape_; }

            size_t size() const { return data_.size(); }

        private:
            std::vector<T> data_;
            QuantizationMetadata metadata_;
            std::vector<int> shape_;
        };

        /**
         * @brief Quantize float values to int8 with a per-tensor scale
         * @param data Source values; the span carries both the pointer and length
         * @param scale Per-tensor scale, or <= 0 to derive it from the max |value|
         * @return Quantized tensor, or nullopt for an empty or non-finite input
         *
         * @note A free function rather than a static member of QuantizedTensor: the two
         *       conversions have different return types, so as members of one class
         *       template they formed an overload set that a plain `std::vector<float>`
         *       argument matched ambiguously.
         */
        inline std::optional<QuantizedTensor<int8_t>> quantize_int8_from_float(std::span<const float> data, float scale = 0.0f) {
            const size_t size = data.size();
            if (size == 0) {
                return std::nullopt;
            }

            // A NaN or an infinity in the input poisons max_val, and every later clamp
            // then produces a constant tensor that looks quantized. Reject it instead.
            for (const float value : data) {
                if (!std::isfinite(value)) {
                    return std::nullopt;
                }
            }

            if (scale <= 0.0f) {
                float max_val = 0.0f;
                for (const float value : data) {
                    max_val = std::max(max_val, std::abs(value));
                }
                scale = max_val / 127.0f;
            }

            if (scale == 0.0f) {
                scale = 1.0f;
            }

            std::vector<int8_t> quantized(size);
            for (size_t i = 0; i < size; ++i) {
                float normalized = data[i] / scale;
                float rounded = std::round(normalized);
                rounded = std::max(-127.0f, std::min(127.0f, rounded));
                quantized[i] = static_cast<int8_t>(rounded);
            }

            QuantizationMetadata meta;
            meta.mode = QuantizationMode::PerTensor;
            meta.scale = scale;
            meta.zero_point = 0.0f;
            meta.num_bits = 8;

            return QuantizedTensor<int8_t>(std::move(quantized), meta, {static_cast<int>(size)});
        }

        /**
         * @brief Convert float values to half precision
         * @param data Source values
         * @return Quantized tensor, or nullopt for an empty input
         */
        inline std::optional<QuantizedTensor<float16>> quantize_fp16_from_float(std::span<const float> data) {
            const size_t size = data.size();
            if (size == 0) {
                return std::nullopt;
            }

            std::vector<float16> quantized(size);
            for (size_t i = 0; i < size; ++i) {
                quantized[i] = float16(data[i]);
            }

            QuantizationMetadata meta;
            meta.mode = QuantizationMode::PerTensor;
            meta.scale = 1.0f;
            meta.zero_point = 0.0f;
            meta.num_bits = 16;

            return QuantizedTensor<float16>(std::move(quantized), meta, {static_cast<int>(size)});
        }

        template <typename T>
        std::vector<float> QuantizedTensor<T>::ToFloat() const {
            std::vector<float> result(size());

            if constexpr (std::is_same_v<T, int8_t>) {
                for (size_t i = 0; i < size(); ++i) {
                    result[i] = static_cast<float>(data_[i]) * metadata_.scale;
                }
            } else if constexpr (std::is_same_v<T, float16>) {
                for (size_t i = 0; i < size(); ++i) {
                    result[i] = static_cast<float>(data_[i]);
                }
            }

            return result;
        }

        using QuantizedInt8 = QuantizedTensor<int8_t>;
        using QuantizedFP16 = QuantizedTensor<float16>;

    }  // namespace quantize
}  // namespace nova

#endif  // NOVA_CUDA_QUANTIZE_TENSOR_HPP
