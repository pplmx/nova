#ifndef NOVA_CUDA_QUANTIZE_TENSOR_HPP
#define NOVA_CUDA_QUANTIZE_TENSOR_HPP

#include <nova/memory/buffer.hpp>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <optional>

namespace nova {
namespace quantize {

enum class QuantizationMode { PerTensor, PerChannel };

struct QuantizationMetadata {
    QuantizationMode mode;
    float scale;
    float zero_point;
    int num_bits;
};

struct alignas(2) float16 {
    uint16_t value;

    float16() = default;
    explicit float16(float f);

    operator float() const;
};

float16::float16(float f) {
    uint32_t bits = 0;
    if (f == 0.0f) {
        value = 0;
        return;
    }

    bool negative = f < 0;
    f = std::abs(f);

    int exp;
    float mantissa = std::frexp(f, &exp);

    exp += 15;
    if (exp <= 0) {
        value = negative ? 0x8000 : 0;
        return;
    }
    if (exp >= 31) {
        value = negative ? 0xFC00 : 0x7C00;
        return;
    }

    mantissa = mantissa * 2.0f - 1.0f;
    mantissa = mantissa * 1024.0f;

    uint16_t sign = negative ? 0x8000 : 0;
    uint16_t exp_bits = static_cast<uint16_t>(exp) << 10;
    uint16_t mantissa_bits = static_cast<uint16_t>(mantissa);

    value = sign | exp_bits | mantissa_bits;
}

float16::operator float() const {
    if (value == 0) return 0.0f;

    uint16_t sign = (value & 0x8000) >> 15;
    uint16_t exp_bits = (value & 0x7C00) >> 10;
    uint16_t mantissa_bits = value & 0x03FF;

    int exp = static_cast<int>(exp_bits) - 15;
    float mantissa = 1.0f + static_cast<float>(mantissa_bits) / 1024.0f;

    float result = mantissa * std::ldexp(1.0f, exp);
    return sign ? -result : result;
}

template<typename T>
class QuantizedTensor {
public:
    QuantizedTensor() = default;

    QuantizedTensor(std::vector<T> data, QuantizationMetadata meta, std::vector<int> shape)
        : data_(std::move(data))
        , metadata_(meta)
        , shape_(std::move(shape)) {}

    static std::optional<QuantizedTensor<int8_t>> FromFloat(
        const float* data, size_t size, float scale = 0.0f);

    static std::optional<QuantizedTensor<float16>> FromFloat(
        const float* data, size_t size);

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

template<>
std::optional<QuantizedTensor<int8_t>> QuantizedTensor<int8_t>::FromFloat(
    const float* data, size_t size, float scale) {

    if (size == 0) return std::nullopt;

    if (scale <= 0.0f) {
        float max_val = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            max_val = std::max(max_val, std::abs(data[i]));
        }
        scale = max_val / 127.0f;
    }

    if (scale == 0.0f) scale = 1.0f;

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

template<>
std::optional<QuantizedTensor<float16>> QuantizedTensor<float16>::FromFloat(
    const float* data, size_t size) {

    if (size == 0) return std::nullopt;

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

template<typename T>
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

} // namespace quantize
} // namespace nova

#endif // NOVA_CUDA_QUANTIZE_TENSOR_HPP
