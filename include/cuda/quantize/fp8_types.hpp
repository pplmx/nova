#ifndef NOVA_CUDA_QUANTIZE_FP8_TYPES_HPP
#define NOVA_CUDA_QUANTIZE_FP8_TYPES_HPP

#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <type_traits>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace nova {
namespace quantize {

class FP8E4M3 {
public:
    uint8_t value;

    __host__ __device__ FP8E4M3() : value(0) {}

    __host__ __device__ explicit FP8E4M3(float f) : value(float_to_fp8(f)) {}

    __host__ __device__ explicit operator float() const { return fp8_to_float(value); }

    __host__ __device__ static FP8E4M3 from_bits(uint8_t bits) {
        FP8E4M3 result;
        result.value = bits;
        return result;
    }

    __host__ __device__ uint8_t to_bits() const { return value; }

    static constexpr uint8_t POS_INF = 0x7C;
    static constexpr uint8_t NEG_INF = 0xFC;
    static constexpr uint8_t NAN_VAL = 0x7D;
    static constexpr uint8_t NEG_ZERO = 0x80;

    static constexpr float MAX_NORMAL = 240.0f;
    static constexpr float MIN_NORMAL = 0.015625f;
    static constexpr int EXP_BIAS = 7;
    static constexpr int MANTISSA_BITS = 3;

private:
    __host__ __device__ static uint8_t float_to_fp8(float f) {
        if (f == 0.0f) return 0;

        uint32_t sign = (f < 0) ? 0x80 : 0;
        f = std::abs(f);

        if (std::isinf(f)) return sign | POS_INF;
        if (std::isnan(f)) return NAN_VAL;

        int exp;
        float mantissa = std::frexp(f, &exp);
        exp += EXP_BIAS - 1;

        if (exp <= 0) {
            uint8_t sign_bit = sign ? 0x80 : 0;
            if (exp < -MANTISSA_BITS) return sign_bit;
            int shift = -(exp - 1);
            uint32_t m = static_cast<uint32_t>(mantissa * 2.0f * (1 << (MANTISSA_BITS + shift)));
            return sign_bit | (m & 0x7F);
        }

        if (exp >= 15) {
            return sign | POS_INF;
        }

        mantissa = mantissa * 2.0f - 1.0f;
        mantissa = mantissa * (1 << MANTISSA_BITS);

        uint8_t exp_bits = static_cast<uint8_t>(exp) << MANTISSA_BITS;
        uint8_t mantissa_bits = static_cast<uint8_t>(std::round(mantissa)) & ((1 << MANTISSA_BITS) - 1);

        return sign | exp_bits | mantissa_bits;
    }

    __host__ __device__ static float fp8_to_float(uint8_t bits) {
        if (bits == 0) return 0.0f;
        if (bits == NEG_ZERO) return -0.0f;
        if (bits == POS_INF) return std::numeric_limits<float>::infinity();
        if (bits == NEG_INF) return -std::numeric_limits<float>::infinity();
        if (bits == NAN_VAL) return std::numeric_limits<float>::quiet_NaN();

        bool negative = (bits & 0x80) != 0;
        int exp_bits = (bits >> MANTISSA_BITS) & 0x0F;
        uint8_t mantissa_bits = bits & ((1 << MANTISSA_BITS) - 1);

        int exp = exp_bits - EXP_BIAS;
        float mantissa = 1.0f + static_cast<float>(mantissa_bits) / (1 << MANTISSA_BITS);

        float result = mantissa * std::ldexp(1.0f, exp);
        return negative ? -result : result;
    }
};

class FP8E5M2 {
public:
    uint8_t value;

    __host__ __device__ FP8E5M2() : value(0) {}

    __host__ __device__ explicit FP8E5M2(float f) : value(float_to_fp8(f)) {}

    __host__ __device__ explicit operator float() const { return fp8_to_float(value); }

    __host__ __device__ static FP8E5M2 from_bits(uint8_t bits) {
        FP8E5M2 result;
        result.value = bits;
        return result;
    }

    __host__ __device__ uint8_t to_bits() const { return value; }

    static constexpr uint8_t POS_INF = 0x7C;
    static constexpr uint8_t NEG_INF = 0xFC;
    static constexpr uint8_t NAN_VAL = 0x7D;
    static constexpr uint8_t NEG_ZERO = 0x80;

    static constexpr float MAX_NORMAL = 57344.0f;
    static constexpr float MIN_NORMAL = 5.9604644775390625e-5f;
    static constexpr int EXP_BIAS = 15;
    static constexpr int MANTISSA_BITS = 2;

private:
    __host__ __device__ static uint8_t float_to_fp8(float f) {
        if (f == 0.0f) return 0;

        uint32_t sign = (f < 0) ? 0x80 : 0;
        f = std::abs(f);

        if (std::isinf(f)) return sign | POS_INF;
        if (std::isnan(f)) return NAN_VAL;

        int exp;
        float mantissa = std::frexp(f, &exp);
        exp += EXP_BIAS - 1;

        if (exp <= 0) {
            uint8_t sign_bit = sign ? 0x80 : 0;
            if (exp < -MANTISSA_BITS) return sign_bit;
            int shift = -(exp - 1);
            uint32_t m = static_cast<uint32_t>(mantissa * 2.0f * (1 << (MANTISSA_BITS + shift)));
            return sign_bit | (m & 0x7F);
        }

        if (exp >= 31) {
            return sign | POS_INF;
        }

        mantissa = mantissa * 2.0f - 1.0f;
        mantissa = mantissa * (1 << MANTISSA_BITS);

        uint8_t exp_bits = static_cast<uint8_t>(exp) << MANTISSA_BITS;
        uint8_t mantissa_bits = static_cast<uint8_t>(std::round(mantissa)) & ((1 << MANTISSA_BITS) - 1);

        return sign | exp_bits | mantissa_bits;
    }

    __host__ __device__ static float fp8_to_float(uint8_t bits) {
        if (bits == 0) return 0.0f;
        if (bits == NEG_ZERO) return -0.0f;
        if (bits == POS_INF) return std::numeric_limits<float>::infinity();
        if (bits == NEG_INF) return -std::numeric_limits<float>::infinity();
        if (bits == NAN_VAL) return std::numeric_limits<float>::quiet_NaN();

        bool negative = (bits & 0x80) != 0;
        int exp_bits = (bits >> MANTISSA_BITS) & 0x1F;
        uint8_t mantissa_bits = bits & ((1 << MANTISSA_BITS) - 1);

        int exp = exp_bits - EXP_BIAS;
        float mantissa = 1.0f + static_cast<float>(mantissa_bits) / (1 << MANTISSA_BITS);

        float result = mantissa * std::ldexp(1.0f, exp);
        return negative ? -result : result;
    }
};

template<typename T>
struct is_fp8_type : std::false_type {};

template<>
struct is_fp8_type<FP8E4M3> : std::true_type {};

template<>
struct is_fp8_type<FP8E5M2> : std::true_type {};

template<typename T>
inline constexpr bool is_fp8_type_v = is_fp8_type<T>::value;

} // namespace quantize
} // namespace nova

#endif // NOVA_CUDA_QUANTIZE_FP8_TYPES_HPP
