#ifndef NOVA_CUDA_QUANTIZE_OPS_HPP
#define NOVA_CUDA_QUANTIZE_OPS_HPP

#include <nova/quantize/quantize_tensor.hpp>
#include <vector>
#include <cmath>

namespace nova {
namespace quantize {

enum class Precision { FP32, FP16, INT8 };

class QuantizedMatmul {
public:
    static void forward(const QuantizedInt8& a,
                        const QuantizedInt8& b,
                        QuantizedInt8& output,
                        int m, int k, int n);

    static std::vector<float> mixed_precision(const float* a,
                                               const int8_t* b,
                                               const float* scale_b,
                                               int m, int k, int n,
                                               Precision output_precision);
};

void QuantizedMatmul::forward(const QuantizedInt8& a,
                               const QuantizedInt8& b,
                               QuantizedInt8& output,
                               int m, int k, int n) {
    const int8_t* a_data = a.data();
    const int8_t* b_data = b.data();
    std::vector<int8_t> output_data(m * n);

    float scale_a = a.metadata().scale;
    float scale_b = b.metadata().scale;
    float output_scale = scale_a * scale_b;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int32_t sum = 0;
            for (int p = 0; p < k; ++p) {
                sum += static_cast<int32_t>(a_data[i * k + p]) *
                       static_cast<int32_t>(b_data[p * n + j]);
            }
            float quantized = sum * output_scale;

            quantized = std::max(-127.0f, std::min(127.0f, quantized));
            output_data[i * n + j] = static_cast<int8_t>(std::round(quantized));
        }
    }

    output = QuantizedInt8(std::move(output_data), a.metadata(), {m, n});
}

std::vector<float> QuantizedMatmul::mixed_precision(const float* a,
                                                     const int8_t* b,
                                                     const float* scale_b,
                                                     int m, int k, int n,
                                                     Precision output_precision) {
    std::vector<float> output(m * n, 0.0f);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                float b_val = static_cast<float>(b[p * n + j]) * scale_b[p];
                sum += a[i * k + p] * b_val;
            }
            output[i * n + j] = sum;
        }
    }

    return output;
}

void quantized_matmul(const QuantizedInt8& a,
                      const QuantizedInt8& b,
                      QuantizedInt8& output,
                      int m, int k, int n) {
    QuantizedMatmul::forward(a, b, output, m, k, n);
}

std::vector<float> mixed_precision_matmul(const float* a,
                                           const int8_t* b,
                                           const float* scale_b,
                                           int m, int k, int n,
                                           Precision precision = Precision::FP32) {
    return QuantizedMatmul::mixed_precision(a, b, scale_b, m, k, n, precision);
}

} // namespace quantize
} // namespace nova

#endif // NOVA_CUDA_QUANTIZE_OPS_HPP
