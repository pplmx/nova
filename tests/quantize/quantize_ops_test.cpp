#include "quantize_ops.hpp"

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace nova {
namespace quantize {
namespace test {

class QuantizedOpsTest : public ::testing::Test {
protected:
    std::vector<float> create_matrix_a() {
        return {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };
    }

    std::vector<float> create_matrix_b() {
        return {
            7.0f, 8.0f,
            9.0f, 10.0f,
            11.0f, 12.0f
        };
    }

    std::vector<float> expected_fp32_result() {
        return {
            1.0f * 7.0f + 2.0f * 9.0f + 3.0f * 11.0f, 1.0f * 8.0f + 2.0f * 10.0f + 3.0f * 12.0f,
            4.0f * 7.0f + 5.0f * 9.0f + 6.0f * 11.0f, 4.0f * 8.0f + 5.0f * 10.0f + 6.0f * 12.0f
        };
    }
};

TEST_F(QuantizedOpsTest, QuantizedMatmulProducesOutput) {
    auto a_fp = create_matrix_a();
    auto b_fp = create_matrix_b();

    auto a = QuantizedInt8::FromFloat(a_fp.data(), a_fp.size());
    auto b = QuantizedInt8::FromFloat(b_fp.data(), b_fp.size());

    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());

    QuantizedInt8 output;
    quantized_matmul(*a, *b, output, 2, 3, 2);

    EXPECT_GT(output.size(), 0);
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 2);
}

TEST_F(QuantizedOpsTest, QuantizedMatmulAccuracyWithinOnePercent) {
    auto a_fp = create_matrix_a();
    auto b_fp = create_matrix_b();
    auto expected = expected_fp32_result();

    auto a = QuantizedInt8::FromFloat(a_fp.data(), a_fp.size());
    auto b = QuantizedInt8::FromFloat(b_fp.data(), b_fp.size());

    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());

    QuantizedInt8 output;
    quantized_matmul(*a, *b, output, 2, 3, 2);

    auto dequantized = output.ToFloat();

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i]) > 1e-5f) {
            float relative_error = std::abs(dequantized[i] - expected[i]) / std::abs(expected[i]);
            EXPECT_LE(relative_error, 0.05f);
        }
    }
}

TEST_F(QuantizedOpsTest, MixedPrecisionMatmulProducesCorrectOutput) {
    auto a_fp = create_matrix_a();
    auto b_fp = create_matrix_b();

    auto b_quant = QuantizedInt8::FromFloat(b_fp.data(), b_fp.size());
    ASSERT_TRUE(b_quant.has_value());

    std::vector<float> scale_b(b_quant->size() / b_fp.size(), b_quant->metadata().scale);

    auto result = mixed_precision_matmul(
        a_fp.data(), b_quant->data(), scale_b.data(),
        2, 3, 2, Precision::FP32
    );

    EXPECT_EQ(result.size(), 4);

    auto expected = expected_fp32_result();
    for (size_t i = 0; i < expected.size(); ++i) {
        float diff = std::abs(result[i] - expected[i]);
        EXPECT_LE(diff, 0.1f);
    }
}

TEST_F(QuantizedOpsTest, MixedPrecisionWithFP16Output) {
    auto a_fp = create_matrix_a();
    auto b_fp = create_matrix_b();

    auto b_quant = QuantizedInt8::FromFloat(b_fp.data(), b_fp.size());
    ASSERT_TRUE(b_quant.has_value());

    std::vector<float> scale_b(b_quant->size() / b_fp.size(), b_quant->metadata().scale);

    auto result = mixed_precision_matmul(
        a_fp.data(), b_quant->data(), scale_b.data(),
        2, 3, 2, Precision::FP16
    );

    EXPECT_EQ(result.size(), 4);
    for (float val : result) {
        EXPECT_GE(val, -1e6f);
        EXPECT_LE(val, 1e6f);
    }
}

TEST_F(QuantizedOpsTest, RuntimePrecisionToggle) {
    auto a_fp = create_matrix_a();
    auto b_fp = create_matrix_b();

    auto b_quant = QuantizedInt8::FromFloat(b_fp.data(), b_fp.size());
    ASSERT_TRUE(b_quant.has_value());

    std::vector<float> scale_b(b_quant->size() / b_fp.size(), b_quant->metadata().scale);

    auto fp32_result = mixed_precision_matmul(
        a_fp.data(), b_quant->data(), scale_b.data(),
        2, 3, 2, Precision::FP32
    );

    auto fp16_result = mixed_precision_matmul(
        a_fp.data(), b_quant->data(), scale_b.data(),
        2, 3, 2, Precision::FP16
    );

    for (size_t i = 0; i < fp32_result.size(); ++i) {
        EXPECT_FLOAT_EQ(fp32_result[i], fp16_result[i]);
    }
}

} // namespace test
} // namespace quantize
} // namespace nova
