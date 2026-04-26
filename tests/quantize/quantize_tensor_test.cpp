#include "quantize_tensor.hpp"

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace nova {
namespace quantize {
namespace test {

class QuantizationTest : public ::testing::Test {
protected:
    std::vector<float> create_test_data() {
        return {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, 0.5f, 0.0f};
    }
};

TEST_F(QuantizationTest, Int8QuantizationCreatesTensor) {
    auto data = create_test_data();
    auto quantized = QuantizedInt8::FromFloat(data.data(), data.size());

    ASSERT_TRUE(quantized.has_value());
    EXPECT_EQ(quantized->size(), data.size());
    EXPECT_EQ(quantized->metadata().num_bits, 8);
}

TEST_F(QuantizationTest, Int8QuantizationHasCorrectScale) {
    auto data = create_test_data();
    auto quantized = QuantizedInt8::FromFloat(data.data(), data.size());

    ASSERT_TRUE(quantized.has_value());
    EXPECT_GT(quantized->metadata().scale, 0.0f);
}

TEST_F(QuantizationTest, Int8DequantizationRecoversValues) {
    auto data = create_test_data();
    auto quantized = QuantizedInt8::FromFloat(data.data(), data.size());

    ASSERT_TRUE(quantized.has_value());
    auto recovered = quantized->ToFloat();

    ASSERT_EQ(recovered.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        float diff = std::abs(recovered[i] - data[i]);
        EXPECT_LE(diff, 0.1f);
    }
}

TEST_F(QuantizationTest, Int8CustomScale) {
    auto data = create_test_data();
    auto quantized = QuantizedInt8::FromFloat(data.data(), data.size(), 0.02f);

    ASSERT_TRUE(quantized.has_value());
    EXPECT_FLOAT_EQ(quantized->metadata().scale, 0.02f);
}

TEST_F(QuantizationTest, FP16QuantizationCreatesTensor) {
    auto data = create_test_data();
    auto quantized = QuantizedFP16::FromFloat(data.data(), data.size());

    ASSERT_TRUE(quantized.has_value());
    EXPECT_EQ(quantized->size(), data.size());
    EXPECT_EQ(quantized->metadata().num_bits, 16);
}

TEST_F(QuantizationTest, FP16DequantizationAccuracy) {
    auto data = create_test_data();
    auto quantized = QuantizedFP16::FromFloat(data.data(), data.size());

    ASSERT_TRUE(quantized.has_value());
    auto recovered = quantized->ToFloat();

    ASSERT_EQ(recovered.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        float relative_diff = std::abs(recovered[i] - data[i]) / (std::abs(data[i]) + 1e-6f);
        EXPECT_LE(relative_diff, 0.01f);
    }
}

TEST_F(QuantizationTest, QuantizationMetadataAccessible) {
    auto data = create_test_data();
    auto int8_quant = QuantizedInt8::FromFloat(data.data(), data.size());

    ASSERT_TRUE(int8_quant.has_value());
    EXPECT_EQ(int8_quant->metadata().mode, QuantizationMode::PerTensor);
    EXPECT_EQ(int8_quant->metadata().zero_point, 0.0f);
}

TEST_F(QuantizationTest, ShapePreserved) {
    std::vector<float> data(24, 1.0f);
    auto quantized = QuantizedInt8::FromFloat(data.data(), data.size());

    ASSERT_TRUE(quantized.has_value());
    ASSERT_EQ(quantized->shape().size(), 1);
    EXPECT_EQ(quantized->shape()[0], 24);
}

} // namespace test
} // namespace quantize
} // namespace nova
