#include <cuda/quantize/qat.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

namespace nova {
namespace quantize {
namespace test {

class QATTest : public ::testing::Test {
protected:
    static constexpr float kTol = 0.01f;
};

TEST_F(QATTest, FakeQuantizeForward) {
    FakeQuantize::Config config;
    config.precision = Precision::INT8;
    config.scale = 0.1f;
    config.zero_point = 0.0f;

    FakeQuantize fq(config);

    float input = 1.5f;
    float output = fq.forward(input);

    EXPECT_NEAR(output, 15.0f, 1.0f);
}

TEST_F(QATTest, FakeQuantizeBackward) {
    FakeQuantize::Config config;
    config.scale = 0.1f;
    config.zero_point = 0.0f;

    FakeQuantize fq(config);

    float grad_output = 1.0f;
    float grad_input = fq.backward(grad_output, 1.5f);

    EXPECT_EQ(grad_input, grad_output);
}

TEST_F(QATTest, FakeQuantizeSte) {
    FakeQuantize::Config config;
    config.scale = 0.1f;
    config.zero_point = 0.0f;

    FakeQuantize fq(config);

    float grad_output = 2.5f;
    float grad_input1 = fq.backward(grad_output, -100.0f);
    float grad_input2 = fq.backward(grad_output, 100.0f);

    EXPECT_EQ(grad_input1, grad_output);
    EXPECT_EQ(grad_input2, grad_output);
}

TEST_F(QATTest, FakeQuantizeUpdateScale) {
    FakeQuantize::Config config;
    config.scale = 0.1f;

    FakeQuantize fq(config);
    fq.update_scale(0.2f);

    float output1 = fq.forward(1.0f);
    EXPECT_NEAR(output1, 5.0f, 0.5f);
}

TEST_F(QATTest, AMPManagerAddLayer) {
    AMPManager amp;
    amp.add_layer("encoder.layer.0", Precision::FP16);

    EXPECT_EQ(amp.num_layers(), 1u);
    EXPECT_EQ(amp.get_precision("encoder.layer.0"), Precision::FP16);
}

TEST_F(QATTest, AMPManagerSetPrecision) {
    AMPManager amp;
    amp.add_layer("encoder.layer.0", Precision::FP16);

    amp.set_precision("encoder.layer.0", Precision::INT8);

    EXPECT_EQ(amp.get_precision("encoder.layer.0"), Precision::INT8);
}

TEST_F(QATTest, AMPManagerSetScale) {
    AMPManager amp;
    amp.add_layer("encoder.layer.0", Precision::FP16);
    amp.set_scale("encoder.layer.0", 0.05f);

    auto config = amp.get_config("encoder.layer.0");
    EXPECT_NEAR(config.scale, 0.05f, 0.001f);
}

TEST_F(QATTest, AMPManagerMissingLayer) {
    AMPManager amp;
    amp.add_layer("encoder.layer.0", Precision::FP16);

    EXPECT_EQ(amp.get_precision("encoder.layer.99"), Precision::FP32);
}

TEST_F(QATTest, AMPManagerCacheRoundtrip) {
    AMPManager amp;
    amp.add_layer("layer1", Precision::FP16);
    amp.add_layer("layer2", Precision::INT8);
    amp.set_scale("layer1", 0.1f);
    amp.set_scale("layer2", 0.05f);

    amp.save_config("/tmp/amp_config.bin");

    AMPManager amp2;
    amp2.load_config("/tmp/amp_config.bin");

    EXPECT_EQ(amp2.num_layers(), 2u);
    EXPECT_EQ(amp2.get_precision("layer1"), Precision::FP16);
    EXPECT_EQ(amp2.get_precision("layer2"), Precision::INT8);
    EXPECT_NEAR(amp2.get_config("layer1").scale, 0.1f, 0.001f);
    EXPECT_NEAR(amp2.get_config("layer2").scale, 0.05f, 0.001f);
}

TEST_F(QATTest, SensitivityAnalyzer) {
    SensitivityAnalyzer analyzer;

    std::vector<float> activations = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> gradients = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    analyzer.analyze_layer("attention", activations.data(), activations.size(),
                          gradients.data(), gradients.size());

    auto sensitivity = analyzer.get_sensitivity("attention");
    EXPECT_EQ(sensitivity.name, "attention");
    EXPECT_GT(sensitivity.gradient_magnitude, 0.0f);
}

TEST_F(QATTest, SensitivityAnalyzerAutoAssign) {
    SensitivityAnalyzer analyzer;
    AMPManager amp;

    std::vector<float> acts1 = {10.0f, 20.0f};
    std::vector<float> grads1 = {1.0f, 2.0f};
    std::vector<float> acts2 = {0.1f, 0.2f};
    std::vector<float> grads2 = {0.01f, 0.02f};

    amp.add_layer("sensitive_layer", Precision::FP32);
    amp.add_layer("robust_layer", Precision::FP32);

    analyzer.analyze_layer("sensitive_layer", acts1.data(), acts1.size(),
                          grads1.data(), grads1.size());
    analyzer.analyze_layer("robust_layer", acts2.data(), acts2.size(),
                          grads2.data(), grads2.size());

    analyzer.auto_assign_precision(amp);

    EXPECT_EQ(amp.get_precision("sensitive_layer"), Precision::FP16);
}

TEST_F(QATTest, PrecisionEnumValues) {
    EXPECT_EQ(static_cast<int>(Precision::FP32), 0);
    EXPECT_EQ(static_cast<int>(Precision::FP16), 1);
    EXPECT_EQ(static_cast<int>(Precision::INT8), 2);
    EXPECT_EQ(static_cast<int>(Precision::FP8_E4M3), 3);
    EXPECT_EQ(static_cast<int>(Precision::FP8_E5M2), 4);
}

} // namespace test
} // namespace quantize
} // namespace nova
