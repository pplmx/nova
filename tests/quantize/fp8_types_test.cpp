#include <cuda/quantize/fp8_types.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>

namespace nova {
namespace quantize {
namespace test {

class FP8TypesTest : public ::testing::Test {
protected:
    static constexpr float kTol = 0.01f;

    bool is_close(float a, float b, float tol = kTol) {
        if (std::isnan(a) && std::isnan(b)) return true;
        if (std::isinf(a) && std::isinf(b)) {
            return (a > 0) == (b > 0);
        }
        return std::abs(a - b) <= tol;
    }

    float relative_error(float original, float recovered) {
        if (std::abs(original) < 1e-6f) return 0.0f;
        return std::abs(original - recovered) / std::abs(original);
    }
};

TEST_F(FP8TypesTest, E4M3ConstructionFromZero) {
    FP8E4M3 val(0.0f);
    EXPECT_EQ(val.to_bits(), 0);
    EXPECT_EQ(static_cast<float>(val), 0.0f);
}

TEST_F(FP8TypesTest, E4M3ConstructionFromPositiveValue) {
    FP8E4M3 val(1.5f);
    float recovered = static_cast<float>(val);
    EXPECT_TRUE(is_close(1.5f, recovered, 0.1f));
}

TEST_F(FP8TypesTest, E4M3ConstructionFromNegativeValue) {
    FP8E4M3 val(-2.0f);
    float recovered = static_cast<float>(val);
    EXPECT_TRUE(is_close(-2.0f, recovered, 0.1f));
}

TEST_F(FP8TypesTest, E4M3RoundtripAccuracy) {
    std::vector<float> test_values = {
        0.1f, 0.5f, 1.0f, 2.0f, 10.0f, 50.0f, 100.0f, 200.0f,
        0.01f, 0.05f, 0.5f, 1.0f, 10.0f, 50.0f, 100.0f, 200.0f
    };

    for (float v : test_values) {
        FP8E4M3 val(v);
        float recovered = static_cast<float>(val);
        float rel_err = relative_error(v, recovered);
        EXPECT_LE(rel_err, 0.1f) << "Failed for value " << v;
    }
}

TEST_F(FP8TypesTest, E4M3PositiveInfinity) {
    FP8E4M3 val(std::numeric_limits<float>::infinity());
    EXPECT_EQ(val.to_bits(), FP8E4M3::POS_INF);
    EXPECT_TRUE(std::isinf(static_cast<float>(val)));
    EXPECT_GT(static_cast<float>(val), 0);
}

TEST_F(FP8TypesTest, E4M3NegativeInfinity) {
    FP8E4M3 val(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(val.to_bits(), FP8E4M3::NEG_INF);
    EXPECT_TRUE(std::isinf(static_cast<float>(val)));
    EXPECT_LT(static_cast<float>(val), 0);
}

TEST_F(FP8TypesTest, E4M3NaN) {
    FP8E4M3 val(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(val.to_bits(), FP8E4M3::NAN_VAL);
    EXPECT_TRUE(std::isnan(static_cast<float>(val)));
}

TEST_F(FP8TypesTest, E4M3NegativeZero) {
    FP8E4M3 val(-0.0f);
    EXPECT_EQ(val.to_bits(), FP8E4M3::NEG_ZERO);
    EXPECT_EQ(static_cast<float>(val), -0.0f);
}

TEST_F(FP8TypesTest, E4M3OverflowClamping) {
    FP8E4M3 val(FP8E4M3::MAX_NORMAL * 2.0f);
    float recovered = static_cast<float>(val);
    EXPECT_TRUE(recovered <= FP8E4M3::MAX_NORMAL * 1.1f);
}

TEST_F(FP8TypesTest, E4M3SmallValueHandling) {
    float small_val = 0.001f;
    FP8E4M3 val(small_val);
    float recovered = static_cast<float>(val);
    EXPECT_GE(recovered, 0.0f);
}

TEST_F(FP8TypesTest, E4M3FromBits) {
    FP8E4M3 val = FP8E4M3::from_bits(0x3C);
    EXPECT_EQ(val.to_bits(), 0x3C);
}

TEST_F(FP8TypesTest, E5M2ConstructionFromZero) {
    FP8E5M2 val(0.0f);
    EXPECT_EQ(val.to_bits(), 0);
    EXPECT_EQ(static_cast<float>(val), 0.0f);
}

TEST_F(FP8TypesTest, E5M2ConstructionFromPositiveValue) {
    FP8E5M2 val(100.0f);
    float recovered = static_cast<float>(val);
    EXPECT_TRUE(is_close(100.0f, recovered, 1.0f));
}

TEST_F(FP8TypesTest, E5M2ConstructionFromNegativeValue) {
    FP8E5M2 val(-100.0f);
    float recovered = static_cast<float>(val);
    EXPECT_TRUE(is_close(-100.0f, recovered, 1.0f));
}

TEST_F(FP8TypesTest, E5M2RoundtripAccuracy) {
    std::vector<float> test_values = {
        1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f,
        0.1f, 0.001f, 0.0001f
    };

    for (float v : test_values) {
        FP8E5M2 val(v);
        float recovered = static_cast<float>(val);
        float rel_err = relative_error(v, recovered);
        EXPECT_LE(rel_err, 0.1f) << "Failed for value " << v;
    }
}

TEST_F(FP8TypesTest, E5M2PositiveInfinity) {
    FP8E5M2 val(std::numeric_limits<float>::infinity());
    EXPECT_EQ(val.to_bits(), FP8E5M2::POS_INF);
    EXPECT_TRUE(std::isinf(static_cast<float>(val)));
}

TEST_F(FP8TypesTest, E5M2NegativeInfinity) {
    FP8E5M2 val(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(val.to_bits(), FP8E5M2::NEG_INF);
    EXPECT_TRUE(std::isinf(static_cast<float>(val)));
}

TEST_F(FP8TypesTest, E5M2NaN) {
    FP8E5M2 val(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(val.to_bits(), FP8E5M2::NAN_VAL);
    EXPECT_TRUE(std::isnan(static_cast<float>(val)));
}

TEST_F(FP8TypesTest, E5M2NegativeZero) {
    FP8E5M2 val(-0.0f);
    EXPECT_EQ(val.to_bits(), FP8E5M2::NEG_ZERO);
    EXPECT_EQ(static_cast<float>(val), -0.0f);
}

TEST_F(FP8TypesTest, E5M2FromBits) {
    FP8E5M2 val = FP8E5M2::from_bits(0x3C);
    EXPECT_EQ(val.to_bits(), 0x3C);
}

TEST_F(FP8TypesTest, E4M3IsFP8TypeTrait) {
    static_assert(is_fp8_type_v<FP8E4M3>, "FP8E4M3 should be an FP8 type");
    static_assert(!is_fp8_type_v<float>, "float should not be an FP8 type");
    static_assert(!is_fp8_type_v<int>, "int should not be an FP8 type");
}

TEST_F(FP8TypesTest, E5M2IsFP8TypeTrait) {
    static_assert(is_fp8_type_v<FP8E5M2>, "FP8E5M2 should be an FP8 type");
    static_assert(!is_fp8_type_v<double>, "double should not be an FP8 type");
}

TEST_F(FP8TypesTest, E4M3WideDynamicRange) {
    FP8E4M3 small(0.01f);
    FP8E4M3 large(100.0f);

    float small_rec = static_cast<float>(small);
    float large_rec = static_cast<float>(large);

    EXPECT_LT(small_rec, large_rec);
}

TEST_F(FP8TypesTest, E5M2WideDynamicRange) {
    FP8E5M2 small(0.0001f);
    FP8E5M2 large(10000.0f);

    float small_rec = static_cast<float>(small);
    float large_rec = static_cast<float>(large);

    EXPECT_LT(small_rec, large_rec);
}

TEST_F(FP8TypesTest, E4M3E5M2Comparison) {
    FP8E4M3 e4m3(100.0f);
    FP8E5M2 e5m2(100.0f);

    float e4m3_rec = static_cast<float>(e4m3);
    float e5m2_rec = static_cast<float>(e5m2);

    float e4m3_rel_err = relative_error(100.0f, e4m3_rec);
    float e5m2_rel_err = relative_error(100.0f, e5m2_rec);

    EXPECT_LE(e5m2_rel_err, e4m3_rel_err);
}

} // namespace test
} // namespace quantize
} // namespace nova
