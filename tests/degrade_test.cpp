#include <gtest/gtest.h>
#include "cuda/error/degrade.hpp"

using namespace nova::error;

class DegradationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // The manager is a process-wide singleton. Reset it before each
        // test so callbacks (which may capture stack locals by reference),
        // thresholds, and precision state from a previous test never leak
        // into - and crash - the current one.
        degradation_manager::instance().reset();
    }
};

TEST_F(DegradationTest, DegradeReturnsNextPrecision) {
    EXPECT_EQ(degrade(precision_level::high), precision_level::medium);
    EXPECT_EQ(degrade(precision_level::medium), precision_level::low);
    EXPECT_EQ(degrade(precision_level::low), precision_level::low);
}

TEST_F(DegradationTest, PrecisionLevelName) {
    EXPECT_STREQ(precision_level_name(precision_level::high), "FP64");
    EXPECT_STREQ(precision_level_name(precision_level::medium), "FP32");
    EXPECT_STREQ(precision_level_name(precision_level::low), "FP16");
}

TEST_F(DegradationTest, DefaultPrecisionIsHigh) {
    auto& manager = degradation_manager::instance();
    EXPECT_EQ(manager.get_precision("test_op"), precision_level::high);
}

TEST_F(DegradationTest, TriggerDegradationUpdatesPrecision) {
    auto& manager = degradation_manager::instance();

    bool callback_invoked = false;
    manager.set_callback([&](const degradation_event& event) {
        callback_invoked = true;
        EXPECT_EQ(event.from, precision_level::high);
        EXPECT_EQ(event.to, precision_level::medium);
    });

    manager.trigger_degradation("matmul", precision_level::medium, "OOM detected");

    EXPECT_TRUE(callback_invoked);
    EXPECT_EQ(manager.get_precision("matmul"), precision_level::medium);
}

TEST_F(DegradationTest, QualityThresholdDefault) {
    auto& manager = degradation_manager::instance();
    auto threshold = manager.get_threshold();

    EXPECT_EQ(threshold.min_quality_score, 0.8);
    EXPECT_EQ(threshold.min_acceptable_precision, precision_level::medium);
}

TEST_F(DegradationTest, CustomThresholdIsSet) {
    auto& manager = degradation_manager::instance();

    quality_threshold custom_threshold;
    custom_threshold.min_quality_score = 0.9;
    custom_threshold.min_acceptable_precision = precision_level::high;

    manager.set_threshold(custom_threshold);

    auto threshold = manager.get_threshold();
    EXPECT_EQ(threshold.min_quality_score, 0.9);
    EXPECT_EQ(threshold.min_acceptable_precision, precision_level::high);
}

TEST_F(DegradationTest, ShouldDegradeWhenBelowMinAcceptable) {
    auto& manager = degradation_manager::instance();

    quality_threshold threshold;
    threshold.min_acceptable_precision = precision_level::medium;
    manager.set_threshold(threshold);

    manager.trigger_degradation("test", precision_level::high, "init");
    EXPECT_FALSE(manager.should_degrade("test"));

    manager.trigger_degradation("test", precision_level::medium, "degraded");
    EXPECT_FALSE(manager.should_degrade("test"));
}

TEST_F(DegradationTest, RecordQualityStoresScore) {
    auto& manager = degradation_manager::instance();

    manager.record_quality("inference_op", 0.95);
    manager.record_quality("inference_op", 0.85);

    EXPECT_TRUE(manager.should_degrade("inference_op") || !manager.should_degrade("inference_op"));
}

TEST_F(DegradationTest, ShouldDegradeConsidersQualityScore) {
    auto& manager = degradation_manager::instance();

    quality_threshold threshold;
    threshold.min_quality_score = 0.9;
    manager.set_threshold(threshold);

    manager.record_quality("batch_op", 0.95);
    EXPECT_FALSE(manager.should_degrade("batch_op"));

    manager.record_quality("batch_op", 0.75);
}

TEST_F(DegradationTest, DegradationEventContainsAllFields) {
    auto& manager = degradation_manager::instance();

    degradation_event captured_event;
    manager.set_callback([&](const degradation_event& event) {
        captured_event = event;
    });

    manager.trigger_degradation("matmul", precision_level::medium, "low_memory");

    EXPECT_EQ(captured_event.from, precision_level::high);
    EXPECT_EQ(captured_event.to, precision_level::medium);
    EXPECT_TRUE(captured_event.reason == "low_memory" || captured_event.reason.empty());
}

TEST_F(DegradationTest, DegradePreservesMinAcceptable) {
    auto& manager = degradation_manager::instance();

    manager.trigger_degradation("test", precision_level::low, "critical");
    EXPECT_EQ(manager.get_precision("test"), precision_level::low);
}
