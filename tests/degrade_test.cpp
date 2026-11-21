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

// should_degrade() compared precision_level values with `>`, but the enum counts down
// in precision: high = 0, medium = 1, low = 2. So a larger value means LESS precision,
// and the comparison asked "is the current precision worse than the minimum acceptable"
// -- then answered it with the name "should degrade".
//
// Concretely, with min_acceptable_precision = medium:
//
//     current high   (0) -> 0 > 1 = false    "no degradation needed"   -- right
//     current medium (1) -> 1 > 1 = false    "no degradation needed"   -- right
//     current low    (2) -> 2 > 1 = TRUE     "should degrade"          -- but it is
//                                            already at the lowest level
//
// Both existing cases stop at medium, which is why this went unnoticed.
TEST_F(DegradationTest, DoesNotAskToDegradeWhenAlreadyAtTheLowestLevel) {
    auto& manager = degradation_manager::instance();

    quality_threshold threshold;
    threshold.min_acceptable_precision = precision_level::medium;
    manager.set_threshold(threshold);

    manager.trigger_degradation("lowest", precision_level::low, "floor");
    ASSERT_EQ(manager.get_precision("lowest"), precision_level::low);

    EXPECT_FALSE(manager.should_degrade("lowest")) << "precision is already at the lowest level, so there is nothing left to degrade";
}

// record_quality() had no observable effect: it discarded a passing score, and for a failing
// one it only registered the operation at high precision without keeping the score. So
// should_degrade() -- which the header's own example calls immediately after this -- gave the
// same answer for a quality of 0.0 and one of 0.99. Measured before the fix: both false.
//
// The case used to be `EXPECT_TRUE(should_degrade(x) || !should_degrade(x))`, true either way
// and therefore asserting nothing.
TEST_F(DegradationTest, RecordQualityDrivesShouldDegrade) {
    auto& manager = degradation_manager::instance();
    manager.reset();

    quality_threshold threshold;
    threshold.min_quality_score = 0.8;
    threshold.min_acceptable_precision = precision_level::low;  // precision alone never triggers
    manager.set_threshold(threshold);

    manager.record_quality("inference_op", 0.95);
    EXPECT_FALSE(manager.should_degrade("inference_op")) << "a passing score is not a degradation signal";

    manager.record_quality("inference_op", 0.5);
    EXPECT_TRUE(manager.should_degrade("inference_op")) << "a failing score must be visible to should_degrade";

    manager.record_quality("inference_op", 0.99);
    EXPECT_FALSE(manager.should_degrade("inference_op")) << "the latest score supersedes the failing one";

    // A score exactly at the threshold passes: the comparison is "below", not "at or below".
    manager.record_quality("inference_op", 0.8);
    EXPECT_FALSE(manager.should_degrade("inference_op")) << "a score equal to the threshold is acceptable";

    manager.reset();
}

// This case asserted nothing after the second score: it set up exactly the situation where
// should_degrade() should flip, and then stopped. The missing assertion is what let the
// record_quality() defect survive.
TEST_F(DegradationTest, ShouldDegradeConsidersQualityScore) {
    auto& manager = degradation_manager::instance();
    manager.reset();

    quality_threshold threshold;
    threshold.min_quality_score = 0.9;
    threshold.min_acceptable_precision = precision_level::low;
    manager.set_threshold(threshold);

    manager.record_quality("batch_op", 0.95);
    EXPECT_FALSE(manager.should_degrade("batch_op"));

    manager.record_quality("batch_op", 0.75);
    EXPECT_TRUE(manager.should_degrade("batch_op")) << "0.75 is below the 0.9 threshold";

    manager.reset();
}

// The precision part of the answer must survive the score part being added: an operation that
// was degraded past the acceptable level reports a degradation even with a passing score.
TEST_F(DegradationTest, PrecisionBelowMinimumStillOverridesAPassingScore) {
    auto& manager = degradation_manager::instance();
    manager.reset();

    quality_threshold threshold;
    threshold.min_quality_score = 0.5;
    threshold.min_acceptable_precision = precision_level::high;
    manager.set_threshold(threshold);

    manager.trigger_degradation("weak_op", precision_level::medium, "test");
    manager.record_quality("weak_op", 0.99);
    EXPECT_TRUE(manager.should_degrade("weak_op")) << "medium is weaker than the required high";

    // At the floor there is nothing to give up, even when both signals would fire.
    manager.trigger_degradation("weak_op", precision_level::low, "floor");
    manager.record_quality("weak_op", 0.1);
    EXPECT_FALSE(manager.should_degrade("weak_op")) << "there is no level below low";

    manager.reset();
}

TEST_F(DegradationTest, DegradationEventContainsAllFields) {
    auto& manager = degradation_manager::instance();

    degradation_event captured_event;
    manager.set_callback([&](const degradation_event& event) { captured_event = event; });

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
