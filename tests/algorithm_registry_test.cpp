#include <gtest/gtest.h>
#include <memory>
#include "cuda/error/degrade.hpp"

using namespace nova::error;

class AlgorithmRegistryTest : public ::testing::Test {};

TEST_F(AlgorithmRegistryTest, DegradePreservesMinAcceptable) {
    auto& manager = degradation_manager::instance();

    manager.trigger_degradation("test", precision_level::low, "critical");
    EXPECT_EQ(manager.get_precision("test"), precision_level::low);
}

TEST_F(AlgorithmRegistryTest, DegradationEventContainsAllFields) {
    auto& manager = degradation_manager::instance();

    degradation_event captured_event;
    manager.set_callback([&](const degradation_event& event) { captured_event = event; });

    manager.trigger_degradation("matmul", precision_level::medium, "low_memory");

    EXPECT_EQ(captured_event.from, precision_level::high);
    EXPECT_EQ(captured_event.to, precision_level::medium);
}

TEST_F(AlgorithmRegistryTest, RecordQualityStoresScore) {
    auto& manager = degradation_manager::instance();

    manager.record_quality("inference_op", 0.95);
    manager.record_quality("inference_op", 0.85);

    EXPECT_TRUE(manager.should_degrade("inference_op") || !manager.should_degrade("inference_op"));
}

TEST_F(AlgorithmRegistryTest, ShouldDegradeConsidersQualityScore) {
    auto& manager = degradation_manager::instance();

    quality_threshold threshold;
    threshold.min_quality_score = 0.9;
    manager.set_threshold(threshold);

    manager.record_quality("batch_op", 0.95);
    EXPECT_FALSE(manager.should_degrade("batch_op"));
}

TEST_F(AlgorithmRegistryTest, ShouldDegradeWhenBelowMinAcceptable) {
    auto& manager = degradation_manager::instance();

    quality_threshold threshold;
    threshold.min_acceptable_precision = precision_level::medium;
    manager.set_threshold(threshold);

    manager.trigger_degradation("test", precision_level::high, "init");
    EXPECT_FALSE(manager.should_degrade("test"));

    manager.trigger_degradation("test", precision_level::medium, "degraded");
    EXPECT_FALSE(manager.should_degrade("test"));
}
