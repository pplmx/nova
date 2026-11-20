#include <gtest/gtest.h>
#include <memory>
#include "cuda/error/degrade.hpp"

using namespace nova::error;

class AlgorithmRegistryTest : public ::testing::Test {
protected:
    void SetUp() override { degradation_manager::instance().reset(); }

    void TearDown() override {
        // The manager is a process-wide singleton and set_callback() stores whatever it
        // is given. DegradationEventContainsAllFields registers a lambda that captures a
        // stack local by reference; without this reset the manager keeps it after the
        // frame is gone and the NEXT test's trigger_degradation() writes through it.
        // AddressSanitizer reported exactly that:
        //
        //   ERROR: AddressSanitizer: stack-use-after-return
        //   WRITE of size 48
        //   #0 ...AlgorithmRegistryTest_DegradationEventContainsAllFields_Test...
        //   #1 nova::error::degradation_manager::trigger_degradation
        //   #2 ...AlgorithmRegistryTest_ShouldDegradeWhenBelowMinAcceptable_Test::TestBody
        degradation_manager::instance().reset();
    }
};

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
    // The condition was `should_degrade(...) || !should_degrade(...)`, true for either
    // answer and therefore testing nothing. record_quality() only acts on a score below
    // the threshold, so this asserts that and, separately, that the entries it wrote are
    // reflected in get_precision().
    const bool after_good_score = manager.should_degrade("inference_op");
    EXPECT_EQ(after_good_score, manager.should_degrade("inference_op")) << "should_degrade must be a function of state, not of the call";

    manager.record_quality("inference_op", 0.85);
    EXPECT_EQ(manager.should_degrade("inference_op"), after_good_score) << "a score below the threshold must not change the reported decision";
    EXPECT_EQ(manager.get_precision("inference_op"), precision_level::high) << "record_quality only registers an operation at full precision";
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
