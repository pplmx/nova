/**
 * @file cross_entropy_validation_test.cpp
 * @brief Contracts for the cross-entropy entry point's input validation
 *
 * include/cuda/neural/loss/loss_functions.h computes the mean cross-entropy over
 * HOST buffers (the device kernels in that file are documented as dead code, and
 * the existing fixture passes std::vector data pointers). Two inputs were used
 * without validation, and both produced results a caller could not act on:
 *
 * - targets[b] was used directly as an index into log_probs:
 *
 *       int target_class = targets[b];
 *       float loss = -log_probs[b * num_classes + target_class];
 *
 *   Measured with target 99 and 3 classes: the process aborted with
 *   "dumped core" — an out-of-bounds read.
 *
 * - batch_size == 0 divided by zero for the mean and returned -nan. num_classes
 *   == 0 left every max_logit at -inf, making the loss meaningless and also
 *   crashing.
 *
 * Both are now rejected with std::invalid_argument naming the offending value, and
 * an empty batch returns 0 rather than NaN.
 */

#include <cuda/neural/loss/loss_functions.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::neural::loss::test {

namespace {

using cuda::neural::loss::CrossEntropyConfig;
using cuda::neural::loss::cross_entropy_loss;

/// Runs `call` and returns the exception message, or an empty string.
template <typename Callable>
std::string message_from(Callable&& call) {
    try {
        call();
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

}  // namespace

// Regression: this aborted the process with an out-of-bounds read.
TEST(CrossEntropyValidationTest, TargetAtOrAboveNumClassesIsRejected) {
    constexpr int batch = 2, classes = 3;
    const std::vector<float> logits(batch * classes, 0.5f);
    const std::vector<int> targets = {0, 99};
    std::vector<float> output(batch, 0.0f);

    const std::string message = message_from([&] {
        (void)cross_entropy_loss(logits.data(), targets.data(), output.data(),
                                 batch, classes, {}, nullptr);
    });

    EXPECT_FALSE(message.empty()) << "an out-of-range label must not be indexed";
    EXPECT_NE(message.find("99"), std::string::npos)
        << "the message should name the offending target: " << message;
    EXPECT_NE(message.find("0, 3"), std::string::npos)
        << "the message should state the valid range: " << message;
}

TEST(CrossEntropyValidationTest, NegativeTargetIsRejected) {
    constexpr int batch = 1, classes = 3;
    const std::vector<float> logits(batch * classes, 0.5f);
    const std::vector<int> targets = {-1};
    std::vector<float> output(batch, 0.0f);

    const std::string message = message_from([&] {
        (void)cross_entropy_loss(logits.data(), targets.data(), output.data(),
                                 batch, classes, {}, nullptr);
    });

    EXPECT_FALSE(message.empty()) << "a negative label would index before the buffer";
    EXPECT_NE(message.find("-1"), std::string::npos) << message;
}

// Regression: this returned -nan.
TEST(CrossEntropyValidationTest, EmptyBatchReturnsZeroNotNaN) {
    const float result = cross_entropy_loss(nullptr, nullptr, nullptr, 0, 4, {},
                                            nullptr);

    EXPECT_TRUE(std::isfinite(result)) << "an empty batch produced " << result;
    EXPECT_FLOAT_EQ(result, 0.0f)
        << "the mean loss over no samples is zero by convention";
}

// Regression: this also crashed.
TEST(CrossEntropyValidationTest, ZeroClassesIsRejected) {
    const std::string message = message_from([] {
        (void)cross_entropy_loss(nullptr, nullptr, nullptr, 2, 0, {}, nullptr);
    });

    EXPECT_FALSE(message.empty())
        << "a class count of zero leaves the softmax denominator at zero";
    EXPECT_NE(message.find("num_classes"), std::string::npos) << message;
}

TEST(CrossEntropyValidationTest, NegativeBatchIsRejected) {
    const std::string message = message_from([] {
        (void)cross_entropy_loss(nullptr, nullptr, nullptr, -1, 4, {}, nullptr);
    });
    EXPECT_FALSE(message.empty());
}

TEST(CrossEntropyValidationTest, ValidInputProducesAPositiveLoss) {
    // One-hot-ish logits: the target class dominates, so the loss is small but
    // strictly positive (a perfect prediction would be zero, which cannot be
    // represented once other classes carry any probability).
    constexpr int batch = 2, classes = 3;
    std::vector<float> logits(batch * classes, 0.0f);
    for (int b = 0; b < batch; ++b) {
        logits[b * classes + (b % classes)] = 2.0f;
    }
    const std::vector<int> targets = {0, 1};
    std::vector<float> output(batch, 0.0f);

    const float loss = cross_entropy_loss(logits.data(), targets.data(),
                                          output.data(), batch, classes, {},
                                          nullptr);

    EXPECT_GT(loss, 0.0f);
    EXPECT_LT(loss, 10.0f) << "a confident prediction should give a small loss";
    EXPECT_TRUE(std::isfinite(loss));
}

TEST(CrossEntropyValidationTest, MeanReductionIsTheAverageOfPerSampleLosses) {
    constexpr int batch = 2, classes = 2;
    // Both samples predict evenly, so each loss is log(2) and the mean is too.
    const std::vector<float> logits(batch * classes, 0.0f);
    const std::vector<int> targets = {0, 1};
    std::vector<float> output(batch, 0.0f);

    CrossEntropyConfig mean_config;
    mean_config.reduction_mean = true;
    const float mean = cross_entropy_loss(logits.data(), targets.data(),
                                          output.data(), batch, classes,
                                          mean_config, nullptr);

    CrossEntropyConfig sum_config;
    sum_config.reduction_mean = false;
    const float sum = cross_entropy_loss(logits.data(), targets.data(),
                                         output.data(), batch, classes,
                                         sum_config, nullptr);

    EXPECT_NEAR(mean, std::log(2.0f), 1e-4f) << "uniform logits over 2 classes";
    EXPECT_NEAR(sum, mean * batch, 1e-4f)
        << "the sum reduction is batch times the mean";
}

TEST(CrossEntropyValidationTest, HighestBoundaryTargetIsAccepted) {
    // num_classes - 1 is the largest legal label; off-by-one here would make the
    // last class unusable.
    constexpr int batch = 1, classes = 4;
    const std::vector<float> logits(classes, 0.0f);
    const std::vector<int> targets = {classes - 1};
    std::vector<float> output(batch, 0.0f);

    float loss = 0.0f;
    EXPECT_NO_THROW(loss = cross_entropy_loss(logits.data(), targets.data(),
                                              output.data(), batch, classes, {},
                                              nullptr));
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(loss, std::log(static_cast<float>(classes)), 1e-4f);
}

}  // namespace cuda::neural::loss::test
