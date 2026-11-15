/**
 * @file sampling_validation_test.cpp
 * @brief Contracts for sampler input validation and its probability maths
 *
 * src/cuda/inference/beam_search.cpp implements the top-k and top-p samplers.
 * Nothing validated the inputs every sampler reads unconditionally:
 *
 * - logits[0] is dereferenced before any size test, so vocab_size == 0 read out of
 *   bounds and aborted the process ("dumped core"). A negative size surfaced as
 *   "cannot create std::vector larger than max_size()", an error about a container
 *   rather than about the caller's argument.
 *
 * - temperature divides the logits; temperature == 0 produced NaN probabilities
 *   (measured: "returned 2 entries, first prob=-nan").
 *
 * The same pattern appeared in three places — TopKSampler::sample,
 * TopPSampler::sample and BeamSearch::step — so the check is a shared helper and
 * all three now report the offending argument by name.
 */

#include <cuda/inference/beam_search.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::inference::test {

namespace {

using cuda::inference::TopKSampler;
using cuda::inference::TopPSampler;

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
TEST(TopKSamplerValidationTest, EmptyVocabularyIsRejected) {
    const std::string message = message_from([] {
        TopKSampler sampler(1);
        (void)sampler.sample(nullptr, 0, 1.0f, 42);
    });

    EXPECT_FALSE(message.empty())
        << "logits[0] would be read from an empty vocabulary";
    EXPECT_NE(message.find("vocab_size"), std::string::npos)
        << "the message should name the argument: " << message;
    EXPECT_NE(message.find("0"), std::string::npos) << message;
}

// Regression: this surfaced as a std::vector container error.
TEST(TopKSamplerValidationTest, NegativeVocabularyIsRejected) {
    const std::string message = message_from([] {
        TopKSampler sampler(1);
        (void)sampler.sample(nullptr, -1, 1.0f, 42);
    });

    EXPECT_FALSE(message.empty());
    EXPECT_NE(message.find("vocab_size"), std::string::npos)
        << "the error should be about the argument, not a container: " << message;
}

// Regression: this produced NaN probabilities.
TEST(TopKSamplerValidationTest, ZeroTemperatureIsRejected) {
    const std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    const std::string message = message_from([&] {
        TopKSampler sampler(2);
        (void)sampler.sample(logits.data(), 3, 0.0f, 42);
    });

    EXPECT_FALSE(message.empty()) << "a zero divisor cannot yield probabilities";
    EXPECT_NE(message.find("temperature"), std::string::npos) << message;
}

TEST(TopKSamplerValidationTest, NegativeTemperatureIsRejected) {
    const std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    const std::string message = message_from([&] {
        TopKSampler sampler(2);
        (void)sampler.sample(logits.data(), 3, -1.0f, 42);
    });
    EXPECT_FALSE(message.empty())
        << "a negative temperature inverts the distribution rather than scaling it";
}

TEST(TopKSamplerValidationTest, TopKIsTheHighestScoringTokensInOrder) {
    // Distinct logits: the ranking is unambiguous.
    const std::vector<float> logits = {0.0f, 5.0f, 1.0f, 4.0f, 2.0f};

    TopKSampler sampler(3);
    const auto result = sampler.sample(logits.data(), 5, 1.0f, 42);

    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0].first, 1) << "logit 5 is the largest";
    EXPECT_EQ(result[1].first, 3) << "logit 4 is second";
    EXPECT_EQ(result[2].first, 4) << "logit 2 is third";

    for (const auto& entry : result) {
        EXPECT_GT(entry.second, 0.0f);
        EXPECT_LT(entry.second, 1.0f);
    }
    EXPECT_GE(result[0].second, result[1].second)
        << "the returned entries must be ordered by probability";
    EXPECT_GE(result[1].second, result[2].second);
}

TEST(TopKSamplerValidationTest, KAboveTheVocabularyIsClamped) {
    const std::vector<float> logits = {0.0f, 1.0f};

    TopKSampler sampler(1000);
    const auto result = sampler.sample(logits.data(), 2, 1.0f, 42);

    EXPECT_EQ(result.size(), 2u) << "cannot return more tokens than exist";
}

TEST(TopKSamplerValidationTest, ProbabilitiesSumToOneAcrossTheWholeVocabulary) {
    // Asking for every token must return the complete normalised distribution.
    const std::vector<float> logits = {0.5f, -0.5f, 1.5f, 0.0f};

    TopKSampler sampler(4);
    const auto result = sampler.sample(logits.data(), 4, 1.0f, 7);

    ASSERT_EQ(result.size(), 4u);
    float total = 0.0f;
    for (const auto& entry : result) {
        EXPECT_TRUE(std::isfinite(entry.second)) << entry.second;
        total += entry.second;
    }
    EXPECT_NEAR(total, 1.0f, 1e-4f);
}

TEST(TopKSamplerValidationTest, SingleTokenVocabularyReturnsThatToken) {
    const std::vector<float> logits = {3.0f};

    TopKSampler sampler(1);
    const auto result = sampler.sample(logits.data(), 1, 1.0f, 42);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].first, 0);
    EXPECT_NEAR(result[0].second, 1.0f, 1e-6f)
        << "the only token carries all the probability";
}

TEST(TopPSamplerValidationTest, EmptyVocabularyIsRejected) {
    const std::string message = message_from([] {
        TopPSampler sampler(0.9f);
        (void)sampler.sample(nullptr, 0, 1.0f, 42);
    });

    EXPECT_FALSE(message.empty())
        << "the top-p sampler has the same unconditional logits[0] read";
    EXPECT_NE(message.find("TopPSampler"), std::string::npos)
        << "the message should name which sampler failed: " << message;
}

TEST(TopPSamplerValidationTest, ZeroTemperatureIsRejected) {
    const std::vector<float> logits = {1.0f, 2.0f};
    const std::string message = message_from([&] {
        TopPSampler sampler(0.9f);
        (void)sampler.sample(logits.data(), 2, 0.0f, 42);
    });
    EXPECT_FALSE(message.empty());
}

TEST(TopPSamplerValidationTest, SamplingReturnsAValidTokenIndex) {
    const std::vector<float> logits = {0.0f, 10.0f, 0.0f};

    TopPSampler sampler(0.9f);
    const int token = sampler.sample(logits.data(), 3, 1.0f, 12345);

    EXPECT_GE(token, 0);
    EXPECT_LT(token, 3) << "the returned index must be inside the vocabulary";
    EXPECT_EQ(token, 1)
        << "with p=0.9 the overwhelming mass is on the largest logit, so the "
           "sampled token is deterministic in practice";
}

}  // namespace cuda::inference::test
