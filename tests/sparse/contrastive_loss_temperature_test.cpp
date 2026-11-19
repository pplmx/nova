/**
 * @file contrastive_loss_temperature_test.cpp
 * @brief contrastive_loss divides by config.temperature and must reject a non-positive one
 *
 * The similarity is computed as
 *
 *     similarities[b] = cos_sim / config.temperature;
 *     neg_sim += expf(cos_sim / config.temperature);
 *
 * so temperature == 0 gives inf and then exp(inf) == inf, and a negative temperature flips the
 * sign of every term, inverting which pairs the loss treats as negatives. The field defaults to
 * 0.07 and nothing rejected an explicit zero — the function had no argument checks at all,
 * while cross_entropy_loss in the same file rejects its own degenerate geometry.
 *
 * It throws std::invalid_argument now.
 */

#include <cuda/neural/loss/loss_functions.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace contrastive_loss_temperature_test {

    TEST(ContrastiveLossTemperatureTest, AZeroTemperatureIsRejected) {
        constexpr int kBatch = 2;
        constexpr int kDim = 4;
        std::vector<float> first(kBatch * kDim, 1.0F);
        std::vector<float> second(kBatch * kDim, 1.0F);
        std::vector<float> output(kBatch, 0.0F);

        cuda::neural::loss::ContrastiveLossConfig config;
        config.temperature = 0.0F;

        // Before the guard this returned inf for every pair rather than saying the argument was
        // wrong. Wrap in a lambda: the macro takes the comma in the call as a separator.
        auto call = [&] { return cuda::neural::loss::contrastive_loss(first.data(), second.data(), output.data(), kBatch, kDim, config); };
        EXPECT_THROW(call(), std::invalid_argument);
    }

    TEST(ContrastiveLossTemperatureTest, ANegativeTemperatureIsRejected) {
        constexpr int kBatch = 2;
        constexpr int kDim = 4;
        std::vector<float> first(kBatch * kDim, 1.0F);
        std::vector<float> second(kBatch * kDim, 1.0F);
        std::vector<float> output(kBatch, 0.0F);

        cuda::neural::loss::ContrastiveLossConfig config;
        config.temperature = -0.07F;  // the default with the sign flipped

        auto call = [&] { return cuda::neural::loss::contrastive_loss(first.data(), second.data(), output.data(), kBatch, kDim, config); };
        EXPECT_THROW(call(), std::invalid_argument);
    }

    TEST(ContrastiveLossTemperatureTest, TheDefaultTemperatureIsAccepted) {
        constexpr int kBatch = 2;
        constexpr int kDim = 4;
        std::vector<float> first(kBatch * kDim, 1.0F);
        std::vector<float> second(kBatch * kDim, 0.5F);
        std::vector<float> output(kBatch, -1.0F);

        cuda::neural::loss::ContrastiveLossConfig config;  // temperature defaults to 0.07

        auto call = [&] { return cuda::neural::loss::contrastive_loss(first.data(), second.data(), output.data(), kBatch, kDim, config); };
        EXPECT_NO_THROW(call());

        for (int b = 0; b < kBatch; ++b) {
            EXPECT_TRUE(std::isfinite(output[b])) << "pair " << b << " should be a finite loss";
        }
    }

}  // namespace contrastive_loss_temperature_test
