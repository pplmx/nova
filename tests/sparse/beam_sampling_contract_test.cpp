/**
 * @file beam_sampling_contract_test.cpp
 * @brief Contracts for the beam-search sampling path and configuration bounds
 *
 * src/cuda/inference/beam_search.cpp's sample_token() read
 *
 *     if (top_p < 1.0f)          -> TopPSampler(top_p)
 *     else if (top_k > 1)        -> top-k, then r = (seed % 1000) / 1000
 *     else                       -> TopPSampler(0.9f)     <- hardcoded
 *
 * and four defects came out of it, all measured before the fix:
 *
 * - The default config is top_k = 1, top_p = 1.0, which every caller reads as greedy.
 *   The final branch ignored both and sampled with top_p = 0.9: of 100 emitted tokens
 *   with an unmistakable argmax, 56 were not the argmax.
 *
 * - search() builds the seed as step * 1000 + beam, and the top-k branch drew
 *   r = (seed % 1000) / 1000 -- exactly beam / 1000. Every beam in a step therefore drew
 *   a number below the first atom's probability whenever that atom exceeded beam/1000.
 *   With four equally likely tokens and four beams, all 100 draws landed on one token and
 *   all four beams emitted an identical sequence.
 *
 * - The branch never consulted top_p once top_k > 1, so a (top_k, top_p) pair behaved as
 *   top_k alone.
 *
 * - enable_length_norm = false returned from compute_length_normalized_scores() without
 *   writing `score`, which is what prune_beams() and the final sort rank on. Every
 *   hypothesis stayed at its initial 0.0f. Measured on four beams with true log_probs
 *   -14.62, -13.62, -14.42, -15.22: the returned order was 14.62, 13.62, 14.42, 15.22,
 *   i.e. not monotone.
 *
 * Also covered: the constructor accepted a BeamSearchConfig that could not work (max_beams
 * = -4 surfaced as "vector::reserve", max_beams = 0 returned an empty result as a success),
 * the samplers accepted out-of-range parameters, and search() dereferenced a null
 * BlockManager on the default enable_reuse = true.
 */

#include <cuda/inference/beam_search.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::inference::test {

    namespace {

        using cuda::inference::BeamHypothesis;
        using cuda::inference::BeamSearchConfig;
        using cuda::inference::BeamSearchManager;
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

        /// Drives search() with constant logits so a case can look at what was emitted.
        std::vector<BeamHypothesis> search_with_logits(const BeamSearchConfig& config, const std::vector<float>& logits, int prompt_length = 1) {
            BeamSearchManager manager(nullptr, config);
            cuda::memory::Buffer<float> embeddings(16);
            cuda::stream::Stream stream;

            const int vocab = static_cast<int>(logits.size());
            auto forward_fn = [&logits](cuda::memory::Buffer<float>& out, const std::vector<int64_t>&, const cuda::stream::Stream&) {
                out.copy_from(logits.data(), logits.size());
            };
            return manager.search(embeddings, prompt_length, vocab, stream, forward_fn);
        }

        /// Constant-logits config: no EOS, no reuse, a fixed number of decode steps.
        BeamSearchConfig base_config(int steps, int beams = 4) {
            BeamSearchConfig config;
            config.max_beams = beams;
            config.max_length = steps + 1;  // prompt length 1
            config.eos_token_id = 99;       // never sampled, so no beam stops early
            config.enable_reuse = false;
            return config;
        }

    }  // namespace

    // Regression: the default top_k = 1 / top_p = 1.0 config sampled with a hardcoded
    // top_p = 0.9, so a "greedy" search was not greedy. Measured before the fix: 56 of 100
    // emitted tokens were not the argmax.
    TEST(BeamSamplingContractTest, TopKOfOneIsGreedy) {
        const BeamSearchConfig config = base_config(25);
        ASSERT_EQ(config.top_k, 1) << "this case is about the default top_k";
        ASSERT_EQ(config.top_p, 1.0f) << "and the default top_p";

        // Token 0 is the argmax with p ~ 0.65; the rest share the remainder.
        const std::vector<BeamHypothesis> result = search_with_logits(config, {2.0f, 1.0f, 1.0f, 1.0f, 1.0f});

        int total = 0;
        int non_argmax = 0;
        for (const auto& hypothesis : result) {
            ASSERT_FALSE(hypothesis.tokens.empty());
            for (int token : hypothesis.tokens) {
                ++total;
                if (token != 0) {
                    ++non_argmax;
                }
            }
        }
        ASSERT_GT(total, 0);
        EXPECT_EQ(non_argmax, 0) << non_argmax << " of " << total << " emitted tokens were not the argmax";
    }

    // Regression: the top-k branch drew (seed % 1000) / 1000, which for search()'s own
    // seed construction is beam/1000. All four beams saw the same draw.
    TEST(BeamSamplingContractTest, TopKSpreadsItsDrawAcrossBeams) {
        BeamSearchConfig config = base_config(25);
        config.top_k = 4;
        config.top_p = 1.0f;

        // Four equally likely candidates, then a token with no chance. A correct uniform
        // draw over the top-4 set reaches all four; the old code reached one.
        const std::vector<BeamHypothesis> result = search_with_logits(config, {1.0f, 1.0f, 1.0f, 1.0f, -20.0f});

        int per_token[4] = {0, 0, 0, 0};
        for (const auto& hypothesis : result) {
            for (int token : hypothesis.tokens) {
                if (token >= 0 && token < 4) {
                    ++per_token[token];
                }
            }
        }

        int distinct = 0;
        for (int count : per_token) {
            if (count > 0) {
                ++distinct;
            }
        }
        EXPECT_GE(distinct, 2) << "all draws landed on one token: " << per_token[0] << "/" << per_token[1] << "/" << per_token[2] << "/" << per_token[3];

        // And the beams must not all be carbon copies of each other.
        bool all_identical = true;
        for (size_t i = 1; i < result.size(); ++i) {
            if (result[i].tokens != result[0].tokens) {
                all_identical = false;
                break;
            }
        }
        EXPECT_FALSE(all_identical) << "every beam emitted the same sequence";
    }

    // The two knobs must compose: top_p narrows the top-k set rather than being ignored.
    //
    // NOTE ON DETECTION: this case passes against the unfixed implementation as well, so it is
    // a guard rather than a regression test. The old top-k branch never read top_p, but its
    // draw was (seed % 1000) / 1000 = beam / 1000, i.e. <= 0.003, which is below the leading
    // candidate's probability in any peaked distribution -- so it returned that candidate and
    // the result looked like correct top_p truncation by accident. What this case pins is the
    // behaviour after the draw was fixed: with a real draw, a top_p below the leading token's
    // mass must still exclude the tail.
    TEST(BeamSamplingContractTest, TopPNarrowsTheTopKSet) {
        // Tokens 0 and 1 carry almost all the mass; 2 and 3 are far behind. With top_k = 4
        // and top_p = 0.5 only the leading token survives, so nothing else may be emitted.
        BeamSearchConfig config = base_config(25);
        config.top_k = 4;
        config.top_p = 0.5f;

        const std::vector<BeamHypothesis> result = search_with_logits(config, {4.0f, 1.0f, -6.0f, -6.0f});

        for (const auto& hypothesis : result) {
            for (int token : hypothesis.tokens) {
                EXPECT_LE(token, 1) << "top_p = 0.5 excludes the tail, but token " << token << " was emitted";
            }
        }
    }

    // Regression: with length normalisation off, `score` was never written, so the final
    // sort ranked four different hypotheses on the constant 0.0f.
    TEST(BeamSamplingContractTest, RankingIsByLogProbabilityWhenNormalisationIsOff) {
        BeamSearchConfig config = base_config(11);
        config.enable_length_norm = false;
        config.top_k = 4;
        config.top_p = 0.9f;

        const std::vector<BeamHypothesis> result = search_with_logits(config, {1.2f, 1.0f, 0.8f, 0.6f, -20.0f});

        ASSERT_GE(result.size(), 2u);
        for (const auto& hypothesis : result) {
            EXPECT_FLOAT_EQ(hypothesis.score, hypothesis.log_prob) << "score must be the ranking key, not a leftover 0.0f";
        }
        for (size_t i = 1; i < result.size(); ++i) {
            EXPECT_GE(result[i - 1].log_prob, result[i].log_prob) << "beam " << i << " is ranked above a better one";
        }
    }

    // The enabled path must keep the documented normalisation.
    TEST(BeamSamplingContractTest, RankingIsByLengthNormalisedScoreWhenEnabled) {
        BeamSearchConfig config = base_config(11);
        config.enable_length_norm = true;
        config.length_penalty = 0.7f;
        config.top_k = 4;
        config.top_p = 0.9f;

        const std::vector<BeamHypothesis> result = search_with_logits(config, {1.2f, 1.0f, 0.8f, 0.6f, -20.0f});

        ASSERT_GE(result.size(), 2u);
        for (size_t i = 0; i < result.size(); ++i) {
            const float expected = result[i].length > 0 ? result[i].log_prob / std::pow(static_cast<float>(result[i].length), config.length_penalty) : result[i].log_prob;
            EXPECT_FLOAT_EQ(result[i].score, expected) << "beam " << i;
            if (i > 0) {
                EXPECT_GE(result[i - 1].score, result[i].score);
            }
        }
    }

    // Regression: a null BlockManager with the default enable_reuse = true dereferenced
    // nullptr inside search(). ASan reported a SEGV on address 0x160 in
    // BlockManager::get_kv_cache() called from beam_search.cpp.
    TEST(BeamSamplingContractTest, NullBlockManagerIsUsableWithReuseEnabled) {
        BeamSearchConfig config = base_config(4, 2);
        config.enable_reuse = true;
        ASSERT_EQ(config.enable_reuse, BeamSearchConfig {}.enable_reuse) << "the default is what crashed";

        // Constructing and running on nullptr used to segfault; both must be no-ops.
        BeamSearchManager manager(nullptr, config);
        cuda::memory::Buffer<float> embeddings(16);
        cuda::stream::Stream stream;
        const std::vector<float> logits = {1.0f, 1.0f, 1.0f, 1.0f};
        auto forward_fn = [&logits](cuda::memory::Buffer<float>& out, const std::vector<int64_t>&, const cuda::stream::Stream&) { out.copy_from(logits.data(), logits.size()); };

        EXPECT_NO_THROW({
            const auto result = manager.search(embeddings, 1, 4, stream, forward_fn);
            EXPECT_EQ(result.size(), 2u);
        });
    }

    // The config is the first thing every decode step divides by or indexes with, so it is
    // checked where it is set rather than after a forward pass has run.
    TEST(BeamSamplingContractTest, ConstructorRejectsAConfigThatCannotWork) {
        const auto rejection = [](BeamSearchConfig config) { return message_from([&config] { BeamSearchManager manager(nullptr, config); }); };

        EXPECT_FALSE(rejection([] {
                         BeamSearchConfig c;
                         c.max_beams = 0;
                         return c;
                     }())
                         .empty())
            << "max_beams = 0 ran a search that returned nothing, as a success";
        EXPECT_FALSE(rejection([] {
                         BeamSearchConfig c;
                         c.max_beams = -4;
                         return c;
                     }())
                         .empty())
            << "max_beams = -4 surfaced as vector::reserve";
        EXPECT_FALSE(rejection([] {
                         BeamSearchConfig c;
                         c.top_k = 0;
                         return c;
                     }())
                         .empty());
        EXPECT_FALSE(rejection([] {
                         BeamSearchConfig c;
                         c.top_p = 0.0f;
                         return c;
                     }())
                         .empty());
        EXPECT_FALSE(rejection([] {
                         BeamSearchConfig c;
                         c.top_p = 1.5f;
                         return c;
                     }())
                         .empty());
        EXPECT_FALSE(rejection([] {
                         BeamSearchConfig c;
                         c.temperature = 0.0f;
                         return c;
                     }())
                         .empty());
        EXPECT_FALSE(rejection([] {
                         BeamSearchConfig c;
                         c.length_penalty = -0.7f;
                         return c;
                     }())
                         .empty());
        EXPECT_FALSE(rejection([] {
                         BeamSearchConfig c;
                         c.length_penalty = std::numeric_limits<float>::quiet_NaN();
                         return c;
                     }())
                         .empty());

        // The defaults must stay valid, or every caller breaks.
        EXPECT_TRUE(rejection(BeamSearchConfig {}).empty());
    }

    TEST(BeamSamplingContractTest, ConfigureRejectsTheSameConfigsAsTheConstructor) {
        BeamSearchConfig valid;
        BeamSearchManager manager(nullptr, valid);

        BeamSearchConfig bad;
        bad.max_beams = 0;
        const std::string message = message_from([&] { manager.configure(bad); });

        EXPECT_FALSE(message.empty()) << "configure() is the documented way to change the bounds";
        EXPECT_EQ(manager.get_config().max_beams, valid.max_beams) << "a rejected config must not be stored";
    }

    // Regression: TopKSampler(0) returned an empty list from a successful sample(), and
    // TopKSampler(-1) surfaced as "cannot create std::vector larger than max_size()".
    TEST(BeamSamplingContractTest, TopKRequiresAPositiveK) {
        const auto rejection = [](int k) {
            return message_from([k] {
                TopKSampler sampler(k);
                const std::vector<float> logits = {1.0f, 2.0f};
                (void)sampler.sample(logits.data(), 2, 1.0f, 7);
            });
        };

        const std::string zero = rejection(0);
        EXPECT_FALSE(zero.empty()) << "k = 0 is not a valid truncation";
        EXPECT_NE(zero.find("TopKSampler"), std::string::npos) << zero;

        const std::string negative = rejection(-1);
        EXPECT_FALSE(negative.empty());
        EXPECT_EQ(negative.find("vector"), std::string::npos) << "the error should name the argument, not a container: " << negative;

        EXPECT_TRUE(rejection(1).empty());
    }

    // Regression: every out-of-range p sampled from the whole vocabulary (or never reached
    // its cutoff) and returned a token as if it had worked.
    TEST(BeamSamplingContractTest, TopPRequiresAProbability) {
        const auto rejection = [](float p) {
            return message_from([p] {
                TopPSampler sampler(p);
                const std::vector<float> logits = {3.0f, 2.0f, 1.0f};
                (void)sampler.sample(logits.data(), 3, 1.0f, 7);
            });
        };

        for (float bad : {0.0f, -0.5f, 1.5f, std::numeric_limits<float>::quiet_NaN()}) {
            const std::string message = rejection(bad);
            EXPECT_FALSE(message.empty()) << "top_p = " << bad << " was accepted";
            EXPECT_NE(message.find("top_p"), std::string::npos) << message;
        }

        EXPECT_TRUE(rejection(1.0f).empty()) << "the upper bound is inclusive";
        EXPECT_TRUE(rejection(0.5f).empty());
    }

    // Regression: vocab_size reached the logits Buffer, so 0 surfaced as a CUDA "invalid
    // argument" and a negative size as "out of memory". A prompt_length that leaves no room
    // to generate returned max_beams empty hypotheses as if the search had succeeded.
    TEST(BeamSamplingContractTest, SearchRejectsArgumentsItCannotHonour) {
        BeamSearchConfig config = base_config(4, 2);
        BeamSearchManager manager(nullptr, config);
        cuda::memory::Buffer<float> embeddings(16);
        cuda::stream::Stream stream;
        const std::vector<float> logits = {1.0f, 1.0f};
        auto forward_fn = [&logits](cuda::memory::Buffer<float>& out, const std::vector<int64_t>&, const cuda::stream::Stream&) { out.copy_from(logits.data(), logits.size()); };

        const std::string zero_vocab = message_from([&] { (void)manager.search(embeddings, 1, 0, stream, forward_fn); });
        EXPECT_NE(zero_vocab.find("vocab_size"), std::string::npos) << "the error should be about the argument, not the allocator: " << zero_vocab;

        const std::string negative_vocab = message_from([&] { (void)manager.search(embeddings, 1, -8, stream, forward_fn); });
        EXPECT_NE(negative_vocab.find("vocab_size"), std::string::npos) << negative_vocab;

        const std::string no_room = message_from([&] { (void)manager.search(embeddings, config.max_length, 2, stream, forward_fn); });
        EXPECT_FALSE(no_room.empty()) << "prompt_length == max_length leaves nothing to decode";
        EXPECT_NE(no_room.find("prompt_length"), std::string::npos) << no_room;
    }

}  // namespace cuda::inference::test
