#pragma once

#include "cuda/inference/block_manager.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "cuda/stream/stream.h"
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace cuda::inference {

    /**
     * @struct BeamSearchConfig
     * @brief Beam search decoding parameters
     *
     * @par The sampling contract
     * `top_k == 1` is greedy: the highest-logit token, with no draw. Otherwise top-k keeps
     * the k most likely tokens, `top_p` keeps the shortest prefix of those whose cumulative
     * probability reaches p, and the emitted token is drawn from what remains. Both knobs are
     * honoured together; neither replaces the other.
     *
     * @par These values are validated where they are set
     * The constructor and configure() reject a config that cannot work, rather than letting
     * it survive until the first decode step: max_beams and top_k must be at least 1, top_p
     * must be in (0, 1], temperature must be positive, and length_penalty must be finite and
     * non-negative. Every one of those previously produced a wrong answer instead of an
     * error -- a non-positive max_beams returned an empty result or surfaced as
     * "vector::reserve", and a NaN length_penalty made every score NaN.
     *
     * @par enable_reuse needs a BlockManager
     * It defaults to true, and a BeamSearchManager may be constructed with a null
     * BlockManager for callers that only want beam bookkeeping. Reuse is then skipped
     * rather than dereferenced.
     */
    struct BeamSearchConfig {
        int max_beams = 4;
        int min_beams = 1;
        float length_penalty = 0.7f;
        float temperature = 1.0f;
        int top_k = 1;
        float top_p = 1.0f;
        int max_length = 2048;
        bool enable_length_norm = true;
        bool enable_reuse = true;
        int eos_token_id = 0;
        int rebase_threshold = 1000;
    };

    struct BeamHypothesis {
        int64_t sequence_id = -1;
        std::vector<int> tokens;
        float score = 0.0f;
        float log_prob = 0.0f;
        int length = 0;
        bool finished = false;
        int parent_beam = -1;
        int64_t kv_source_sequence = -1;
    };

    struct BeamSearchTraceEntry {
        int step;
        int beam_id;
        int token;
        float score;
        float normalized_score;
        float log_prob;
        int64_t sequence_id;
    };

    /**
     * @class TopKSampler
     * @brief Truncates a distribution to its k most likely tokens
     */
    class TopKSampler {
    public:
        /**
         * @brief Construct a top-k truncation
         * @param k How many tokens survive; must be at least 1
         * @throws std::invalid_argument if k is less than 1
         */
        explicit TopKSampler(int k);

        /**
         * @brief Return the k most likely tokens with their probabilities
         * @param logits Raw logits, at least vocab_size entries
         * @param vocab_size Number of logits; must be positive
         * @param temperature Softmax temperature; must be positive
         * @param seed Seed for the tie-breaking draw between equal probabilities
         * @throws std::invalid_argument if vocab_size or temperature is out of range
         */
        std::vector<std::pair<int, float>> sample(const float* logits, int vocab_size, float temperature, uint64_t seed);

    private:
        int k_;
    };

    /**
     * @class TopPSampler
     * @brief Truncates a distribution to its nucleus of cumulative probability p
     */
    class TopPSampler {
    public:
        /**
         * @brief Construct a top-p truncation
         * @param p Cumulative probability to keep; must be in (0, 1]
         * @throws std::invalid_argument if p is outside (0, 1]
         */
        explicit TopPSampler(float p);

        /**
         * @brief Sample one token from the nucleus
         * @param logits Raw logits, at least vocab_size entries
         * @param vocab_size Number of logits; must be positive
         * @param temperature Softmax temperature; must be positive
         * @param seed Draw seed
         * @throws std::invalid_argument if vocab_size or temperature is out of range
         */
        int sample(const float* logits, int vocab_size, float temperature, uint64_t seed);

    private:
        float p_;
    };

    /**
     * @class BeamSearchManager
     * @brief Runs beam-search decoding over a caller-supplied forward pass
     */
    class BeamSearchManager {
    public:
        /**
         * @brief Construct a beam-search manager
         * @param block_manager KV block manager used to fork prefix blocks; may be null, in
         *        which case enable_reuse is skipped rather than dereferenced
         * @param config Decoding parameters; validated here
         * @throws std::invalid_argument if the config is out of range
         */
        explicit BeamSearchManager(BlockManager* block_manager, const BeamSearchConfig& config);

        BeamSearchManager(const BeamSearchManager&) = delete;
        BeamSearchManager& operator=(const BeamSearchManager&) = delete;
        BeamSearchManager(BeamSearchManager&&) = default;
        BeamSearchManager& operator=(BeamSearchManager&&) = default;

        /**
         * @brief Decode up to max_length - prompt_length tokens for each beam
         * @param prompt_embeddings Prompt activations passed to forward_fn unchanged
         * @param prompt_length Tokens already consumed; must be in [0, max_length)
         * @param vocab_size Logits per position; must be positive
         * @param stream Stream for the forward pass
         * @param forward_fn Fills a logits buffer for the given sequence ids
         * @return Surviving hypotheses, best first
         * @throws std::invalid_argument if prompt_length or vocab_size is out of range
         */
        std::vector<BeamHypothesis> search(const memory::Buffer<float>& prompt_embeddings,
                                           int prompt_length,
                                           int vocab_size,
                                           const stream::Stream& stream,
                                           std::function<void(memory::Buffer<float>&, const std::vector<int64_t>&, const stream::Stream&)> forward_fn);

        /**
         * @brief Replace the decoding parameters
         * @param config Validated exactly as the constructor validates it
         * @throws std::invalid_argument if the config is out of range, leaving the old one
         *         in place
         */
        void configure(const BeamSearchConfig& config);
        BeamSearchConfig get_config() const { return config_; }

        std::string export_trace_json() const;
        std::string export_trace_csv() const;
        void clear_trace();

        struct TraceStats {
            int total_steps;
            int avg_beam_width;
            float avg_score;
            float avg_length_norm;
        };
        TraceStats get_trace_stats() const;

    private:
        void initialize_beams(int prompt_length);
        /// Declared but neither defined nor called. A scan for methods that have a declaration
        /// in a header and no definition in src/ flags this one; it does not break the build
        /// because nothing calls it, and no test can reach it. Left in place rather than removed
        /// because the surrounding beam-search code is what the tests exercise and this is not
        /// part of that; noted so the next reader does not assume it is usable.
        void expand_beams(const memory::Buffer<float>& logits, const stream::Stream& stream);
        void prune_beams();
        void compute_length_normalized_scores();
        void rebase_scores();
        int sample_token(const float* logits, int vocab_size, uint64_t seed);

        BlockManager* block_manager_;
        BeamSearchConfig config_;
        std::vector<BeamHypothesis> hypotheses_;
        std::vector<BeamHypothesis> next_hypotheses_;

        std::vector<BeamSearchTraceEntry> trace_;
        int next_sequence_id_ = 1;

        /// Seeded with a constant and never read. The draws are made from a local generator
        /// seeded by sample_token()'s own argument, so this member carries no state into
        /// them; it is kept rather than removed because it is part of the class layout and
        /// nothing about the sampling contract depends on its absence. Do not read it as the
        /// stream the search draws from -- it is not.
        std::mt19937 rng_;
    };

}  // namespace cuda::inference
