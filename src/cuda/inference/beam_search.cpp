#include "cuda/inference/beam_search.h"
#include "cuda/device/error.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cuda::inference {

    namespace {

        /// Checks the inputs every sampler reads unconditionally: logits[0] is
        /// dereferenced before any size test, and temperature divides the logits.
        /// vocab_size == 0 aborted the process, a negative size surfaced as an unrelated
        /// std::vector error, and temperature == 0 made every probability NaN.
        void validate_sampling_inputs(int vocab_size, float temperature, const char* fn) {
            if (vocab_size <= 0) {
                throw std::invalid_argument(std::string(fn) + ": vocab_size must be positive, got " + std::to_string(vocab_size));
            }
            if (!(temperature > 0.0f)) {
                throw std::invalid_argument(std::string(fn) + ": temperature must be positive, got " + std::to_string(temperature));
            }
        }

        /// The top-p sampler's only parameter. It is a cumulative probability, so the
        /// meaningful range is (0, 1]. Nothing checked it: p = 0 and p = -0.5 fell through
        /// the cutoff loop without ever accumulating past p and sampled from the whole
        /// vocabulary, p = 1.5 never reached its cutoff, and p = NaN made every comparison
        /// false -- all four returned token 0 from peaked logits as if they had sampled.
        void validate_top_p(float p, const char* fn) {
            if (!(p > 0.0f && p <= 1.0f)) {
                throw std::invalid_argument(std::string(fn) + ": top_p must be in (0, 1], got " + std::to_string(p));
            }
        }

        /// Everything the sampling maths divides by or indexes with. The values were
        /// previously unvalidated until the first decode step: max_beams = -4 surfaced as
        /// "vector::reserve" from initialize_beams, max_beams = 0 returned an empty result
        /// as if search had run, and a bad temperature or top_p only threw once a forward
        /// pass had already produced logits.
        void validate_beam_config(const BeamSearchConfig& config, const char* fn) {
            if (config.max_beams < 1) {
                throw std::invalid_argument(std::string(fn) + ": max_beams must be at least 1, got " + std::to_string(config.max_beams));
            }
            if (config.top_k < 1) {
                throw std::invalid_argument(std::string(fn) + ": top_k must be at least 1, got " + std::to_string(config.top_k));
            }
            validate_top_p(config.top_p, fn);
            if (!(config.temperature > 0.0f)) {
                throw std::invalid_argument(std::string(fn) + ": temperature must be positive, got " + std::to_string(config.temperature));
            }
            // length_penalty is the exponent in length^penalty. A negative exponent turns
            // the normaliser into a multiplier of the length, which rewards long sequences
            // instead of penalising them; a NaN one makes every score NaN and every
            // comparison in the sorts false.
            if (!std::isfinite(config.length_penalty) || config.length_penalty < 0.0f) {
                throw std::invalid_argument(std::string(fn) + ": length_penalty must be finite and non-negative, got " + std::to_string(config.length_penalty));
            }
        }

    }  // namespace

    TopKSampler::TopKSampler(int k)
        : k_(k) {
        // `k` is how many tokens survive, so it cannot be zero or negative. Nothing checked
        // it: k = 0 returned an empty list from a successful call, and k = -1 surfaced as
        // "cannot create std::vector larger than max_size()", an error about a container
        // rather than about the argument.
        if (k < 1) {
            throw std::invalid_argument("TopKSampler: k must be at least 1, got " + std::to_string(k));
        }
    }

    std::vector<std::pair<int, float>> TopKSampler::sample(const float* logits, int vocab_size, float temperature, uint64_t seed) {
        validate_sampling_inputs(vocab_size, temperature, "TopKSampler::sample");

        const int effective_k = std::min(k_, vocab_size);
        std::vector<std::pair<int, float>> top_k(effective_k);

        std::vector<float> probs(static_cast<size_t>(vocab_size));
        float max_logit = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            max_logit = std::max(max_logit, logits[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp((logits[i] - max_logit) / temperature);
            sum += probs[i];
        }

        for (int i = 0; i < vocab_size; ++i) {
            probs[i] /= sum;
        }

        std::vector<int> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + effective_k, indices.end(), [&probs](int a, int b) { return probs[a] > probs[b]; });

        std::mt19937 rng(static_cast<uint64_t>(seed));
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < effective_k; ++i) {
            if (i > 0 && std::abs(probs[indices[i]] - probs[indices[i - 1]]) < 1e-6f) {
                if (dist(rng) < 0.5f) {
                    std::swap(indices[i], indices[i - 1]);
                }
            }
            top_k[i] = {indices[i], probs[indices[i]]};
        }

        return top_k;
    }

    TopPSampler::TopPSampler(float p)
        : p_(p) {
        validate_top_p(p_, "TopPSampler");
    }

    int TopPSampler::sample(const float* logits, int vocab_size, float temperature, uint64_t seed) {
        validate_sampling_inputs(vocab_size, temperature, "TopPSampler::sample");

        std::vector<float> probs(static_cast<size_t>(vocab_size));
        float max_logit = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            max_logit = std::max(max_logit, logits[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp((logits[i] - max_logit) / temperature);
            sum += probs[i];
        }

        for (int i = 0; i < vocab_size; ++i) {
            probs[i] /= sum;
        }

        std::vector<std::pair<int, float>> sorted(static_cast<size_t>(vocab_size));
        for (int i = 0; i < vocab_size; ++i) {
            sorted[i] = {i, probs[i]};
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; ++i) {
            cumsum += sorted[i].second;
            if (cumsum >= p_) {
                cutoff = i + 1;
                break;
            }
        }

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float target = dist(rng) * cumsum;
        cumsum = 0.0f;
        for (int i = 0; i < cutoff; ++i) {
            cumsum += sorted[i].second;
            if (cumsum >= target) {
                return sorted[i].first;
            }
        }

        return sorted[0].first;
    }

    BeamSearchManager::BeamSearchManager(BlockManager* block_manager, const BeamSearchConfig& config)
        : block_manager_(block_manager),
          config_(config),
          rng_(42) {
        validate_beam_config(config_, "BeamSearchManager");
    }

    void BeamSearchManager::configure(const BeamSearchConfig& config) {
        // Validated here as well as in the constructor: configure() is the documented way to
        // change the bounds after construction, and a bad config used to survive until the
        // first decode step, after the forward pass had already produced logits.
        validate_beam_config(config, "BeamSearchManager::configure");
        config_ = config;
    }

    void BeamSearchManager::initialize_beams(int prompt_length) {
        hypotheses_.clear();
        hypotheses_.reserve(config_.max_beams);

        for (int i = 0; i < config_.max_beams; ++i) {
            BeamHypothesis hyp;
            hyp.sequence_id = next_sequence_id_++;
            hyp.tokens = {};
            hyp.score = 0.0f;
            hyp.log_prob = 0.0f;
            hyp.length = prompt_length;
            hyp.finished = false;
            hyp.parent_beam = -1;
            hyp.kv_source_sequence = -1;
            hypotheses_.push_back(hyp);
        }
    }

    void BeamSearchManager::compute_length_normalized_scores() {
        // score is what prune_beams() and the final sort rank on, so it must be written on
        // both paths. Leaving it alone when length normalisation is disabled left every
        // hypothesis at its initial 0.0f, and the ranking then sorted on a constant:
        // measured with four beams whose true log_probs were -14.62, -13.62, -14.42 and
        // -15.22, the returned order was 14.62, 13.62, 14.42, 15.22 -- not monotone, i.e.
        // an arbitrary order presented as a shortest-first beam list.
        if (!config_.enable_length_norm) {
            for (auto& hyp : hypotheses_) {
                hyp.score = hyp.log_prob;
            }
            return;
        }

        for (auto& hyp : hypotheses_) {
            if (hyp.length > 0) {
                float length_norm = std::pow(hyp.length, config_.length_penalty);
                hyp.score = hyp.log_prob / length_norm;
            } else {
                hyp.score = hyp.log_prob;
            }
        }
    }

    void BeamSearchManager::prune_beams() {
        compute_length_normalized_scores();

        std::sort(hypotheses_.begin(), hypotheses_.end(), [](const BeamHypothesis& a, const BeamHypothesis& b) { return a.score > b.score; });

        if (static_cast<int>(hypotheses_.size()) > config_.max_beams) {
            hypotheses_.resize(config_.max_beams);
        }
    }

    void BeamSearchManager::rebase_scores() {
        if (hypotheses_.empty()) {
            return;
        }

        float max_log_prob = hypotheses_[0].log_prob;
        for (const auto& hyp : hypotheses_) {
            max_log_prob = std::max(max_log_prob, hyp.log_prob);
        }

        const float threshold = -1e4f;
        if (max_log_prob < threshold) {
            const float rebase_factor = max_log_prob;
            for (auto& hyp : hypotheses_) {
                hyp.log_prob -= rebase_factor;
            }
            for (auto& entry : trace_) {
                entry.score -= rebase_factor;
            }
        }
    }

    int BeamSearchManager::sample_token(const float* logits, int vocab_size, uint64_t seed) {
        // The branch table used to read:
        //
        //   top_p < 1.0            -> TopPSampler(top_p)
        //   else if top_k > 1      -> top-k, then draw with r = (seed % 1000) / 1000
        //   else                   -> TopPSampler(0.9f)      <- hardcoded 0.9
        //
        // Three defects came out of that. The final branch ignored the configuration
        // entirely: the default config is top_k = 1, top_p = 1.0, which every caller reads as
        // greedy, and it sampled with top_p = 0.9 instead -- measured on 100 emitted tokens
        // with a clear argmax, 56 of them were not the argmax. The top-k branch drew
        // r = (seed % 1000) / 1000, and search() builds the seed as step * 1000 + beam, so r
        // was beam / 1000: every beam in a step drew essentially the same number, below the
        // first atom's probability whenever that atom exceeded beam/1000. With four equally
        // likely tokens and four beams, 100 of 100 draws landed on token 1 and all four beams
        // emitted an identical sequence. The branch also never consulted top_p once top_k > 1,
        // so a (top_k, top_p) pair behaved as top_k alone.
        //
        // The contract now: top_k = 1 is greedy (the argmax, no sampling); otherwise top-k
        // narrows the candidates and top-p narrows them further, and the draw comes from the
        // sampler's own mt19937 rather than from the raw seed.
        const int effective_k = std::min(config_.top_k, vocab_size);

        if (effective_k <= 1) {
            int best = 0;
            for (int i = 1; i < vocab_size; ++i) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            return best;
        }

        TopKSampler k_sampler(effective_k);
        const auto candidates = k_sampler.sample(logits, vocab_size, config_.temperature, seed);

        // top_p = 1 keeps every one of the k candidates; a smaller value keeps the shortest
        // prefix whose cumulative probability reaches p, and always at least one token.
        size_t keep = candidates.size();
        if (config_.top_p < 1.0f) {
            keep = 0;
            float cumulative = 0.0f;
            for (const auto& candidate : candidates) {
                cumulative += candidate.second;
                ++keep;
                if (cumulative >= config_.top_p) {
                    break;
                }
            }
            if (keep == 0) {
                keep = 1;
            }
        }

        float total = 0.0f;
        for (size_t i = 0; i < keep; ++i) {
            total += candidates[i].second;
        }
        if (!(total > 0.0f)) {
            return candidates[0].first;
        }

        // Independent of the seed's low-order structure: the seed selects the stream, this
        // draw selects within it.
        std::mt19937 rng(static_cast<uint32_t>(seed));
        std::uniform_real_distribution<float> dist(0.0f, total);
        const float target = dist(rng);

        float cumulative = 0.0f;
        for (size_t i = 0; i < keep; ++i) {
            cumulative += candidates[i].second;
            if (target <= cumulative) {
                return candidates[i].first;
            }
        }
        return candidates[keep - 1].first;
    }

    void BeamSearchManager::clear_trace() {
        trace_.clear();
    }

    std::vector<BeamHypothesis> BeamSearchManager::search(const memory::Buffer<float>& prompt_embeddings,
                                                          int prompt_length,
                                                          int vocab_size,
                                                          const stream::Stream& stream,
                                                          std::function<void(memory::Buffer<float>&, const std::vector<int64_t>&, const stream::Stream&)> forward_fn) {
        // vocab_size is passed straight to the logits Buffer, so a non-positive value
        // surfaced from cudaMalloc as "invalid argument" or "out of memory" -- an allocator
        // message for a caller error. prompt_length sizes the generated region: when it is
        // not smaller than max_length the loop body never runs and search() returns
        // max_beams empty hypotheses, which reads as a successful search.
        if (vocab_size <= 0) {
            throw std::invalid_argument("BeamSearchManager::search: vocab_size must be positive, got " + std::to_string(vocab_size));
        }
        if (prompt_length < 0 || prompt_length >= config_.max_length) {
            throw std::invalid_argument("BeamSearchManager::search: prompt_length must be in [0, max_length), got " + std::to_string(prompt_length) + " with max_length "
                                        + std::to_string(config_.max_length));
        }

        initialize_beams(prompt_length);
        clear_trace();

        memory::Buffer<float> logits(vocab_size);

        for (int step = 0; step < config_.max_length - prompt_length; ++step) {
            std::vector<int64_t> seq_ids;
            for (const auto& hyp : hypotheses_) {
                if (!hyp.finished) {
                    seq_ids.push_back(hyp.sequence_id);
                }
            }

            if (seq_ids.empty()) {
                break;
            }

            forward_fn(logits, seq_ids, stream);

            next_hypotheses_.clear();
            int beam_idx = 0;

            // logits is a device buffer; the softmax/sampling below is host-side,
            // so stage a host copy first (dereferencing Battery::data() on the CPU
            // would read device memory — the same segfault class the speculative
            // runner guards against).
            std::vector<float> h_logits(vocab_size);
            logits.copy_to(h_logits.data(), vocab_size);
            const float* logits_data = h_logits.data();

            validate_sampling_inputs(vocab_size, config_.temperature, "BeamSearch::step");
            std::vector<float> probs(static_cast<size_t>(vocab_size));
            float max_logit = logits_data[0];
            for (int i = 1; i < vocab_size; ++i) {
                max_logit = std::max(max_logit, logits_data[i]);
            }
            float sum = 0.0f;
            for (int i = 0; i < vocab_size; ++i) {
                probs[i] = std::exp((logits_data[i] - max_logit) / config_.temperature);
                sum += probs[i];
            }
            for (int i = 0; i < vocab_size; ++i) {
                probs[i] /= sum;
            }

            for (size_t i = 0; i < hypotheses_.size() && beam_idx < config_.max_beams; ++i) {
                auto& hyp = hypotheses_[i];

                uint64_t seed = static_cast<uint64_t>(step) * 1000 + beam_idx;
                int token = sample_token(logits_data, vocab_size, seed);

                BeamHypothesis new_hyp;
                new_hyp.sequence_id = next_sequence_id_++;
                new_hyp.tokens = hyp.tokens;
                new_hyp.tokens.push_back(token);
                new_hyp.length = hyp.length + 1;
                new_hyp.parent_beam = static_cast<int>(i);
                new_hyp.kv_source_sequence = hyp.sequence_id;

                float token_log_prob = std::log(probs[token] + 1e-10f);
                new_hyp.log_prob = hyp.log_prob + token_log_prob;

                new_hyp.finished = (token == config_.eos_token_id);

                // enable_reuse is a request, not a guarantee: it needs a BlockManager to
                // fork from, and the constructor accepts a null one for callers that only
                // want beam bookkeeping. Dereferencing it unconditionally crashed the
                // process under ASan on the DEFAULT config (enable_reuse = true), so the
                // header's "may be null" was a contract no code honoured.
                if (config_.enable_reuse && block_manager_ != nullptr && hyp.sequence_id >= 0) {
                    auto* kv_cache = block_manager_->get_kv_cache();
                    if (kv_cache) {
                        int num_blocks = 1;
                        kv_cache->fork_prefix_blocks(hyp.sequence_id, new_hyp.sequence_id, num_blocks);
                    }
                }

                float normalized = new_hyp.length > 0 ? static_cast<float>(new_hyp.log_prob / std::pow(new_hyp.length, config_.length_penalty)) : new_hyp.log_prob;

                BeamSearchTraceEntry entry {.step = step,
                                            .beam_id = static_cast<int>(i),
                                            .token = token,
                                            .score = new_hyp.log_prob,
                                            .normalized_score = normalized,
                                            .log_prob = token_log_prob,
                                            .sequence_id = new_hyp.sequence_id};
                trace_.push_back(entry);

                next_hypotheses_.push_back(new_hyp);
                beam_idx++;
            }

            hypotheses_ = std::move(next_hypotheses_);
            prune_beams();

            if (config_.rebase_threshold > 0 && step > 0 && step % config_.rebase_threshold == 0) {
                rebase_scores();
            }
        }

        compute_length_normalized_scores();

        std::sort(hypotheses_.begin(), hypotheses_.end(), [](const BeamHypothesis& a, const BeamHypothesis& b) { return a.score > b.score; });

        return hypotheses_;
    }

    std::string BeamSearchManager::export_trace_json() const {
        std::ostringstream oss;
        oss << "{\"traces\":[";
        for (size_t i = 0; i < trace_.size(); ++i) {
            const auto& e = trace_[i];
            if (i > 0) {
                oss << ",";
            }
            oss << "{" << "\"step\":" << e.step << ",\"beam\":" << e.beam_id << ",\"token\":" << e.token << ",\"score\":" << (std::isfinite(e.score) ? e.score : 0.0f)
                << ",\"norm_score\":" << (std::isfinite(e.normalized_score) ? e.normalized_score : 0.0f) << ",\"seq\":" << e.sequence_id << "}";
        }
        oss << "]}";
        return oss.str();
    }

    std::string BeamSearchManager::export_trace_csv() const {
        std::ostringstream oss;
        oss << "step,beam,token,score,norm_score,sequence\n";
        for (const auto& e : trace_) {
            oss << e.step << "," << e.beam_id << "," << e.token << "," << e.score << "," << e.normalized_score << "," << e.sequence_id << "\n";
        }
        return oss.str();
    }

    BeamSearchManager::TraceStats BeamSearchManager::get_trace_stats() const {
        TraceStats stats {};
        stats.total_steps = 0;

        if (!trace_.empty()) {
            int max_step = trace_[0].step;
            for (const auto& e : trace_) {
                max_step = std::max(max_step, e.step);
            }
            stats.total_steps = max_step + 1;

            float sum_score = 0.0f;
            float sum_norm = 0.0f;
            int count = 0;

            for (const auto& e : trace_) {
                sum_score += e.score;
                sum_norm += e.normalized_score;
                count++;
            }

            stats.avg_beam_width = count > 0 ? count / std::max(1, stats.total_steps) : 0;
            stats.avg_score = count > 0 ? sum_score / count : 0.0f;
            stats.avg_length_norm = count > 0 ? sum_norm / count : 0.0f;
        }

        return stats;
    }

}  // namespace cuda::inference
