#include "cuda/inference/beam_search.h"
#include "cuda/device/error.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace cuda::inference {

TopKSampler::TopKSampler(int k) : k_(k) {}

std::vector<std::pair<int, float>> TopKSampler::sample(
    const float* logits,
    int vocab_size,
    float temperature,
    uint64_t seed
) {
    const int effective_k = std::min(k_, vocab_size);
    std::vector<std::pair<int, float>> top_k(effective_k);

    std::vector<float> probs(vocab_size);
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
    std::partial_sort(
        indices.begin(),
        indices.begin() + effective_k,
        indices.end(),
        [&probs](int a, int b) { return probs[a] > probs[b]; }
    );

    std::mt19937 rng(static_cast<uint64_t>(seed));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < effective_k; ++i) {
        if (i > 0 && std::abs(probs[indices[i]] - probs[indices[i-1]]) < 1e-6f) {
            if (dist(rng) < 0.5f) {
                std::swap(indices[i], indices[i-1]);
            }
        }
        top_k[i] = {indices[i], probs[indices[i]]};
    }

    return top_k;
}

TopPSampler::TopPSampler(float p) : p_(p) {}

int TopPSampler::sample(
    const float* logits,
    int vocab_size,
    float temperature,
    uint64_t seed
) {
    std::vector<float> probs(vocab_size);
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

    std::vector<std::pair<int, float>> sorted(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        sorted[i] = {i, probs[i]};
    }
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

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

BeamSearchManager::BeamSearchManager(
    BlockManager* block_manager,
    const BeamSearchConfig& config
) : block_manager_(block_manager), config_(config), rng_(42) {}

void BeamSearchManager::configure(const BeamSearchConfig& config) {
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
    if (!config_.enable_length_norm) return;

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

    std::sort(hypotheses_.begin(), hypotheses_.end(),
        [](const BeamHypothesis& a, const BeamHypothesis& b) {
            return a.score > b.score;
        });

    if (static_cast<int>(hypotheses_.size()) > config_.max_beams) {
        hypotheses_.resize(config_.max_beams);
    }
}

void BeamSearchManager::rebase_scores() {
    if (hypotheses_.empty()) return;

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

int BeamSearchManager::sample_token(
    const float* logits,
    int vocab_size,
    uint64_t seed
) {
    if (config_.top_p < 1.0f) {
        TopPSampler sampler(config_.top_p);
        return sampler.sample(logits, vocab_size, config_.temperature, seed);
    } else if (config_.top_k > 1) {
        TopKSampler sampler(config_.top_k);
        auto top_k = sampler.sample(logits, vocab_size, config_.temperature, seed);

        float r = static_cast<float>(seed % 1000) / 1000.0f;
        float cumsum = 0.0f;
        for (const auto& [token, prob] : top_k) {
            cumsum += prob;
            if (r <= cumsum) return token;
        }
        return top_k[0].first;
    } else {
        TopPSampler sampler(0.9f);
        return sampler.sample(logits, vocab_size, config_.temperature, seed);
    }
}

void BeamSearchManager::clear_trace() {
    trace_.clear();
}

std::vector<BeamHypothesis> BeamSearchManager::search(
    const memory::Buffer<float>& prompt_embeddings,
    int prompt_length,
    int vocab_size,
    const stream::Stream& stream,
    std::function<void(memory::Buffer<float>&, const std::vector<int64_t>&, const stream::Stream&)>
        forward_fn
) {
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

        if (seq_ids.empty()) break;

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

        std::vector<float> probs(vocab_size);
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

            if (config_.enable_reuse && hyp.sequence_id >= 0) {
                auto* kv_cache = block_manager_->get_kv_cache();
                if (kv_cache) {
                    int num_blocks = 1;
                    kv_cache->fork_prefix_blocks(
                        hyp.sequence_id, new_hyp.sequence_id, num_blocks);
                }
            }

            float normalized = new_hyp.length > 0
                ? static_cast<float>(new_hyp.log_prob / std::pow(new_hyp.length, config_.length_penalty))
                : new_hyp.log_prob;

            BeamSearchTraceEntry entry{
                .step = step,
                .beam_id = static_cast<int>(i),
                .token = token,
                .score = new_hyp.log_prob,
                .normalized_score = normalized,
                .log_prob = token_log_prob,
                .sequence_id = new_hyp.sequence_id
            };
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

    std::sort(hypotheses_.begin(), hypotheses_.end(),
        [](const BeamHypothesis& a, const BeamHypothesis& b) {
            return a.score > b.score;
        });

    return hypotheses_;
}

std::string BeamSearchManager::export_trace_json() const {
    std::ostringstream oss;
    oss << "{\"traces\":[";
    for (size_t i = 0; i < trace_.size(); ++i) {
        const auto& e = trace_[i];
        if (i > 0) oss << ",";
        oss << "{"
            << "\"step\":" << e.step
            << ",\"beam\":" << e.beam_id
            << ",\"token\":" << e.token
            << ",\"score\":" << (std::isfinite(e.score) ? e.score : 0.0f)
            << ",\"norm_score\":" << (std::isfinite(e.normalized_score) ? e.normalized_score : 0.0f)
            << ",\"seq\":" << e.sequence_id
            << "}";
    }
    oss << "]}";
    return oss.str();
}

std::string BeamSearchManager::export_trace_csv() const {
    std::ostringstream oss;
    oss << "step,beam,token,score,norm_score,sequence\n";
    for (const auto& e : trace_) {
        oss << e.step << ","
            << e.beam_id << ","
            << e.token << ","
            << e.score << ","
            << e.normalized_score << ","
            << e.sequence_id << "\n";
    }
    return oss.str();
}

BeamSearchManager::TraceStats BeamSearchManager::get_trace_stats() const {
    TraceStats stats{};
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
