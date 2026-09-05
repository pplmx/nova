#include "cuda/inference/speculative_decoding.h"
#include "cuda/inference/beam_search.h"
#include "cuda/device/error.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace cuda::inference {

namespace {

// Reject any configuration flag that selects a feature this module does not
// implement (ISS-010): the config used to silently advertise tree attention,
// async drafting, EAGLE-3 and xgrammar while every path ran plain chain-only
// drafting — the per-flag no-ops made the advertised features a silent lie.
// Fail fast at the two configuration boundaries (constructor / configure) so a
// caller opting in gets an explicit "not implemented" error instead of
// spending tokens on a decode that never did what the flags promised.
[[noreturn]] void throw_unimplemented(const char* feature, const char* flag) {
    throw std::runtime_error(
        std::string("SpeculativeDecodingRunner: ") + feature +
        " is not implemented (config flag " + flag +
        "); set " + flag + "=false to use the supported chain-only path");
}

void validate_implemented_features(const SpeculativeDecodingConfig& config) {
    if (config.enable_tree_attention) {
        throw_unimplemented("tree-masked draft attention", "enable_tree_attention");
    }
    if (config.enable_async_draft) {
        throw_unimplemented("asynchronous draft generation", "enable_async_draft");
    }
    if (config.enable_eagle3) {
        throw_unimplemented("EAGLE-3 draft heads", "enable_eagle3");
    }
    if (config.enable_xgrammar) {
        throw_unimplemented("xgrammar grammar constraints", "enable_xgrammar");
    }
}

int sample_from_logits(
    const float* logits,
    int vocab_size,
    float temperature,
    uint64_t seed
) {
    if (temperature <= 0.0f) {
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

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
    float target = static_cast<float>(seed % 10000) / 10000.0f;
    cumsum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += sorted[i].second;
        if (cumsum >= target) {
            return sorted[i].first;
        }
    }

    return sorted[0].first;
}

// Host-side probability of `token_id` under the softmax of a single
// next-token logits row (`n` entries, max-shifted for stability). Used by both
// verify_draft_tokens() (one shared row) and decode() (per-position rows).
float softmax_prob(const float* logits, int n, int token_id) {
    float max_logit = logits[0];
    for (int i = 1; i < n; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += std::exp(logits[i] - max_logit);
    }
    return std::exp(logits[token_id] - max_logit) / sum;
}

}  // anonymous namespace

void LogProbTracker::record(
    int token_id,
    float log_prob,
    float draft_log_prob,
    bool accepted
) {
    LogProbEntry entry{
        .token_id = token_id,
        .log_prob = log_prob,
        .draft_log_prob = draft_log_prob,
        .accepted = accepted
    };
    history_.push_back(entry);
}

float LogProbTracker::compute_total_kl_divergence() const {
    float total_kl = 0.0f;
    for (const auto& entry : history_) {
        float p = std::exp(entry.log_prob);
        float q = std::exp(entry.draft_log_prob);
        if (p > 0.0f && q > 0.0f) {
            total_kl += p * std::log(p / q);
        }
    }
    return total_kl;
}

float LogProbTracker::compute_average_kl_divergence() const {
    if (history_.empty()) return 0.0f;
    return compute_total_kl_divergence() / history_.size();
}

int LogProbTracker::num_accepted() const {
    int count = 0;
    for (const auto& entry : history_) {
        if (entry.accepted) count++;
    }
    return count;
}

int LogProbTracker::num_rejected() const {
    return static_cast<int>(history_.size()) - num_accepted();
}

void LogProbTracker::clear() {
    history_.clear();
}

SpeculativeDecodingRunner::SpeculativeDecodingRunner(
    BlockManager* block_manager,
    const SpeculativeDecodingConfig& config
) : block_manager_(block_manager), config_(config), logprob_tracker_() {
    validate_implemented_features(config_);
}

void SpeculativeDecodingRunner::configure(const SpeculativeDecodingConfig& config) {
    validate_implemented_features(config);
    config_ = config;
}

void SpeculativeDecodingRunner::snapshot_kv_state() {
    kv_snapshot_ = KVCacheSnapshot{};
    kv_snapshot_->sequence_ids.clear();
    kv_snapshot_->num_blocks.clear();

    for (const auto& [seq_id, seq] : block_manager_->get_active_sequences()) {
        (void)seq;
        kv_snapshot_->sequence_ids.push_back(seq_id);
        kv_snapshot_->num_blocks.push_back(static_cast<int>(seq->block_table.size()));
    }
}

void SpeculativeDecodingRunner::rollback_kv_state() {
    if (!kv_snapshot_) return;

    for (size_t i = 0; i < kv_snapshot_->sequence_ids.size(); ++i) {
        block_manager_->free_sequence(kv_snapshot_->sequence_ids[i]);
    }

    kv_snapshot_.reset();
}

void SpeculativeDecodingRunner::commit_kv_state() {
    kv_snapshot_.reset();
}

std::vector<int> SpeculativeDecodingRunner::generate_draft_tokens(
    const memory::Buffer<float>& prompt_embeddings,
    int prompt_length,
    const stream::Stream& stream,
    std::function<void(memory::Buffer<float>&, const std::vector<int64_t>&, bool, const stream::Stream&)>
        forward_fn
) {
    std::vector<int> draft_tokens;
    draft_tokens.reserve(config_.draft_depth);

    auto* seq = block_manager_->create_sequence(0, prompt_length + config_.max_draft_depth);
    int64_t seq_id = seq->id;

    const int vocab_size = config_.vocab_size > 0 ? config_.vocab_size : 32000;
    memory::Buffer<float> logits(vocab_size);

    std::mt19937 rng(static_cast<unsigned int>(seq_id));
    float temperature = config_.temperature > 0.0f ? config_.temperature : 0.8f;

    for (int step = 0; step < config_.draft_depth; ++step) {
        std::vector<int64_t> seq_ids = {seq_id};
        forward_fn(logits, seq_ids, false, stream);

        // sample_from_logits is a host function; the draft logits live on the
        // device, so stage a host copy before sampling (reading Buffer::data()
        // on the CPU would dereference device memory and segfault).
        std::vector<float> h_logits(vocab_size);
        logits.copy_to(h_logits.data(), vocab_size);
        uint64_t seed = static_cast<uint64_t>(step) * 1000 + rng();

        int token = sample_from_logits(h_logits.data(), vocab_size, temperature, seed);
        draft_tokens.push_back(token);

        block_manager_->append_tokens(seq_id, 1);
    }

    block_manager_->free_sequence(seq_id);

    return draft_tokens;
}

VerificationResult SpeculativeDecodingRunner::verify_draft_tokens(
    const std::vector<int>& draft_tokens,
    const memory::Buffer<float>& draft_logits,
    const memory::Buffer<float>& target_logits,
    const stream::Stream& stream
) {
    (void)stream;
    apply_tree_attention_mask(static_cast<int>(draft_tokens.size()), stream);

    VerificationResult result;
    result.tokens.reserve(draft_tokens.size());
    result.num_accepted = 0;
    result.kl_divergence = 0.0f;

    const int vocab_size = config_.vocab_size > 0 ? config_.vocab_size : 32000;

    // Use the actual logits buffer size: callers may pass buffers smaller than
    // config_.vocab_size (which defaults to a large value), and reading or
    // copying beyond the buffer is out-of-bounds.
    const size_t n = std::min(static_cast<size_t>(vocab_size),
                              std::min(draft_logits.size(), target_logits.size()));

    // The logits live on the device; stage host copies before the softmax
    // math below (reading Buffer::data() from the CPU segfaults).
    std::vector<float> h_draft(n);
    std::vector<float> h_target(n);
    draft_logits.copy_to(h_draft.data(), n);
    target_logits.copy_to(h_target.data(), n);
    const float* draft_data = h_draft.data();
    const float* target_data = h_target.data();

    if (n == 0) {
        return result;
    }

    // Both drafts share this single logits pair (the caller supplies one row
    // per model), so each token is scored against the same probabilities —
    // this is the single-position contract of verify_draft_tokens().
    constexpr float epsilon = 1e-8f;
    for (size_t i = 0; i < draft_tokens.size(); ++i) {
        int token_id = draft_tokens[i];
        if (token_id < 0 || static_cast<size_t>(token_id) >= n) {
            continue;
        }

        const float draft_prob = softmax_prob(draft_data, static_cast<int>(n), token_id);
        const float target_prob = softmax_prob(target_data, static_cast<int>(n), token_id);

        float acceptance;
        if (draft_prob > epsilon) {
            acceptance = std::fmin(1.0f, target_prob / draft_prob);
        } else {
            acceptance = 0.0f;
        }

        bool accepted = acceptance >= config_.acceptance_threshold;

        DraftToken dt{
            .token_id = token_id,
            .draft_prob = draft_prob,
            .target_prob = target_prob,
            .accepted = accepted
        };
        result.tokens.push_back(dt);

        if (accepted) {
            result.num_accepted++;
        }

        logprob_tracker_.record(token_id, std::log(target_prob + epsilon), std::log(draft_prob + epsilon), accepted);
    }

    result.kl_divergence = logprob_tracker_.compute_average_kl_divergence();

    return result;
}

float SpeculativeDecodingRunner::compute_kl_divergence(
    const std::vector<DraftToken>& tokens,
    const memory::Buffer<float>& draft_logits,
    const memory::Buffer<float>& target_logits
) const {
    (void)draft_logits;
    (void)target_logits;

    float kl = 0.0f;
    for (const auto& token : tokens) {
        if (token.draft_prob > 0.0f && token.target_prob > 0.0f) {
            kl += token.target_prob * std::log(token.target_prob / token.draft_prob);
        }
    }
    return kl;
}

void SpeculativeDecodingRunner::apply_tree_attention_mask(
    int num_draft_tokens,
    const stream::Stream& stream
) {
    (void)num_draft_tokens;
    (void)stream;
    // Chain-only drafts (the only path this runner generates) make the tree
    // mask the identity, so there is nothing to apply here. The constructor /
    // configure() reject enable_tree_attention=true, so this guard is a
    // defensive backstop in case a config reaches the runner some other way
    // (ISS-010): a requested tree mask must never be silently dropped.
    if (config_.enable_tree_attention) {
        throw std::runtime_error(
            "SpeculativeDecodingRunner::apply_tree_attention_mask: tree "
            "attention is not implemented — draft generation is chain-only "
            "(set enable_tree_attention=false)");
    }
}

std::vector<int> SpeculativeDecodingRunner::decode(
    const memory::Buffer<float>& prompt_embeddings,
    int prompt_length,
    const stream::Stream& stream,
    std::function<void(memory::Buffer<float>&, const std::vector<int64_t>&, bool, const stream::Stream&)>
        forward_fn
) {
    logprob_tracker_.clear();
    draft_tokens_ = generate_draft_tokens(prompt_embeddings, prompt_length, stream, forward_fn);

    snapshot_kv_state();

    const int k = static_cast<int>(draft_tokens_.size());
    const int vocab_size = config_.vocab_size > 0 ? config_.vocab_size : 32000;

    // No drafted tokens: nothing to verify. Commit (nothing changed) and
    // return empty rather than forwarding on an empty verify sequence.
    if (k == 0) {
        commit_kv_state();
        return {};
    }

    // Replay the draft pass on a fresh verify sequence and collect per-position
    // logits. At step j the sequence holds prompt + t_1..t_{j-1}, so both
    // models predict t_j there (the bool selects draft vs target on a
    // dual-model backend — same contract generate_draft_tokens() uses). The
    // old code appended (k-1) then k tokens (2k-1 total), which both polluted
    // the verify prefix and could exceed max_tokens when max_draft_depth < 2k-1.
    auto* verify_seq = block_manager_->create_sequence(0, prompt_length + k);
    const int64_t verify_seq_id = verify_seq->id;
    std::vector<int64_t> seq_ids = {verify_seq_id};

    memory::Buffer<float> draft_logits(vocab_size);
    memory::Buffer<float> target_logits(vocab_size);
    std::vector<std::vector<float>> h_draft(static_cast<size_t>(k));
    std::vector<std::vector<float>> h_target(static_cast<size_t>(k));
    for (int j = 0; j < k; ++j) {
        forward_fn(draft_logits, seq_ids, true, stream);
        forward_fn(target_logits, seq_ids, false, stream);
        h_draft[static_cast<size_t>(j)].resize(vocab_size);
        h_target[static_cast<size_t>(j)].resize(vocab_size);
        draft_logits.copy_to(h_draft[static_cast<size_t>(j)].data(), vocab_size);
        target_logits.copy_to(h_target[static_cast<size_t>(j)].data(), vocab_size);
        block_manager_->append_tokens(verify_seq_id, 1);
    }

    // Verify each draft token against the logits of its own position.
    std::vector<int> output_tokens;
    int num_accepted = 0;
    constexpr float epsilon = 1e-8f;
    for (int j = 0; j < k; ++j) {
        const int token_id = draft_tokens_[static_cast<size_t>(j)];
        const float draft_prob = softmax_prob(h_draft[static_cast<size_t>(j)].data(), vocab_size, token_id);
        const float target_prob = softmax_prob(h_target[static_cast<size_t>(j)].data(), vocab_size, token_id);

        float acceptance;
        if (draft_prob > epsilon) {
            acceptance = std::fmin(1.0f, target_prob / draft_prob);
        } else {
            acceptance = 0.0f;
        }

        const bool accepted = acceptance >= config_.acceptance_threshold;
        if (accepted) {
            num_accepted++;
            output_tokens.push_back(token_id);
        }
        logprob_tracker_.record(token_id, std::log(target_prob + epsilon), std::log(draft_prob + epsilon), accepted);
    }

    if (num_accepted < k) {
        rollback_kv_state();
    } else {
        commit_kv_state();
    }

    block_manager_->free_sequence(verify_seq_id);

    return output_tokens;
}

}  // namespace cuda::inference
