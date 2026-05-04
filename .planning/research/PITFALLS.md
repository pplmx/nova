# Pitfalls Research — Transformer Inference Optimization

**Domain:** CUDA transformer inference (Speculative Decoding, Beam Search, KV Cache)
**Researched:** 2026-05-05
**Confidence:** HIGH

## Context

Adding Speculative Decoding, Beam Search, and KV Cache improvements to an existing CUDA transformer inference library. These features interact deeply with attention mechanisms, memory management, and generation loops.

---

## Critical Pitfalls

### Pitfall 1: Speculative Decoding — Distribution Mismatch

**What goes wrong:**
Generated output diverges from target model distribution, producing incorrect or nonsensical text despite passing rejection sampling.

**Why it happens:**
The rejection sampling logic compares draft and target model probabilities incorrectly. Common mistakes:
- Using argmax from draft instead of sampling
- Incorrect handling of temperature during verification
- Comparing unnormalized logits instead of proper probability distributions
- Floating-point precision loss when computing acceptance ratios near 1.0

**How to avoid:**
```cpp
// CORRECT: Proper acceptance check per token
float draft_prob = softmax(draft_logits)[draft_token];
float target_prob = softmax(target_logits)[draft_token];
float acceptance = fminf(1.0f, target_prob / draft_prob);

// WRONG: Comparing logits directly
if (draft_logits[token] >= target_logits[token]) // FAILS
```

**Warning signs:**
- Output quality degrades at longer sequence lengths
- Acceptance rate too high (>95%) indicates loose verification
- Different outputs on repeated runs with same seed

**Phase to address:** Speculative Decoding kernel implementation phase

---

### Pitfall 2: Speculative Decoding — KV Cache Contamination

**What goes wrong:**
Rejected speculative tokens corrupt the KV cache, causing subsequent tokens to be generated incorrectly.

**Why it happens:**
When a speculative token is rejected, the KV cache still contains the attention states computed for that token. The next accepted token then attends over incorrect historical context.

**How to avoid:**
- Cache prefix tokens only (before speculative phase)
- Use sequence IDs to track which tokens are "official" vs speculative
- Implement rollback mechanism that restores KV cache to last accepted state
- Consider copy-on-write KV cache entries during speculation

```cpp
// Strategy 1: Snapshot before speculation
KVCache snapshot = kv_cache.clone();
run_speculative_phase(kv_cache, drafts);
if (rejected) {
    kv_cache = snapshot;  // Rollback
}

// Strategy 2: Prefix-only attention
bool is_prefix_token = (token_id < num_accepted_tokens);
attention_mask.add_bias(is_prefix_token ? 0.0f : -inf);
```

**Warning signs:**
- Errors only appear after first rejection
- Longer rejection chains produce worse output
- Output improves when speculation depth is reduced to 1

**Phase to address:** Integration testing phase

---

### Pitfall 3: Beam Search — Beam Width Explosion

**What goes wrong:**
Memory usage grows unbounded with sequence length, causing OOM errors on long generations.

**Why it happens:**
Naive beam search stores `beam_width * sequence_length * vocab_size` scores at each step. With beam_width=8 and sequence_length=1000, this is significant.

**How to avoid:**
- Store only top-k candidate scores, not full beam state
- Prune beams aggressively based on score delta threshold
- Use compact beam representation (store parent pointers, not full history)
- Consider memory-mapped storage for archival beams

```cpp
// Instead of storing full beam matrix:
float beams[beam_width][seq_len][vocab_size];  // FAILS at scale

// Store only necessary state:
struct BeamState {
    float score;
    int parent_idx;
    int token;
    // KV cache can be shared for common prefixes
};
std::vector<BeamState> active_beams;  // Compact representation
```

**Warning signs:**
- Memory usage grows linearly with sequence length (should be sublinear)
- OOM errors only occur past certain sequence threshold
- Profiler shows memory allocation spikes at each decode step

**Phase to address:** Beam Search implementation phase

---

### Pitfall 4: Beam Search — Score Underflow

**What goes wrong:**
Beam scores become -inf due to repeated multiplication of probabilities, causing all beams to terminate prematurely or produce garbage.

**Why it happens:**
Log probabilities are summed over long sequences. Each token contributes a negative logprob (~0.5-5.0). After 500 tokens with avg logprob of -2.0, cumulative score is -1000.0 — still representable. But after 2000 tokens with poor tokens (logprob -10.0), you get -20000.0, which underflows to -inf on float32.

**How to avoid:**
- Normalize scores per step: `score = score / length_penalty`
- Use length normalization: `score = cumulative_logprob / (seq_len ^ alpha)`
- Switch to float64 for cumulative scores (acceptable overhead for small beam state)
- Track minimum score and rebase: `score -= min_score` every N steps

```cpp
float normalized_score(float cumulative_logprob, int seq_len, float alpha) {
    // Length-normalized score prevents underflow
    return cumulative_logprob / std::pow(seq_len, alpha);
}

// Or rebase periodically:
if (step % 100 == 0) {
    float min_score = *std::min_element(scores.begin(), scores.end());
    for (auto& score : scores) score -= min_score;
}
```

**Warning signs:**
- Generation stops at exact same length across different prompts
- Beam diversity drops to zero at longer sequences
- Inf/nan appears in scores but not in logits

**Phase to address:** Beam Search kernel implementation phase

---

### Pitfall 5: KV Cache — PagedAttention Fragmentation

**What goes wrong:**
KV cache allocation becomes fragmented across many small blocks, causing memory allocator overhead and cache misses.

**Why it happens:**
PagedAttention allocates memory in fixed-size blocks per token. With variable-length requests and concurrent sequences, the allocator produces many small holes. Over time, this fragments memory even when total free space is sufficient.

**How to avoid:**
- Implement block compaction during idle cycles
- Use segregated free lists with power-of-two block sizes
- Set `CUDA malloc pool reuse` for KV cache allocations
- Profile allocator hit rate and trigger compaction below threshold

```cpp
// Profile this metric:
float fragmentation_ratio = (bytes_allocated - bytes_in_use) / bytes_allocated;
if (fragmentation_ratio > 0.3) {
    compact_kv_cache();  // Defragment during low-usage window
}

// Use CUDA memory pool for KV cache
cudaMemPool_t kv_pool;
cudaMempoolCreate(&kv_pool, ...);
cudaMallocFromPoolAsync(&kv_ptr, size, kv_pool, stream);
```

**Warning signs:**
- Allocating small blocks (<512 bytes) repeatedly
- Fragmentation ratio increases over time
- Memory usage spikes when switching between batch sizes
- Profiler shows cudaMalloc/cudaFree in hot path

**Phase to address:** KV Cache optimization phase (memory management)

---

### Pitfall 6: KV Cache — Attention Sink Dominance

**What goes wrong:**
KV cache is dominated by attention to first token ("attention sink"), wasting memory on low-value entries while evicting important recent tokens.

**Why it happens:**
Transformers exhibit attention sink phenomenon — excessive attention to first token. Without management, this first token's KV entries consume cache space disproportionately.

**How to avoid:**
- Implement importance-based cache eviction (evict low-attention entries)
- Use streaming cache with configurable retention policy
- Apply attention-aware prefetching based on attention patterns
- Consider specialized "sink" storage separate from LRU cache

```cpp
struct CacheEntry {
    int token_id;
    float attention_weight;  // Rolling average
    KVBlock* kv_data;
};

// Eviction policy: remove lowest attention_weight first
void evict_low_attention_entries(int num_to_evict) {
    auto sorted = entries | std::views::sorted_by_attention();
    for (int i = 0; i < num_to_evict; ++i) {
        free(sorted[i].kv_data);
    }
}
```

**Warning signs:**
- Cache hit rate drops despite recent-token reuse patterns
- First token dominates cache usage
- Attention visualization shows heavy first-token attention

**Phase to address:** KV Cache optimization phase (policy implementation)

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Flat KV cache array (no paging) | Simpler implementation | OOM on long sequences | MVP only, never production |
| Reject all speculation on first failure | Correctness guarantee | Loses all speedup | Debug only |
| Single beam width | Simpler code | Suboptimal quality | Initial testing |
| Disable cache eviction | Predictable memory | Memory exhaustion | Short-context apps only |
| Synchronous KV cache ops | Simpler concurrency | Throughput loss | Single-sequence only |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Tensor Parallelism | KV cache not synchronized across ranks | All-gather after each layer, not just final |
| Mixed batch sizes | Block allocator fragmentation | Bin-packing based on max_seq_len |
| Dynamic sequence lengths | Pre-allocated buffer waste | On-demand allocation with growth strategy |
| CUDA Graphs | KV cache operations break capture | Mark KV ops as non-graph or use conditional nodes |
| Attention backprop | KV cache gradients accumulate | Clear gradients each forward pass |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Speculative rejection overhead | GPU util < 50% during verification | Batch verification kernels | >30% rejection rate |
| Beam scoring O(n*vocab) | Decode step time grows with vocab | Top-k sampling kernel | vocab > 50k |
| KV cache miss penalty | 3-5x slower on cache miss | Prefetch next-token KV | >100ms latency spike |
| Lock contention on cache | GPU stalls waiting for allocator | Per-stream allocation pools | >8 concurrent sequences |
| Attention computation O(n^2) | Quadratic decode time growth | FlashAttention with caching | seq_len > 2048 |

---

## Security Mistakes

Domain-specific issues beyond general CUDA security:

| Mistake | Risk | Prevention |
|---------|------|------------|
| Unvalidated speculation depth | DoS via memory exhaustion | Cap max_speculation_depth, reject if exceeded |
| Unbounded beam width | OOM leading to crash | Validate beam_width <= MAX_ALLOWED |
| Untrusted batch sizes | Integer overflow in allocation | Saturating arithmetic for size calculation |
| KV cache poisoning | Malicious KV injection | Validate sequence IDs match expectations |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Hidden latency cliff | Sudden 10x slowdown at long context | Warn when approaching limit |
| Non-deterministic output | Same prompt gives different answer | Offer seed-locked mode with explanation |
| OOM without recovery | Generation fails mid-output | Save partial output, allow resume |
| Speculation without fallback | Garbage output if speculation fails | Always produce non-speculative baseline |

---

## "Looks Done But Isn't" Checklist

For CUDA kernel verification of these features:

- [ ] **Speculative Decoding:** Output distribution matches reference (verify with KL divergence)
- [ ] **Speculative Decoding:** Memory leak test — run 1000 speculative loops, verify memory stable
- [ ] **Speculative Decoding:** Rejection at step N doesn't corrupt step N+1 output
- [ ] **Beam Search:** Beam width scaling test — verify memory scales correctly with beam_width
- [ ] **Beam Search:** Score comparison correctness — verify top beam is actually best
- [ ] **Beam Search:** Edge case at EOS — verify beam terminates correctly with length penalties
- [ ] **KV Cache:** Cache hit rate test — verify hits increase with temporal locality
- [ ] **KV Cache:** Fragmentation test — verify fragmentation stays bounded (< 20%)
- [ ] **KV Cache:** Concurrent access test — verify no race conditions with parallel sequences
- [ ] **KV Cache:** Eviction correctness — verify evicted entries are truly unused
- [ ] **Integration:** Mixed feature test — all features work together without interference
- [ ] **Integration:** Long sequence stress test — verify behavior at seq_len = max_supported

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| KV Cache contamination | MEDIUM | Clear cache from last checkpoint, re-decode |
| Score underflow | LOW | Rebase scores, restart beam expansion |
| Fragmentation explosion | MEDIUM | Pause new allocations, compact in background |
| Speculation divergence | LOW | Fall back to non-speculative, log for analysis |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Distribution mismatch | Kernel implementation | KL divergence test against reference |
| KV cache contamination | Integration testing | End-to-end test with forced rejections |
| Beam width explosion | Memory architecture design | Memory profiling at max sequence |
| Score underflow | Kernel implementation | Stress test at seq_len=2000 |
| PagedAttention fragmentation | Memory management design | Fragmentation ratio monitoring |
| Attention sink dominance | Cache policy design | Attention weight distribution analysis |

---

## Sources

- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (arXiv:2302.01318)
- NVIDIA Transformer Engine documentation
- vLLM speculative decoding implementation (known pitfalls documented in issues)
- Hugging Face generation utilities (beam search edge cases)
- FlashAttention paper (KV cache integration considerations)
- Community discussions: /r/MachineLearning, LangChain Discord, GitHub issues

---

*Pitfalls research for: Transformer Inference Optimization*
*Researched: 2026-05-05*
