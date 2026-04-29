# CUDA Transformer Inference Pitfalls

**Project:** Transformer & Inference Optimization
**Domain:** CUDA Attention Kernel Implementation
**Context:** Adding FlashAttention, KV cache, and paged attention to existing CUDA library
**Researched:** 2026-04-29
**Confidence:** HIGH (based on FlashAttention official implementation, vLLM source analysis, and established ML optimization patterns)

## Executive Summary

This document catalogs common pitfalls when implementing attention optimization features (FlashAttention, KV cache, paged attention) in CUDA transformer libraries. Each pitfall includes root cause analysis, production consequences, and actionable mitigation strategies.

Key cross-cutting themes:
- **Softmax numerical stability** is the most common correctness bug in attention kernels
- **Memory layout assumptions** cause subtle correctness issues across different batch/sequence dimensions
- **Workspace allocation lifecycle** mismatches lead to crashes or memory corruption
- **Block table consistency** between CPU and GPU is frequently mishandled in paged attention
- **Backward pass communication** in tensor/pipeline parallelism has complex synchronization requirements

---

## 1. Attention Kernel Correctness Pitfalls

### 1.1 Softmax Numerical Overflow/Underflow

**What goes wrong:** Attention scores overflow or underflow during softmax computation, producing NaN outputs or incorrect probabilities.

**Why it happens:** The standard softmax formula `exp(x_i) / sum(exp(x_j))` is numerically unstable when `x_i` values are large. The exponential of large values overflows to infinity, and subtraction of large values causes underflow to zero.

```cpp
// WRONG: Naive softmax - numerically unstable
__device__ float naive_softmax(float score, float max_score, float sum_exp) {
    return exp(score - max_score) / sum_exp;  // exp of large negative = 0 (underflow)
    // exp of large positive = inf (overflow)
}

// Problematic input: scores = [1000, 1001, 1002]
// exp(1000) = inf (overflow)
// exp(-1) = 0.368, but all subtractions from 1002 give underflow
```

**Consequences:**
- NaN outputs propagate through the model
- Training divergence or inference hallucinations
- Extremely hard to debug: depends on input magnitude, only manifests at certain scales

**Prevention:**
```cpp
// CORRECT: Numerically stable softmax using max subtraction
template <int THREADS>
__device__ float softmax_kernel(float* scores, int seq_len, int tid) {
    // Step 1: Find max (will be most negative, preventing overflow)
    float thread_max = -INFINITY;
    for (int i = tid; i < seq_len; i += THREADS) {
        thread_max = fmaxf(thread_max, scores[i]);
    }
    
    // Warp-level reduction for max
    float max_val = warp_reduce_max(thread_max);
    
    // Step 2: Compute exp with stable subtraction
    float thread_sum = 0.0f;
    for (int i = tid; i < seq_len; i += THREADS) {
        float shifted = scores[i] - max_val;  // Always <= 0, no overflow
        thread_sum += expf(shifted);
    }
    
    // Step 3: Normalize
    float sum_exp = warp_reduce_sum(thread_sum);
    return expf(scores[tid] - max_val) / sum_exp;
}

// CORRECT: Alternative - use log-sum-exp trick
float stable_log_softmax(float* scores, int n) {
    float max_score = scores[0];
    for (int i = 1; i < n; i++) max_score = fmax(max_score, scores[i]);
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(scores[i] - max_score);
    
    return max_score + logf(sum);  // Log-sum-exp is numerically stable
}
```

**Detection:**
```cpp
// CORRECT: Sanity check in tests
void test_attention_numerical_stability() {
    std::vector<float> large_scores = {1000.0f, 1001.0f, 1002.0f, -1000.0f, -1001.0f};
    auto result = attention(large_scores);
    
    // Check no NaN/Inf
    for (float v : result) {
        ASSERT_FALSE(std::isnan(v)) << "NaN detected";
        ASSERT_FALSE(std::isinf(v)) << "Inf detected";
    }
    
    // Check sum to 1.0
    float sum = std::accumulate(result.begin(), result.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Probabilities don't sum to 1";
}
```

**Phase Recommendation:** Phase 1 (FlashAttention Integration) — Implement stable softmax from day one.

---

### 1.2 Incorrect Mask Handling in FlashAttention

**What goes wrong:** Attention masks are incorrectly applied, causing tokens to attend where they shouldn't or missing valid attention patterns.

**Why it happens:** FlashAttention requires specific mask integration. Common mistakes include:
- Applying masks after softmax instead of before
- Wrong mask shape/striding
- Ignoring causal masking in streaming scenarios

```cpp
// WRONG: Applying mask after softmax - corrupts probability distribution
__global__ void wrong_mask_kernel(float* output, float* scores, 
                                   bool* mask, int seq_len) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    // Compute softmax first
    float max_score = // ... reduction
    float exp_score = expf(scores[row * seq_len + col] - max_score);
    float sum_exp = // ... reduction
    
    // WRONG: Mask after softmax - wrong!
    if (!mask[row * seq_len + col]) {
        exp_score = 0.0f;  // Changes normalization
    }
    
    output[row * seq_len + col] = exp_score / sum_exp;
}

// WRONG: Incorrect stride for packed sequence masks
__global__ void wrong_stride_mask(float* output, float* scores,
                                   int32_t* cu_seqlens, int32_t* seq_ids) {
    // cu_seqlens has shape [batch_size + 1]
    // WRONG: Treating cu_seqlens as having same stride as scores
    int bid = seq_ids[col];  // Might be wrong if sequences packed differently
    int local_pos = col - cu_seqlens[bid];  // Incorrect offset calculation
}
```

**Consequences:**
- Data leakage in masked regions (attention to future tokens)
- Missing attention to valid tokens
- Incorrect results when sequences of different lengths are batched

**Prevention:**
```cpp
// CORRECT: Apply mask before softmax
__global__ void correct_mask_kernel(float* output, float* scores,
                                     bool* mask, int seq_len,
                                     float attn_scale) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    int idx = row * seq_len + col;
    
    // Apply mask BEFORE softmax
    float score = scores[idx];
    if (mask[idx]) {
        score = -INFINITY;  // Softmax treats -inf as 0 probability
    }
    
    // Then compute softmax on masked scores
    // ... stable softmax implementation ...
}

// CORRECT: Proper packed sequence mask handling
struct PackedSequenceMask {
    int32_t* cu_seqlens_q;  // Cumulative sequence lengths for queries
    int32_t* cu_seqlens_k;  // Cumulative sequence lengths for keys
    int32_t* seq_ids;       // Which sequence each position belongs to
    
    __device__ bool is_valid(int global_pos, int seq_len) {
        if (seq_ids == nullptr) return true;  // No packing
        
        int seq_id = seq_ids[global_pos];
        int seq_start = cu_seqlens_k[seq_id];
        int seq_len_actual = cu_seqlens_k[seq_id + 1] - seq_start;
        
        return (global_pos - seq_start) < seq_len_actual;
    }
};
```

**Phase Recommendation:** Phase 2 (KV Cache Integration) — Test mask handling with variable-length sequences.

---

### 1.3 Incorrect Dropout Mask Generation/Replication

**What goes wrong:** Attention dropout masks are inconsistently generated between forward and backward passes, or masks are incorrectly replicated across heads.

**Why it happens:** FlashAttention's dropout is deterministic with a seed. Common issues:
- Different dropout rates in forward/backward
- Seed not properly propagated
- Dropout mask not matching the scale factor

```cpp
// WRONG: Different dropout treatment in forward vs backward
// Forward pass:
float dropout_mask = (curand_uniform(&rng) < dropout_prob) ? 0.0f : 
                     (1.0f / (1.0f - dropout_prob));  // Scale factor applied
attn_weights *= dropout_mask;

// Backward pass (different code path):
if (curand_uniform(&rng_backward) < dropout_prob) {  // WRONG: Different RNG state
    grad_input = 0.0f;
}
```

**Consequences:**
- Training divergence (dropout mask differs between forward/backward)
- Incorrect gradients
- Non-deterministic training behavior

**Prevention:**
```cpp
// CORRECT: Use cuBLASLt dropout descriptor or explicit state propagation
struct AttentionDropout {
    uint64_t seed;
    uint64_t offset;
    float prob;
    
    // Forward pass generates and caches mask state
    DropoutState forward(float* output, const float* input, int elements) {
        DropoutState state;
        state.seed = curand4(&rng).x;
        state.offset = 0;
        
        // Kernel uses curand4 with state for deterministic dropout
        dropout_kernel<<<blocks, threads>>>(output, input, state, prob, elements);
        return state;
    }
    
    // Backward pass receives same state
    void backward(float* grad_input, const float* grad_output,
                  DropoutState state, const float* fwd_input) {
        // Same seed, same offset = same dropout pattern
        dropout_backward_kernel<<<blocks, threads>>>(
            grad_input, grad_output, state, prob, elements);
    }
};
```

---

## 2. KV Cache Memory Pitfalls

### 2.1 KV Cache Memory Fragmentation

**What goes wrong:** KV cache allocations cause severe memory fragmentation over time, leading to OOM failures even when total free memory exceeds required allocation.

**Why it happens:** Variable sequence lengths and dynamic allocation patterns create fragmentation:

```cpp
// WRONG: Naive per-sequence allocation causes fragmentation
void allocate_kv_cache(Request& req) {
    // Request 1: seq_len=512, needs 512 * 128 * sizeof(float) = 256KB
    // Request 2: seq_len=1000, needs 1000 * 128 * sizeof(float) = 500KB
    // Request 3: seq_len=256, needs 256 * 128 * sizeof(float) = 128KB
    // ...
    // After many allocations/deallocations: holes between blocks
    
    // Eventually: 2GB free but no contiguous 500KB block
    void* ptr = cudaMalloc(&ptr, needed_size);  // FAILS!
}

// WRONG: Mixing allocation sizes from different pools
void process_batch() {
    // KV cache blocks from pool A
    allocate_from_pool_A(batch_kv_cache, batch_size);
    
    // Attention outputs from pool B (different allocator)
    allocate_from_pool_B(attn_output, output_size);
    
    // Both pools fragment independently, neither is efficiently used
}
```

**Consequences:**
- OOM errors despite sufficient total memory
- Reduced effective batch size over time
- Performance degradation as allocations become more scattered

**Prevention:**
```cpp
// CORRECT: Use block-based KV cache with fixed block size
class KVCacheManager {
    static constexpr int BLOCK_SIZE = 64;  // Sequence positions per block
    static constexpr int NUM_BLOCKS = 16384;  // Pre-allocated blocks
    
    // All blocks are same size, eliminating fragmentation
    struct Block {
        float k_cache[MAX_HEADS][BLOCK_SIZE][HEAD_DIM];
        float v_cache[MAX_HEADS][BLOCK_SIZE][HEAD_DIM];
        int seq_len;  // Actual used length
        int ref_count;
    };
    
    std::vector<Block> blocks_;  // Single contiguous allocation
    std::unordered_map<int64_t, std::vector<int>> block_indices_;  // seq_id -> blocks
    
    Block* allocate(int64_t seq_id, int num_blocks_needed) {
        // Simple freelist - all blocks same size = no fragmentation
        std::vector<int> indices;
        for (int i = 0; i < num_blocks_needed; i++) {
            int idx = free_list_.back();
            free_list_.pop_back();
            indices.push_back(idx);
        }
        block_indices_[seq_id] = indices;
        return &blocks_[indices[0]];
    }
};

// CORRECT: Use CUDA's stream-ordered allocator for dynamic sizes
void setup_kv_cache_pool() {
    cudaMemPoolHandle_t pool;
    cudaMempoolCreate(&pool, {
        .allocType = cudaMemPoolAllocType::cudaMemPoolAllocation,
        .minGranularity = 4096,
        .releaseThreshold = 0,
    });
    
    // Enable reuse across streams
    cudaMemPoolSetAttribute(pool, 
        cudaMemPoolAttrReleaseThreshold, &release_threshold);
}
```

**Phase Recommendation:** Phase 3 (Memory Management) — Implement block-based KV cache allocation.

---

### 2.2 KV Cache Eviction Without Coordinate Update

**What goes wrong:** KV cache blocks are evicted but internal state still references old block indices, causing reads from garbage memory.

**Why it happens:** Eviction removes data but cached references become dangling pointers:

```cpp
// WRONG: Eviction without invalidating references
class KVCache {
    std::unordered_map<int64_t, std::vector<int>> seq_to_blocks_;
    
    void evict_if_needed() {
        if (memory_pressure_ > threshold_) {
            auto oldest = find_oldest_sequence();
            
            // Remove from eviction tracking
            seq_to_blocks_.erase(oldest);  // But blocks_[] still contains data!
            
            // Mark blocks as free
            for (int idx : blocks_[oldest]) {
                free_list_.push_back(idx);
            }
        }
    }
    
    // BUG: Someone still holds reference to evicted sequence
    void attention_with_cached_kv(Request& req) {
        auto it = seq_to_blocks_.find(req.seq_id);
        if (it == seq_to_blocks_.end()) {
            // req.seq_id was evicted but we're looking for it
            // This path won't be taken if we tracked eviction correctly
        }
    }
};
```

**Consequences:**
- Reading from freed/reallocated memory
- Silent data corruption
- Intermittent crashes that are hard to reproduce

**Prevention:**
```cpp
// CORRECT: Use sequence version tracking
struct KVCacheBlock {
    int64_t seq_id;
    uint64_t version;  // Incremented on each update
    bool in_use;
    
    bool is_valid(int64_t current_seq_id, uint64_t current_version) const {
        return in_use && seq_id == current_seq_id && version == current_version;
    }
};

class SafeKVCache {
    std::vector<KVCacheBlock> blocks_;
    std::atomic<uint64_t> global_version_{0};
    
    void evict(int64_t seq_id) {
        uint64_t new_version = ++global_version_;
        
        for (auto& block : blocks_) {
            if (block.seq_id == seq_id) {
                block.seq_id = -1;  // Mark as invalid
                block.in_use = false;
            }
        }
    }
    
    bool verify_block(int block_idx, int64_t expected_seq_id, 
                      uint64_t expected_version) {
        const auto& block = blocks_[block_idx];
        return block.is_valid(expected_seq_id, expected_version);
    }
};
```

---

## 3. FlashAttention Implementation Pitfalls

### 3.1 Incorrect Head Dimension Alignment

**What goes wrong:** FlashAttention fails or produces incorrect results when head dimension is not a multiple of the required alignment.

**Why it happens:** FlashAttention kernels require specific alignments for efficient execution:

```cpp
// WRONG: Assuming any head_dim works
// FlashAttention v2 requires head_dim % 8 == 0 for FP16
// FlashAttention v3 requires head_dim % 16 == 0 for FP16 with Tensor Cores

template <int HEAD_DIM>
__global__ void flash_attention_kernel(...) {
    // HEAD_DIM might be 96 (not multiple of 8)
    // FlashAttention will fall back to slow kernel or fail
}

// WRONG: Hardcoding alignment assumptions
constexpr int THREADS_PER_HEAD = 128;
constexpr int ELEMENTS_PER_THREAD = HEAD_DIM / THREADS_PER_HEAD;  // Div by 0 if HEAD_DIM=0
```

**Consequences:**
- Kernel launch failure with misaligned configuration
- Fallback to slow reference implementation without warning
- Silent accuracy degradation if padding is done incorrectly

**Prevention:**
```cpp
// CORRECT: Query and adapt to alignment requirements
struct FlashAttentionConfig {
    static constexpr int MIN_HEAD_DIM = 64;   // FlashAttention minimum
    static constexpr int ALIGNMENT = 8;       // Must be multiple of 8
    static constexpr int MAX_HEAD_DIM = 256;  // Common maximum
    
    static int round_up_head_dim(int head_dim) {
        if (head_dim < MIN_HEAD_DIM) return MIN_HEAD_DIM;
        return ((head_dim + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }
    
    static bool is_supported(int head_dim) {
        return head_dim >= MIN_HEAD_DIM && 
               head_dim <= MAX_HEAD_DIM &&
               head_dim % ALIGNMENT == 0;
    }
};

// CORRECT: Pad heads if necessary
Tensor prepare_kv_heads(const Tensor& input, int num_kv_heads) {
    int padded_kv_heads = FlashAttentionConfig::round_up_head_dim(num_kv_heads);
    
    if (padded_kv_heads != num_kv_heads) {
        Tensor padded(input.shape());
        // Copy input to first num_kv_heads positions
        // Pad remaining with zeros
        pad_heads(padded, input, num_kv_heads, padded_kv_heads);
        return padded;
    }
    return input;
}

// CORRECT: Fallback for unsupported head dims
void attention_dispatch(const Tensor& q, const Tensor& k, const Tensor& v,
                        Tensor& output) {
    int head_dim = q.shape(-1);
    
    if (FlashAttentionConfig::is_supported(head_dim)) {
        flash_attention_forward(q, k, v, output);
    } else {
        // Fall back to memory-efficient attention or paddle to supported size
        LOG(WARNING) << "Head dim " << head_dim << " not supported, using fallback";
        naive_attention_forward(q, k, v, output);
    }
}
```

---

### 3.2 FlashAttention Workspace Allocation Issues

**What goes wrong:** Workspace buffer is too small, not properly synchronized, or has incorrect lifetime.

**Why it happens:** FlashAttention requires workspace for intermediate computations:

```cpp
// WRONG: Fixed workspace size for all configurations
float* workspace;
cudaMalloc(&workspace, 1024 * 1024);  // 1MB - might not be enough!

// FlashAttention requires workspace proportional to:
// - batch_size * num_heads * seq_len * sizeof(float)
// For large sequences, 1MB is insufficient

// WRONG: Checking workspace size once at initialization
void init_attention() {
    // Query workspace once
    size_t workspace_bytes;
    flash_attn_v2_get_workspace_size(...);  // Called only at startup
    cudaMalloc(&workspace_, workspace_bytes);
    
    // But user changes batch_size at runtime!
    // workspace_bytes is now wrong
}

// WRONG: Using workspace after it's been freed
class AttentionRunner {
    Buffer workspace_;  // RAII buffer
    
    void run(const Config& cfg) {
        // Run attention
        flash_attention(q, k, v, output, workspace_.data(), workspace_.size());
        
        // Free workspace for reuse
        workspace_.reset();
        
        // Later: try to use workspace for another operation
        // BUG: workspace_.data() now points to freed memory
        other_operation(workspace_.data());  // Crash!
    }
};
```

**Consequences:**
- CUDA out-of-memory errors
- Incorrect results from buffer overflow
- Crashes from use-after-free

**Prevention:**
```cpp
// CORRECT: Dynamic workspace allocation per configuration
class FlashAttentionWorkspace {
    Buffer buffer_;
    
    void ensure_size(const FlashAttentionParams& params) {
        size_t required = flash_attn_v2_get_workspace_size(
            params.batch_size, params.num_heads, params.seq_len, params.head_dim,
            params.causal, params.dropout_prob);
        
        if (buffer_.size() < required) {
            buffer_ = Buffer(required);  // Reallocate if needed
        }
    }
    
    void* data() { return buffer_.data(); }
    size_t size() { return buffer_.size(); }
};

// CORRECT: Proper workspace lifecycle management
class AttentionSession {
    FlashAttentionWorkspace workspace_;
    
    void run_batch(const Batch& batch) {
        // Ensure workspace is sized for this batch
        workspace_.ensure_size(batch.to_params());
        
        // Workspace lives for entire session, not per-call
        flash_attention(q, k, v, output, workspace_.data(), workspace_.size());
        
        // Workspace persists - safe for subsequent operations
    }
    
    ~AttentionSession() {
        // Workspace freed here, after all operations complete
    }
};

// CORRECT: Verify workspace allocation succeeded
void run_with_workspace_check(...) {
    size_t required = get_workspace_size(params);
    
    if (available_memory() < required) {
        // Handle OOM gracefully
        throw OutOfMemoryError(required, available_memory());
    }
    
    Buffer workspace(required);
    NOVA_CHECK(flash_attention(..., workspace.data(), workspace.size()));
}
```

**Phase Recommendation:** Phase 1 (FlashAttention Integration) — Implement dynamic workspace sizing.

---

## 4. Paged Attention Pitfalls

### 4.1 Block Table CPU-GPU Consistency

**What goes wrong:** Block table modifications on CPU are not visible to GPU kernel, or synchronization is incorrect.

**Why it happens:** CPU and GPU have separate memory views. Block table updates require explicit synchronization:

```cpp
// WRONG: Modifying block table on CPU, then launching kernel immediately
void update_and_attend(int seq_id, const std::vector<int>& new_blocks) {
    // CPU writes new block indices
    for (int i = 0; i < new_blocks.size(); i++) {
        block_table_cpu_[seq_id * max_blocks + i] = new_blocks[i];
    }
    
    // WRONG: Launch kernel without synchronization
    // CPU write may not be visible to GPU yet!
    paged_attention_kernel<<<...>>>(q, block_table_gpu_, seq_id, ...);
    
    // Results are undefined - might use stale block indices
}

// WRONG: Wrong synchronization primitive
void wrong_sync(int seq_id) {
    update_block_table(seq_id);
    
    // WRONG: cudaStreamSynchronize on default stream
    // But kernel launched on different stream!
    cudaStreamSynchronize(stream_);  // Wrong stream!
    
    launch_kernel<<<..., stream_>>>(...);
}
```

**Consequences:**
- Attention reads from wrong KV cache blocks
- Silent correctness corruption
- Non-deterministic behavior depending on timing

**Prevention:**
```cpp
// CORRECT: Proper CPU-GPU synchronization for block table updates
class PagedAttentionManager {
    // CPU mirror of block table
    std::vector<int32_t> block_table_cpu_;
    
    // GPU block table
    Buffer block_table_gpu_;
    
    // Flag to track pending updates
    std::atomic<bool> update_pending_{false};
    
    void update_block_table(int seq_id, const std::vector<int>& blocks) {
        // Update CPU copy
        int offset = seq_id * max_blocks_;
        for (int i = 0; i < blocks.size(); i++) {
            block_table_cpu_[offset + i] = blocks[i];
        }
        
        // Copy to GPU synchronously
        int copy_size = blocks.size() * sizeof(int32_t);
        cudaMemcpyAsync(
            block_table_gpu_.data() + offset,
            block_table_cpu_.data() + offset,
            copy_size,
            cudaMemcpyHostToDevice,
            sync_stream_  // Dedicated synchronization stream
        );
        
        // Wait for copy to complete before kernel launch
        NOVA_CHECK_WITH_STREAM(cudaStreamSynchronize(sync_stream_), sync_stream_);
        
        update_pending_.store(false);
    }
    
    void attention(int seq_id, const Tensor& q) {
        // Ensure block table is synchronized
        assert(!update_pending_.load());
        
        // Now safe to launch attention kernel
        paged_attention_kernel<<<..., attention_stream_>>>(
            q.data(), block_table_gpu_.data(), seq_id, ...);
    }
};

// CORRECT: Alternative using CUDA events for fine-grained sync
void update_with_events(int seq_id) {
    // Record event after CPU update
    cudaEvent_t update_event;
    cudaEventCreate(&update_event);
    
    // CPU writes complete at this point
    cudaEventRecord(update_event, sync_stream_);
    
    // Make kernel wait for the event
    cudaStreamWaitEvent(attention_stream_, update_event, 0);
    
    // Launch kernel - guaranteed to see updated block table
    paged_attention_kernel<<<..., attention_stream_>>>(...);
    
    cudaEventDestroy(update_event);
}
```

---

### 4.2 Block Table Out-of-Bounds Access

**What goes wrong:** Kernel accesses block_table beyond allocated bounds when sequence uses maximum possible blocks.

**Why it happens:** Block table indexing doesn't validate sequence length against max_blocks:

```cpp
// WRONG: No bounds checking on block table access
__global__ void paged_attention_kernel(
    float* q,           // Query
    int32_t* block_table,  // Block indices
    int max_blocks,     // Maximum blocks per sequence
    int seq_len,        // Actual sequence length
    ...) {
    
    int block_idx = blockIdx.x;  // Which block we're computing
    int token_idx = block_idx * BLOCK_SIZE;  // First token in block
    int block_offset = token_idx % BLOCK_SIZE;
    
    // WRONG: Accessing block_table[block_idx] without checking block_idx < max_blocks
    int kv_block_idx = block_table[block_idx];  // Could be out of bounds!
    
    // If seq_len == max_blocks * BLOCK_SIZE exactly, block_idx could equal max_blocks
}
```

**Consequences:**
- CUDA illegal memory access
- Crashes or silent data corruption
- Non-deterministic failures

**Prevention:**
```cpp
// CORRECT: Explicit bounds validation
__global__ void paged_attention_safe(
    float* q, int32_t* block_table,
    int max_blocks, int seq_len,
    int max_seq_len,  // Absolute maximum sequence length
    ...) {
    
    int block_idx = blockIdx.x;
    int token_idx = block_idx * BLOCK_SIZE;
    
    // Guard against out-of-bounds
    if (token_idx >= seq_len) {
        return;  // This block doesn't need computation
    }
    
    // Calculate which block table entry we need
    int table_entry = token_idx / BLOCK_SIZE;
    
    // Validate table entry is within bounds
    assert(table_entry < max_blocks && "Block table access out of bounds");
    
    // Now safe to access
    int kv_block_idx = block_table[table_entry];
    assert(kv_block_idx < num_kv_blocks && "KV block index out of bounds");
}

// CORRECT: Host-side validation before kernel launch
void launch_paged_attention(...) {
    int num_blocks_needed = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Validate sequence fits in block table
    if (num_blocks_needed > max_blocks) {
        throw std::runtime_error(
            "Sequence too long: " + std::to_string(seq_len) + 
            " requires " + std::to_string(num_blocks_needed) + 
            " blocks, but max is " + std::to_string(max_blocks));
    }
    
    // Validate all block indices are valid
    for (int i = 0; i < num_blocks_needed; i++) {
        int block_idx = block_table[seq_id * max_blocks + i];
        if (block_idx < 0 || block_idx >= num_kv_blocks) {
            throw std::runtime_error("Invalid block index: " + std::to_string(block_idx));
        }
    }
    
    paged_attention_kernel<<<...>>>(...);
}
```

---

## 5. Sequence Length Handling Pitfalls

### 5.1 Variable Sequence Length in Batched Attention

**What goes wrong:** Batched attention processes variable-length sequences incorrectly, causing cross-contamination or incorrect masking.

**Why it happens:** Sequences with different lengths share batched operations without proper offset handling:

```cpp
// WRONG: Assuming all sequences have same length
__global__ void batch_attention_kernel(float* output, float* input, int batch_size, int seq_len) {
    int batch_idx = blockIdx.x / seq_len;  // WRONG if sequences differ in length
    int seq_idx = blockIdx.x % seq_len;    // Misleading calculation
    
    // This approach breaks when:
    // Sequence 0: length 512
    // Sequence 1: length 128
    // Both are padded to seq_len=512, but sequence 1's "padding" is garbage
}

// WRONG: Not handling ragged tensors properly
void ragged_attention(float* q, int* cu_seqlens, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        int start = cu_seqlens[b];
        int end = cu_seqlens[b + 1];
        
        // WRONG: Processing entire [start, end) range
        // But kernel has no awareness of start offset for shared memory
        attention_kernel<<<blocks, threads>>>(q + b * max_seq_len * head_dim, ...);
        // Uses q[b * max_seq_len * head_dim ..], not q[start * head_dim ..]
    }
}
```

**Consequences:**
- Attention reads from wrong positions
- Cross-sequence contamination
- Incorrect outputs for variable-length batches

**Prevention:**
```cpp
// CORRECT: Use cumulative sequence lengths for offset calculation
struct VariableSeqLengths {
    int* cu_seqlens_q;  // Shape: [batch_size + 1], cumulative lengths
    int* cu_seqlens_k;  // May differ for cross-attention
    int max_seq_len;
    
    int num_tokens(int batch_idx) const {
        return cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
    }
    
    int token_offset(int batch_idx) const {
        return cu_seqlens_q[batch_idx];
    }
};

__global__ void packed_attention_kernel(
    float* output, float* q, int32_t* cu_seqlens, int total_tokens, ...) {
    
    int token_idx = blockIdx.x * BLOCK_TOKENS + threadIdx.x;
    if (token_idx >= total_tokens) return;
    
    // Find which sequence this token belongs to
    int seq_idx = find_sequence(token_idx, cu_seqlens);
    
    // Local position within sequence (for causal masking)
    int local_pos = token_idx - cu_seqlens[seq_idx];
    
    // Now compute attention with correct offsets
    compute_attention(output, q, token_idx, local_pos, ...);
}

// CORRECT: Separate kernel launches per sequence when necessary
void attention_variable_seqlen(float* output, float* q, 
                               const VariableSeqLengths& seq_lens) {
    int offset = 0;
    for (int b = 0; b < seq_lens.batch_size; b++) {
        int seq_len = seq_lens.num_tokens(b);
        
        // Launch kernel for this specific sequence
        dim3 blocks((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
        attention_kernel<<<blocks, threads>>>(
            output + offset, q + offset, seq_len, ...);
        
        offset += seq_len * head_dim;
        NOVA_CHECK(cudaGetLastError());
    }
}
```

---

### 5.2 Position Embedding Mismatch with Sequence Length

**What goes wrong:** Position embeddings (RoPE, Alibi) produce incorrect values for sequences longer than trained on or with unusual lengths.

**Why it happens:** Position encoding tables or precomputed frequencies don't scale correctly:

```cpp
// WRONG: Fixed-size position embedding table
float position_embeddings[4096 * head_dim];  // Only works up to seq_len=4096

void attention_with_pos_emb(float* q, int seq_len) {
    for (int pos = 0; pos < seq_len; pos++) {
        if (pos >= 4096) {
            // WRONG: Truncating or accessing out of bounds
            break;  // Or: pos = 4095;  // Wrong position encoding!
        }
        apply_position_embedding(q + pos * head_dim, position_embeddings[pos]);
    }
}

// WRONG: Precomputed frequencies don't scale
float freqs_cis[seq_len * head_dim / 2];  // Recomputing requires knowing max_seq_len
```

**Consequences:**
- Incorrect attention patterns for long sequences
- Degraded model quality
- Silent failures that are hard to detect

**Prevention:**
```cpp
// CORLECT: Dynamic position embedding computation
void apply_rope(float* q, int seq_len, int pos, int head_dim, float base = 10000.0f) {
    // Compute RoPE frequencies on-the-fly
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = base * exp2(-2.0f * i / head_dim);
        float theta = pos / freq;
        
        float cos_val = cos(theta);
        float sin_val = sin(theta);
        
        int dim = i * 2;
        float q0 = q[dim], q1 = q[dim + 1];
        
        // Rotate
        q[dim] = q0 * cos_val - q1 * sin_val;
        q[dim + 1] = q0 * sin_val + q1 * cos_val;
    }
}

// CORRECT: Extend RoPE with position scaling for fine-tuned models
void apply_extended_rope(float* q, int seq_len, int pos, int rope_cfg_len,
                         float rope_scaling = 1.0f) {
    int effective_pos = (pos < rope_cfg_len) ? 
                        pos : 
                        rope_cfg_len + (pos - rope_cfg_len) * rope_scaling;
    
    apply_rope(q, seq_len, effective_pos, head_dim);
}
```

---

## 6. Tensor/Pipeline Parallelism Pitfalls

### 6.1 Attention Backward Pass Communication Deadlock

**What goes wrong:** Backward pass in tensor parallelism hangs due to incorrect NCCL operation ordering or collective communication errors.

**Why it happens:** Attention backward requires all-reduce after local computation:

```cpp
// WRONG: Missing all-reduce in backward pass
__global__ void attention_backward_kernel(float* grad_q, float* grad_k, float* grad_v,
                                           float* grad_out, ...) {
    // Compute local gradients
    compute_local_gradients(grad_q, grad_k, grad_v, grad_out, ...);
    
    // WRONG: Not synchronizing across ranks!
    // Each rank has partial gradient, not the full gradient
}

// WRONG: All-reduce in wrong position
void attention_backward(float* grad_q, ...) {
    // Compute gradients w.r.t. Q, K, V
    compute_dQ_dK_dV(grad_q, grad_k, grad_v, ...);
    
    // WRONG: Reducing Q gradient here
    ncclAllReduce(grad_q, grad_q, NCCL_SUM);  // Q should use local gradients only
    
    // More local computation...
    compute_dS(grad_q, ...);
    
    // BUG: K and V gradients never reduced!
}

// WRONG: Deadlock from mismatched collective calls
void parallel_backward(float* grad_q, float* grad_k, float* grad_v) {
    // Rank 0:
    ncclAllReduce(grad_k, grad_k, NCCL_SUM);  // K first
    
    // Rank 1:
    ncclAllReduce(grad_v, grad_v, NCCL_SUM);  // V first
    
    // DEADLOCK: Both waiting for different operations
}
```

**Consequences:**
- Collective operations deadlocking
- Incorrect gradients (unreduced)
- Hanging training jobs

**Prevention:**
```cpp
// CORRECT: Proper gradient synchronization
class ParallelAttention {
    ncclComm_t comm_;
    int rank_;
    int world_size_;
    
    void backward(float* grad_q, float* grad_k, float* grad_v,
                  float* grad_out, ...) {
        // Step 1: Compute local gradients
        compute_local_dQ_dK_dV(grad_q, grad_k, grad_v, grad_out, ...);
        
        // Step 2: Synchronize K and V gradients (shared across ranks)
        // Each rank has correct gradients for its K/V partition
        // But we need full gradient for all-reduce with other activations
        ncclAllReduce(grad_k, grad_k, num_kv_heads * head_dim, NCCL_SUM, comm_);
        ncclAllReduce(grad_v, grad_v, num_kv_heads * head_dim, NCCL_SUM, comm_);
        
        // Step 3: Propagate gradients through attention scores
        compute_dS(grad_q, grad_k, grad_v, ...);
        
        // Step 4: Q gradient needs all-reduce
        ncclAllReduce(grad_q, grad_q, num_q_heads * head_dim, NCCL_SUM, comm_);
    }
};

// CORRECT: Use deterministic collective operation order
void synchronized_backward(float* grad_q, float* grad_k, float* grad_v) {
    // All ranks MUST use same operation order
    // Order: dK -> dV -> dQ (or any consistent order)
    
    cudaStreamSynchronize(stream_);  // Ensure all local computation done
    
    // All ranks call in same order
    ncclAllReduce(grad_k, grad_k, ..., comm_, stream_);
    ncclAllReduce(grad_v, grad_v, ..., comm_, stream_);
    ncclAllReduce(grad_q, grad_q, ..., comm_, stream_);
    
    cudaStreamSynchronize(stream_);  // Wait for all operations
}

// CORRECT: Use NCCL's built-in grouping
void grouped_backward(float* grad_q, float* grad_k, float* grad_v) {
    // Group related operations for efficiency
    ncclGroupStart();
    ncclAllReduce(grad_k, grad_k, ..., comm_, stream_);
    ncclAllReduce(grad_v, grad_v, ..., comm_, stream_);
    ncclGroupEnd();
    
    // Local computation
    
    ncclGroupStart();
    ncclAllReduce(grad_q, grad_q, ..., comm_, stream_);
    ncclGroupEnd();
}
```

---

### 6.2 Pipeline Parallelism PP Stage Boundary Errors

**What goes wrong:** KV cache communication between pipeline stages is incorrect, causing gradient misalignment or missing updates.

**Why it happens:** Pipelined forward/backward creates micro-batches that must synchronize at stage boundaries:

```cpp
// WRONG: Not handling PP schedule correctly
class PipelineStage {
    void forward_microbatch(Microbatch& mb) {
        // Compute with local layers
        compute_local(mb);
        
        // WRONG: Sending KV cache immediately, but backward needs it too
        send_kv_cache(mb.k_cache, mb.v_cache, next_stage_);
        
        // Backward will fail - KV cache already sent!
    }
    
    void backward_microbatch(Microbatch& mb) {
        // Need KV cache for backward, but it's already been sent!
        compute_backward(mb);  // BUG: missing KV cache
    }
};

// WRONG: Overwriting KV cache before backward completes
void pp_forward(Microbatch& mb) {
    // Receive KV from previous stage
    if (has_prev_stage) {
        recv_kv_cache(mb.k_cache, prev_stage_);
    }
    
    // Store for later backward
    kv_store_[mb.creation_time] = mb.kv_cache;  // Save reference
    
    // Compute attention
    attention(mb);
    
    // Send to next stage
    send_kv_cache(mb.k_cache, next_stage_);
    
    // WRONG: Overwriting storage before backward completes
    // If backward uses async communication, this could be race condition
    mb.k_cache = {};  // Clearing too early!
}
```

**Consequences:**
- Backward pass uses stale or missing KV cache
- Gradient desynchronization
- Training divergence

**Prevention:**
```cpp
// CORRECT: PP-aware KV cache management
class PipelineStage {
    struct PPContext {
        std::vector<float> k_cache;
        std::vector<float> v_cache;
        int microbatch_id;
        bool backward_complete;
    };
    
    std::vector<PPContext> kv_contexts_;
    int forward_microbatch_count_ = 0;
    
    void forward_microbatch(Microbatch& mb) {
        // Receive KV if first stage
        if (has_prev_stage_) {
            recv_kv_cache(mb.k_cache, prev_stage_);
        }
        
        // Store context for backward (with ref counting)
        int ctx_id = forward_microbatch_count_++;
        kv_contexts_.push_back({
            .k_cache = mb.k_cache.clone(),
            .v_cache = mb.v_cache.clone(),
            .microbatch_id = mb.id,
            .backward_complete = false,
        });
        
        // Send after storing
        if (has_next_stage_) {
            send_kv_cache(mb.k_cache, next_stage_);
        }
        
        // Local computation
        compute_attention(mb);
    }
    
    void backward_microbatch(int microbatch_id) {
        // Find matching context
        auto it = std::find_if(kv_contexts_.begin(), kv_contexts_.end(),
            [microbatch_id](const PPContext& ctx) { 
                return ctx.microbatch_id == microbatch_id; 
            });
        
        // Wait for context to be available
        while (it == kv_contexts_.end() || !it->backward_complete) {
            // Wait for context to be created and backward to be ready
            std::this_thread::yield();
        }
        
        // Use stored KV cache for backward
        compute_backward_with_kv(microbatch_id, it->k_cache, it->v_cache);
        
        // Mark complete - now safe to reuse
        it->backward_complete = true;
        cleanup_if_safe(it);
    }
    
    void cleanup_if_safe(std::vector<PPContext>::iterator it) {
        // Clean up if backward is done and no more need
        if (it->backward_complete && !has_prev_stage_) {
            // Safe to free
            kv_contexts_.erase(it);
        }
    }
};
```

**Phase Recommendation:** Phase 4 (Parallelism) — Implement PP-aware KV cache lifecycle management.

---

## Summary: Phase Recommendations

| Phase Topic | Likely Pitfall | Mitigation Strategy |
|-------------|----------------|---------------------|
| FlashAttention Integration | Softmax overflow | Stable softmax from day 1 |
| FlashAttention Integration | Head dim alignment | Query requirements, pad if needed |
| FlashAttention Integration | Workspace sizing | Dynamic allocation per config |
| KV Cache | Memory fragmentation | Block-based fixed-size allocation |
| KV Cache | Stale references | Version tracking on blocks |
| Paged Attention | CPU-GPU sync | Dedicated sync stream with synchronization |
| Paged Attention | Out-of-bounds access | Bounds validation before kernel launch |
| Sequence Handling | Cross-batch contamination | Cumulative length arrays |
| Sequence Handling | Position embedding limits | Dynamic computation or extrapolation |
| Parallelism | Gradient deadlocks | Deterministic collective ordering |
| Parallelism | PP KV lifecycle | Reference counting with cleanup |

---

## Sources

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) - **HIGH confidence** (official source, v3.1 release)
- [FlashAttention GitHub Repository](https://github.com/Dao-AILab/flash-attention) - **HIGH confidence** (reference implementation)
- [vLLM Paged Attention Paper](https://arxiv.org/abs/2309.06180) - **HIGH confidence** (official vLLM source)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm) - **HIGH confidence** (production implementation)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) - **HIGH confidence** (NVIDIA official)
- [RoPE Position Encoding](https://arxiv.org/abs/2104.09864) - **HIGH confidence** (original paper)
- [Megatron-LM Tensor Parallelism](https://arxiv.org/abs/1909.08053) - **HIGH confidence** (NVIDIA deep learning)
- [GPipe Pipeline Parallelism](https://arxiv.org/abs/1811.06965) - **HIGH confidence** (Google Brain)

---

*Last updated: 2026-04-29 for Transformer & Inference Optimization milestone*
