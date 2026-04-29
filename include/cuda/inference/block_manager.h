#pragma once

#include "cuda/algo/flash_attention.h"
#include "cuda/memory/kv_cache_allocator.h"
#include "cuda/stream/stream.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace cuda::inference {

struct Sequence {
    int64_t id;
    int64_t created_at;
    int num_tokens;
    int max_tokens;
    std::vector<int> block_table;

    Sequence(int64_t sequence_id, int max_toks)
        : id(sequence_id),
          created_at(0),
          num_tokens(0),
          max_tokens(max_toks) {}
};

struct BlockManagerConfig {
    int max_model_len = 8192;
    int block_size = 16;
    int num_cpu_blocks = 2048;
    int num_gpu_blocks = 4096;
    bool enable_cuda_graph = true;

    memory::KVCacheAllocatorConfig kv_cache_config{
        .num_heads = 32,
        .head_dim = 128,
        .block_size_tokens = 16,
        .num_blocks = 4096,
        .num_layers = 32,
        .eviction_threshold_pct = 10,
        .enable_prefix_caching = true
    };

    algo::FlashAttentionConfig attention_config{
        .num_heads = 32,
        .num_kv_heads = 32,
        .head_dim = 128,
        .seq_len = 512,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };
};

class BlockManager {
public:
    explicit BlockManager(const BlockManagerConfig& config);
    ~BlockManager();

    BlockManager(const BlockManager&) = delete;
    BlockManager& operator=(const BlockManager&) = delete;
    BlockManager(BlockManager&&) = default;
    BlockManager& operator=(BlockManager&&) = default;

    Sequence* create_sequence(int64_t sequence_id, int max_tokens);
    void append_tokens(int64_t sequence_id, int num_tokens);
    Sequence* get_sequence(int64_t sequence_id);
    const Sequence* get_sequence(int64_t sequence_id) const;
    void free_sequence(int64_t sequence_id);

    void forward_batch(
        const std::vector<int64_t>& sequence_ids,
        const memory::Buffer<float>& query,
        memory::Buffer<float>& output,
        const stream::Stream& stream
    );

    void sync_block_tables(const stream::Stream& stream);
    void maybe_evict();
    int get_num_free_blocks() const;
    memory::KVCacheAllocator* get_kv_cache() const { return kv_cache_.get(); }

private:
    void allocate_blocks_for_sequence(Sequence* seq, int num_blocks);
    void validate_block_index(int block_idx) const;
    void update_block_table_gpu(Sequence* seq, const stream::Stream& stream);

    BlockManagerConfig config_;
    std::unordered_map<int64_t, std::unique_ptr<Sequence>> sequences_;
    mutable std::shared_mutex sequence_mutex_;

    std::unique_ptr<memory::KVCacheAllocator> kv_cache_;
    std::unique_ptr<algo::FlashAttention> attention_;

    memory::Buffer<int> block_table_gpu_;
    int max_blocks_per_sequence_;
    int num_allocated_sequences_ = 0;
};

class PagedAttention {
public:
    static void forward(
        memory::Buffer<float>& output,
        const memory::Buffer<float>& query,
        const memory::Buffer<void>& key_cache,
        const memory::Buffer<void>& value_cache,
        const std::vector<int>& block_table,
        int num_tokens,
        int num_heads,
        int head_dim,
        int block_size,
        const stream::Stream& stream
    );
};

}  // namespace cuda::inference
