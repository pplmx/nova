# Phase 71 Plan: Paged Attention Integration

## Goal

Implement paged attention combining FlashAttention with block-based KV cache management.

## Requirements

- PA-01: BlockManager with block table mapping
- PA-02: Token append with block allocation
- PA-03: CPU-GPU block table synchronization
- PA-04: Paged attention forward with bounds validation

## Implementation

### 1. Create BlockManager Header

**File:** `include/cuda/inference/block_manager.h`

```cpp
namespace cuda::inference {

struct Sequence {
    int64_t id;
    int64_t created_at;
    int num_tokens;
    int max_tokens;
    std::vector<int> block_table;
    memory::Buffer<float> k_cache_view;
    memory::Buffer<float> v_cache_view;
};

struct BlockManagerConfig {
    int max_model_len = 8192;
    int block_size = 16;
    int num_cpu_blocks = 2048;
    int num_gpu_blocks = 4096;
    bool enable_cuda_graph = true;
    memory::KVCacheAllocatorConfig kv_cache_config;
    algo::FlashAttentionConfig attention_config;
};

class BlockManager {
public:
    explicit BlockManager(const BlockManagerConfig& config);
    ~BlockManager();

    Sequence* create_sequence(int64_t sequence_id, int max_tokens);
    void append_tokens(int64_t sequence_id, int num_tokens);
    Sequence* get_sequence(int64_t sequence_id);
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

private:
    void allocate_blocks_for_sequence(Sequence* seq, int num_blocks);
    void validate_block_index(int block_idx) const;

    BlockManagerConfig config_;
    std::unordered_map<int64_t, std::unique_ptr<Sequence>> sequences_;
    std::shared_mutex sequence_mutex_;

    std::unique_ptr<memory::KVCacheAllocator> kv_cache_;
    std::unique_ptr<algo::FlashAttention> attention_;

    std::vector<int> free_blocks_;
    memory::Buffer<int> block_table_gpu_;
    stream::Stream sync_stream_;
    int num_allocated_blocks_ = 0;
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
```

### 2. Implement BlockManager

**File:** `src/cuda/inference/block_manager.cpp`

- create_sequence(): Allocate initial blocks, create block table
- append_tokens(): Call KVCacheAllocator::append, update block table
- free_sequence(): Call KVCacheAllocator::free, remove Sequence
- forward_batch(): Run FlashAttention with block table access
- sync_block_tables(): Copy block table to GPU, sync stream

### 3. Implement PagedAttention Kernel

- Read block table to get physical block indices
- Gather KV cache from non-contiguous blocks
- Run FlashAttention on gathered data
- Scatter output back

### 4. Create inference Directory Structure

```
include/cuda/inference/
├── block_manager.h
└── types.h
src/cuda/inference/
├── block_manager.cpp
tests/inference/
└── block_manager_test.cpp
```

### 5. Create Tests

- create_sequence returns valid block table
- append_tokens allocates new blocks
- free_sequence returns blocks to allocator
- forward_batch processes multiple sequences
- Block table sync synchronization
- Out-of-bounds validation

## Files to Create

1. `include/cuda/inference/block_manager.h`
2. `include/cuda/inference/types.h`
3. `src/cuda/inference/block_manager.cpp`
4. `tests/inference/block_manager_test.cpp`

## Success Criteria

1. BlockManager.create_sequence returns valid block table
2. append_tokens allocates additional physical blocks
3. cudaStreamSynchronize called before attention kernel
4. Paged attention output matches contiguous attention
5. Out-of-bounds block table access returns error
