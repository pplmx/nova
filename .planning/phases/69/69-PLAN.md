# Phase 69 Plan: FlashAttention Integration

## Goal

Implement FlashAttention-2 kernel integration with attention backend selection and stable softmax.

## Requirements

- FA-01: User can select attention backend via enum
- FA-02: User can compute attention forward pass with IO-aware tiling
- FA-03: FlashAttention supports BF16 and FP16 with stable softmax
- FA-04: User can compute attention backward pass for training

## Implementation

### 1. Create Attention Backend Enum

**File:** `include/cuda/neural/transformer/attention.h`

Add enum before MultiHeadAttentionConfig:
```cpp
enum class AttentionBackend {
    Standard,
    FlashAttention,
    PagedAttention
};
```

### 2. Create FlashAttention Class

**File:** `include/cuda/algo/flash_attention.h` (new)

```cpp
namespace cuda::algo {

enum class FlashAttentionVersion {
    V2,
    V3  // For Hopper
};

struct FlashAttentionConfig {
    int num_heads = 8;
    int head_dim = 64;
    int seq_len = 512;
    int batch_size = 1;
    float dropout_rate = 0.0f;
    bool causal = true;
    bool is_fp16 = true;
    int num_splits = 1;
    FlashAttentionVersion version = FlashAttentionVersion::V2;
};

class FlashAttention {
public:
    explicit FlashAttention(const FlashAttentionConfig& config);
    ~FlashAttention();
    
    void forward(
        const memory::Buffer& output,
        const memory::Buffer& query,
        const memory::Buffer& key,
        const memory::Buffer& value,
        memory::Buffer& softmax_lse,
        const stream::Stream& stream
    );
    
    void backward(
        memory::Buffer& dq,
        memory::Buffer& dk,
        memory::Buffer& dv,
        const memory::Buffer& output,
        const memory::Buffer& dout,
        const memory::Buffer& query,
        const memory::Buffer& key,
        const memory::Buffer& value,
        const memory::Buffer& softmax_lse,
        const stream::Stream& stream
    );
    
    void set_dropout(float rate);
    size_t get_workspace_size() const;
    void ensure_workspace(size_t bytes);

private:
    FlashAttentionConfig config_;
    memory::Buffer<void> workspace_;
    uint64_t dropout_seed_ = 0;
};

std::unique_ptr<FlashAttention> create_flash_attention(
    const FlashAttentionConfig& config
);

}  // namespace cuda::algo
```

### 3. Implement Stable Softmax Kernel

**File:** `src/algo/flash_attention.cu` (new)

- Online softmax computation with max subtraction
- Warp-level reduction for efficiency
- Support for causal masking
- Numerical stability for large sequence lengths

### 4. Implement FlashAttention Forward Kernel

- Tile-based computation (64x64 or 128x64 blocks)
- SRAM usage for intermediate results
- HBM reads reduced from O(N²d) to O(Nd)
- Workspace allocation per configuration

### 5. Implement FlashAttention Backward Kernel

- Gradient computation for Q, K, V
- Deterministic dropout with seed propagation
- dP → dQ, dK, dV computation

### 6. Add Tests

**File:** `tests/algo/flash_attention_test.cu`

- Numerical accuracy vs standard attention
- BF16/FP16 correctness
- Stable softmax overflow prevention
- Backward pass gradient correctness
- GQA/MQA support

### 7. Update TransformerLayer

**File:** `include/cuda/neural/transformer/layer.h` (new or extend)

- Add set_attention_backend() method
- Backend switching without interface change
- FlashAttention as drop-in replacement

## Files to Create

1. `include/cuda/algo/flash_attention.h`
2. `src/algo/flash_attention.cu`
3. `tests/algo/flash_attention_test.cu`
4. Update `include/cuda/neural/transformer/attention.h`

## Files to Modify

1. `include/cuda/neural/transformer/attention.h` - add enum
2. `CMakeLists.txt` - add new source files

## Success Criteria

1. User can select attention backend via AttentionBackend enum
2. FlashAttention forward matches standard attention within 1e-3 relative error
3. Stable softmax prevents numerical overflow for large sequences
4. Backward pass computes correct gradients
5. Workspace allocation is dynamic based on query shape
