# Phase 25: Distributed Batch Normalization - Context

**Phase:** 25
**Goal:** Implement synchronized batch normalization across GPUs
**Requirements:** DBN-01 to DBN-03

## Requirements Analysis

### DBN-01: SyncBatchNorm with all-reduce for mean/variance

Synchronized batch normalization requires:
1. Computing local batch statistics (mean, variance) on each GPU
2. All-reducing statistics across all GPUs
3. Synchronizing running mean/variance for inference

### DBN-02: Cross-GPU batch statistics aggregation

Key challenges:
- Batch size per GPU (total_batch = local_batch * num_gpus)
- Numerical stability (Welford's algorithm or two-pass)
- Async vs sync reduction timing

### DBN-03: Evaluation vs training mode handling

Modes:
- Training: Use batch statistics, update running stats
- Inference: Use running population statistics

## Design Decisions

### D-01: Sync Strategy

**Decision:** Synchronous all-reduce before forward pass completes

**Rationale:**
- Ensures deterministic results across runs
- Simpler correctness guarantees
- Performance acceptable for typical batch sizes

**Alternative considered:** Async overlap with computation
- Higher complexity, harder to debug
- Marginal performance gains for small batches

### D-02: Statistics Computation

**Decision:** Two-pass computation for numerical stability

**Rationale:**
- Standard batch norm uses E[x²] - E[x]² which can be unstable
- Two-pass (compute mean, then variance) is more stable
- Welford's algorithm is equivalent but more complex

### D-03: Running Statistics Update

**Decision:** Exponential moving average (EMA) with momentum

**Rationale:**
- Standard PyTorch behavior
- Configurable momentum parameter
- Works well for typical training scenarios

## Architecture

```
SyncBatchNorm
├── Forward Training
│   ├── compute_local_stats(input)
│   ├── all_reduce_mean(mean)
│   ├── all_reduce_variance(var)  [uses synced mean]
│   ├── normalize(input, synced_mean, synced_var)
│   └── update_running_stats(mean, var)
├── Forward Inference
│   └── normalize(input, running_mean, running_var)
└── Backward (gradient computation)
```

## Implementation Plan

1. **Files to create:**
   - `include/cuda/neural/sync_batch_norm.h`
   - `src/cuda/neural/sync_batch_norm.cu`

2. **Dependencies:**
   - Existing LayerNorm for reference
   - DistributedReduce for all-reduce
   - DeviceMesh for GPU enumeration

3. **Key functions:**
   - `sync_batch_norm_forward_training()`
   - `sync_batch_norm_forward_inference()`
   - `sync_batch_norm_backward()`

## Success Criteria

1. Mean and variance synchronized across GPUs via all-reduce
2. Training mode maintains running statistics correctly
3. Evaluation mode uses population statistics
4. Results match single-GPU batch norm when batch_size_per_gpu * num_gpus = total_batch

## Pitfalls to Avoid

1. **Lazy initialization:** Ensure running stats initialized on first forward pass
2. **Numerical instability:** Use two-pass variance computation
3. **Synchronization overhead:** Keep communication minimal
4. **Gradient synchronization:** Sync gradients in backward pass

## References

- PyTorch SyncBatchNorm implementation
- "Batch Normalization: Accelerating Deep Network Training" (Ioffe & Szegedy, 2015)
- NVIDIA BatchNorm best practices
