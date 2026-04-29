# Stack Research

**Domain:** CUDA GPU Inference Optimization
**Researched:** 2026-04-29
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| NVIDIA Transformer Engine | 2.14+ | Attention primitives, fused layers, FP8/FP4 quantization | Official NVIDIA library with C/C++ API, native paged KV cache support, CUDA Graphs integration |
| FlashAttention | 2.5.x (FA2) / 3.x (FA3) | IO-aware exact attention algorithm | 2-4x faster than standard attention, O(n) memory instead of O(n^2), supports MQA/GQA |
| cuDNN | 9.x | Low-level CUDA primitives | Hardware-accelerated attention via `cudnnAttentionForward`, fused attention kernels |
| CUTLASS | 3.x | CUDA Templates for Linear Algebra | Reference GEMM implementations for custom fused kernels, FP8 support |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| FlashAttention (hopper) | 3.x beta | Optimized for H100/H800 | When targeting Hopper architectures, need FP8 forward |
| FlashAttention-4 (CuTeDSL) | 4.x | Next-gen kernels for Hopper/Blackwell | When targeting B200 or newer, maximum performance |
| NCCL | 2.21+ | GPU interconnects | Sequence parallelism, tensor pipeline communication |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| NVCC | CUDA compiler (CUDA 20+) | Required for device code compilation |
| NVTX | NVIDIA Tools Extension | Existing nova integration (v2.4) - continue using for profiling |
| CUDA Graphs | Kernel launch optimization | Existing nova integration (v2.4) - essential for paged attention |
| Nsight Compute | Kernel profiling | Verify attention kernel efficiency |

## Key Architectural Decisions

### 1. Attention Backend Selection

**For pure C++ (no PyTorch dependency):**

| Architecture | Recommended Backend | Justification |
|--------------|---------------------|---------------|
| Ampere (A100) | TransformerEngine C API | Stable, full feature set, paged attention support |
| Ada (RTX 4090) | TransformerEngine C API | Same as Ampere, hardware supports all TE features |
| Hopper (H100) | TE + FA3 kernel reference | TE uses FA3 internally, consider direct FA3 for max control |

**Why TransformerEngine over raw FlashAttention:**
- Native C/C++ API (no PyTorch dependency)
- Built-in paged KV cache management
- CUDA Graphs support via `make_graphed_callables`
- FP8/FP4 quantization with calibrated scaling factors
- Context parallelism for sequence parallelism

### 2. KV Cache Strategy

| Approach | Pros | Cons |
|----------|------|------|
| Vanilla contiguous KV cache | Simple | Memory fragmentation, max seq length fixed |
| **Paged KV cache (recommended)** | Memory efficient, variable length, batching | More complex management |

TransformerEngine 2.x provides paged attention via `DotProductAttention` with block table support - aligns with vLLM's architecture.

### 3. Precision Selection

| Precision | Use Case | Notes |
|-----------|----------|-------|
| BF16 | Default | Best accuracy/performance balance |
| FP8 (E4M3/E5M2) | Large batch inference | Requires calibration, 2x memory bandwidth improvement |
| FP8 with block scaling | Quantized weights | Better accuracy for inference-only |
| FP4 | Extreme compression | Experimental, NVFP4 only on Hopper+ |

## Installation

```bash
# TransformerEngine (C++ only, requires CUDA 12+)
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
mkdir build && cd build
cmake .. -DBUILD_C=ON -DBUILD_PYTHON=OFF -DDEV=ON
make -j$(nproc)
make install

# FlashAttention (for kernel reference / PyTorch integration)
pip install flash-attn --no-build-isolation

# cuDNN (should be in CUDA container)
# Typically included in: nvidia/cuda:12.x-cudnn8-runtime
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| TransformerEngine | Raw cuDNN attention | Need minimal dependencies, only basic attention |
| TransformerEngine | Triton kernels | Custom attention patterns, AMD ROCm support |
| TE C API | FlashAttention Python | PyTorch-based deployment, faster iteration |

### Why NOT Raw cuDNN Attention Alone

- cuDNN `cudnnAttentionForward` is a low-level primitive, not a complete solution
- No paged KV cache support
- No FP8/quantization integration
- Requires significant glue code

### Why NOT Triton Alone

- Triton is a higher-level language but adds compilation overhead
- Less control over low-level CUDA details for performance-critical paths
- TE uses optimized CUTLASS kernels with better utilization

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| FlashAttention v1.x | Outdated, no longer maintained | FlashAttention-2+ |
| PyTorch-only attention backends | Incompatible with pure C++ | TransformerEngine C API |
| Standard O(n^2) attention | Memory quadratic in sequence length | FlashAttention or TE DotProductAttention |
| Custom GEMM without CUTLASS | reinventing wheel | CUTLASS templates or TE fused layers |

## Stack Patterns by Architecture

**If targeting Ampere (A100) or Ada (RTX 4090):**
- Use TransformerEngine 2.14+ with C API
- BF16 for inference, FP8 for large batch
- Paged KV cache via `DotProductAttention`
- CUDA Graphs for iterative decoding

**If targeting Hopper (H100/H800):**
- Use TransformerEngine with FA3 backend internally
- Consider FP8 block scaling for weight quantization
- FlashAttention-3 beta for maximum control (via `hopper/` subdirectory)
- Leverage WGMMA instructions via TE

**If supporting multiple architectures:**
- Abstract attention backend behind interface
- Use compile-time CUDA architecture flags
- Consider FATBIN for precompiled kernels

## Version Compatibility

| Component | Compatible With | Notes |
|-----------|-----------------|-------|
| TransformerEngine 2.14 | CUDA 12.0+, cuDNN 8.9+ | Requires PyTorch or standalone C API |
| FlashAttention 2.5.x | CUDA 12.0+, PyTorch 2.0+ | Head dims up to 256 |
| FlashAttention 3 beta | CUDA 12.3+, H100/H800 only | FP16/BF16, FP8 forward |
| FlashAttention 4 | CUDA 13+, H100/B200 | CuTeDSL-based, best Blackwell support |
| cuDNN 9.x | CUDA 12.0+ | Attention primitives, fused attention |

## Integration with Existing nova Infrastructure

### Existing Components (v2.2, v1.3, v2.4)

| Component | New Integration Point |
|-----------|----------------------|
| `MultiHeadAttention` (v2.2) | Replace internals with TE `DotProductAttention` |
| `TensorParallelMatmul` (v1.3) | Works with TE `Linear.set_tensor_parallel_group()` |
| `PipelineParallelism` (v1.3) | Compatible with TE layer interfaces |
| `CUDA Graphs` (v2.4) | TE `make_graphed_callables()` for attention |
| `NVTX` (v2.4) | TE exposes `get_cudnn_version()` for diagnostics |

### Recommended Integration Approach

1. **Phase 1:** Replace attention internals with TE C API
2. **Phase 2:** Add paged KV cache management layer
3. **Phase 3:** Integrate sequence parallelism via TE context parallel groups
4. **Phase 4:** Add FP8/quantization support

## Sources

- [NVIDIA TransformerEngine 2.14 Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) — HIGH confidence
- [FlashAttention GitHub (Dao-AILab)](https://github.com/Dao-AILab/flash-attention) — HIGH confidence
- [FlashAttention-3 Blog](https://tridao.me/blog/2024/flash3/) — HIGH confidence
- [Transformer Engine Gemma Inference Tutorial](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_gemma/tutorial_generation_gemma_with_te.html) — HIGH confidence

---
*Stack research for: CUDA GPU Inference Optimization*
*Researched: 2026-04-29*
