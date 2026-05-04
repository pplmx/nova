# Stack Research: Transformer Optimization Features

**Domain:** CUDA/HPC Transformer Inference
**Researched:** 2026-05-05
**Confidence:** MEDIUM-HIGH (verified against NVIDIA official docs, vLLM, TensorRT-LLM, FlashInfer)

## Recommended Stack

### Core CUDA Infrastructure

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| CUDA | 12.x+ | GPU compute platform | Required for Hopper/Blackwell features. CUDA 12.8+ needed for Blackwell support (Compute Capability 12.0). |
| cuDNN | 9.x | DNN primitives | FlashAttention integration, attention backends. cuDNN 9.10+ for Blackwell optimizations. |
| CUTLASS | 3.x | GEMM templates | Custom GEMM kernels for MoE, quantization. TensorRT-LLM uses CUTLASS for FP8/INT4 kernels. |

### Speculative Decoding Stack

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **FlashInfer** | 0.6+ | Sampling kernels | Torch Sampler in TensorRT-LLM/vLLM uses FlashInfer for chain speculative sampling, rejection sampling. Sorting-free GPU kernels for LLM sampling. |
| **Draft Model Support** | N/A | Self-speculative | EAGLE3, MTP (Multi-Token Prediction), self-speculative decoding built into model architecture |
| **N-gram/GPU N-gram** | N/A | Lightweight spec | No extra model. N-gram lives in GPU memory. Good for throughput vs latency tradeoff. |

### Beam Search Stack

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Torch Sampler** | Default | Sampling backend | TensorRT-LLM default sampler. Supersedes deprecated TRTLLMSampler. Uses FlashInfer kernels. |
| **FlashInfer Sampling** | 0.6+ | TopK/TopP/Nucleus | Sorting-free implementations, batched sampling with heterogeneous parameters. |

### KV Cache Improvements Stack

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **FlashInfer (page)** | 0.6+ | Paged KV-Cache | `append_paged_kv_cache`, `append_paged_mla_kv_cache` for vLLM-compatible page tables |
| **FlashAttention** | 3.x | Attention kernel | v2.6+ for NHD/HND layouts, cascading attention states |
| **TensorRT-LLM Attention** | Latest | Production kernels | TrtllmAttention backend, XQA kernel (2.4x Llama-70B throughput improvement) |
| **FlashMLA** | Latest | H100-optimized | Low-latency attention for Hopper architecture |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **xGrammar** | Latest | Guided decoding | Structured output with speculative decoding. Replaces guidance library. |
| **CTranslate2** | 3.x | Alternative backend | If needing CPU/portable inference alongside CUDA |
| **vLLM kernels** | 0.20+ | Reference implementation | PagedAttention kernel reference (`csrc/attention/`) |
| **Triton** | 3.x | Kernel authoring | Custom attention kernels, piecewise CUDA graphs |

## Integration Points for Existing Architecture

### Five-Layer Integration (memory → device → algo → api)

Based on existing v2.6 FlashAttention + PagedAttention + FP8 quantization stack:

```
Layer 5 (API):
  - Add: SpeculativeDecodingConfig, BeamSearchParams, KVCacheConfigV2

Layer 4 (Algorithm):
  - Add: spec_decode/* strategies (EAGLE3, MTP, Ngram, Suffix)
  - Add: beam_search module
  - Add: cascade attention wrapper

Layer 3 (Device/Algo):
  - Add: FlashInfer attention backends
  - Add: XQA kernel integration
  - Add: Streaming attention for speculative prefix

Layer 2 (Memory):
  - Modify: KV cache block allocation for cascade layout
  - Add: Paged KV-Cache with FlashInfer page API
  - Add: KV quantization support (NVFP4 for KV cache)
```

### Existing v2.6 → v2.12+ Upgrade Path

| Feature | Existing | Required Addition |
|---------|----------|-------------------|
| FlashAttention | 2.6 | Upgrade to 3.x for Blackwell, HND layout support |
| PagedAttention | 2.6 | Add FlashInfer page API compatibility |
| FP8 Quantization | 2.12 | Already present; ensure INT4 KV quantization compatible |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **TRTLLMSampler** | Deprecated in TensorRT-LLM 1.4+, removed | Torch Sampler with FlashInfer |
| **Old FlashAttention 1.x** | Missing Ring Attention, cascade states | FlashAttention 3.x |
| **CUTLASS 2.x** | Missing Blackwell support, FP4 tensors | CUTLASS 3.x |
| **Guided Decoding (guidance lib)** | Not compatible with speculative decoding | xGrammar |
| **PyTorch naive attention** | 10x slower than FlashAttention | FlashAttention or FlashInfer |
| **v0.10.x vLLM** | No draft model speculative decoding | Upgrade to v0.20+ |

## Installation

```bash
# Core CUDA dependencies
# CUDA 12.8+ for Blackwell (H200, B200, GB200)
# cuDNN 9.10+ for Blackwell attention

# FlashInfer (required for spec decode + sampling)
pip install flashinfer

# FlashAttention 3.x (required for modern attention)
pip install flash-attn --no-build-isolation

# xGrammar (for guided decoding with spec decode)
pip install xgrammar

# TensorRT-LLM (if deploying with TRT)
# See: nvidia.github.io/TensorRT-LLM/install/index.html
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| FlashInfer | vLLM native sampler | Only if avoiding new dependency; vLLM sampler is well-optimized |
| FlashAttention 3.x | FlashAttention 2.x | Only if targeting legacy Ampere (30xx) without Blackwell needs |
| Torch Sampler | Custom CUDA sampler | Only if needing features FlashInfer doesn't support |
| XQA kernel | FlashAttention decode | XQA is 2.4x faster on H100 for decode; FlashAttention better for prefill |

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| FlashInfer 0.6+ | CUDA 12.x, PyTorch 2.x | Requires CUDA 12 for TMA support |
| FlashAttention 3.x | CUDA 12.x (Hopper), 12.8+ (Blackwell) | 3.x uses TMA, async copy |
| TensorRT-LLM latest | CUDA 12.x, cuDNN 9.x | Check release notes for Blackwell support timeline |
| vLLM 0.20+ | CUDA 12.x, FlashInfer 0.5+ | v0.20 adds v1 spec decode engine |

## Key Integration Notes

### Speculative Decoding Requires
1. **Rejection sampling kernel** (FlashInfer `chain_speculative_sampling`)
2. **KV cache access for draft tokens** (existing PagedAttention extends here)
3. **Parallel drafting support** (EAGLE3, PARD for multi-token speculation)

### Beam Search Requires
1. **Beam width management in KV cache** (block allocation per beam)
2. **Logits processor integration** (for prefix caching across beams)
3. **Sampling back-compat** (max_beam_width = best_of)

### KV Cache Improvements
1. **NVFP4 KV quantization** (FlashInfer `nvfp4_kv_quantize`)
2. **Cascade attention** (merge states for prefix reuse)
3. **Prefix caching** (automatic with block-level hash)
4. **Disaggregated serving** (KV transfer between prefill/decode)

## Sources

- **TensorRT-LLM Documentation** — nvidia.github.io/TensorRT-LLM/ — Speculative Decoding, Sampling, KV Cache System (verified 2026-04-26)
- **FlashInfer 0.6.9 Documentation** — docs.flashinfer.ai/ — Attention kernels, sampling, page API (verified current)
- **vLLM Documentation** — docs.vllm.ai/ — PagedAttention design, speculative decoding methods (verified 2026-04-10)
- **CUDA C++ Programming Guide** — docs.nvidia.com/cuda/cuda-c-programming-guide/ — CUDA 13.2 (verified current)
- **LightLLM GitHub** — github.com/ModelTC/LightLLM — Reference for token-level KV cache management, 4k stars
- **vLLM GitHub** — github.com/vllm-project/vllm — 79k stars, state-of-the-art reference

---

*Stack additions for Speculative Decoding, Beam Search, and KV Cache improvements*
*Researched: 2026-05-05*
