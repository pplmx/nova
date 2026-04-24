# Roadmap: Nova CUDA Library Enhancement

**Created:** 2026-04-23
**Updated:** 2026-04-24
**Granularity:** Standard

## Milestones

- ✅ **v1.0 Production Release** — Phases 1-6 (shipped 2026-04-24)
- ✅ **v1.1 Multi-GPU Support** — Phases 7-10 (shipped 2026-04-24)
- 🚧 **v1.2** — Planning needed

## Phase Progress

<details>
<summary>✅ v1.0 Production Release (Phases 1-6) — SHIPPED 2026-04-24</summary>

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 1 | Performance Foundations | Device-aware kernels, memory metrics, validation, benchmarks | PERF-01 to PERF-06, BMCH-01 to BMCH-04 | ✅ Complete |
| 2 | Async & Streaming | CUDA streams, pinned memory, pool improvements | ASYNC-01 to ASYNC-04, POOL-01 to POOL-04 | ✅ Complete |
| 3 | FFT Module | Fast Fourier Transform implementation | FFT-01 to FFT-04 | ✅ Complete |
| 4 | Ray Tracing Primitives | Intersection tests and BVH helpers | RAY-01 to RAY-04 | ✅ Complete |
| 5 | Graph Algorithms | BFS and PageRank on GPU | GRAPH-01 to GRAPH-04 | ✅ Complete |
| 6 | Neural Net Primitives | Matmul, softmax, ReLU, layer norm | NN-01 to NN-04 | ✅ Complete |

See `.planning/milestones/v1.0-ROADMAP.md` for full phase details.

</details>

<details>
<summary>✅ v1.1 Multi-GPU Support (Phases 7-10) — SHIPPED 2026-04-24</summary>

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 7 | Device Mesh Detection | GPU enumeration, peer access matrix, async P2P copy | MGPU-01 to MGPU-04 | ✅ Complete |
| 8 | Multi-GPU Data Parallelism | All-reduce, broadcast, all-gather, barrier sync | MGPU-05 to MGPU-08 | ✅ Complete |
| 9 | Distributed Memory Pool | Per-device pools, auto-allocation, cross-device tracking | MGPU-09 to MGPU-11 | ✅ Complete |
| 10 | Multi-GPU Matmul | Row-wise split matmul, single-GPU fallback | MGPU-12 to MGPU-13 | ✅ Complete |

See `.planning/milestones/v1.1-ROADMAP.md` for full phase details.

</details>

---

## Next Milestone (v1.2)

**Status:** Planning needed

**Candidate features:**
- NCCL integration for optimized multi-GPU collectives
- Tensor parallelism for large layer support
- Pipeline parallelism for deep model support
- Distributed batch normalization
- Device mesh topology optimization (NVLink-aware)

---

*Roadmap updated: 2026-04-24*
