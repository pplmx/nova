# Nova CUDA Library Enhancement

## What This Is

A production-ready CUDA parallel algorithms library with a five-layer architecture, supporting education, extensibility, and production use cases. This project adds production-quality foundations and new algorithm capabilities.

## Current Milestone: v1.3 NCCL Integration, Tensor & Pipeline Parallelism

**Previous milestone:** v1.2 Toolchain Upgrade — SHIPPED 2026-04-24

**Goal:** Enable efficient multi-GPU training with NCCL-based collectives, tensor parallelism for large layers, and pipeline parallelism for deep models.

**Target features:**
- NCCL integration for optimized multi-GPU collectives
- Tensor parallelism for large layer support
- Pipeline parallelism for deep model support
- Distributed batch normalization

## Core Value

A reliable, high-performance CUDA compute library that can be trusted in production environments, with comprehensive algorithms for scientific computing, image processing, and emerging workloads.

## Requirements

### Validated

- ✓ Five-layer CUDA architecture (memory → device → algo → api) — existing
- ✓ Memory management (Buffer, unique_ptr, MemoryPool) — existing
- ✓ Algorithm wrappers (reduce, scan, sort, histogram) — existing
- ✓ Image processing (blur, sobel, morphology, brightness) — existing
- ✓ Matrix operations (add, mult, ops) — existing
- ✓ Device capability queries and auto block size selection — v1.0
- ✓ Memory pool statistics and fragmentation reporting — v1.0
- ✓ Stream-based async operations with event synchronization — v1.0
- ✓ Signal/image processing via FFT — v1.0
- ✓ Ray tracing primitives (ray-box, ray-sphere, BVH) — v1.0
- ✓ Graph processing (BFS, PageRank) — v1.0
- ✓ Deep learning primitives (matmul, activations, normalization) — v1.0
- ✓ Device mesh detection and peer memory access — v1.1
- ✓ Multi-GPU data parallelism primitives (reduce, broadcast, all-gather, barrier) — v1.1
- ✓ Distributed memory pool across GPU devices — v1.1
- ✓ Multi-GPU matmul with single-GPU fallback — v1.1
- ✓ C++23 standard (CMAKE_CXX_STANDARD 23) — v1.2
- ✓ CUDA 20 standard (CMAKE_CUDA_STANDARD 20) — v1.2
- ✓ CMake 4.0+ minimum version — v1.2
- ✓ 444 tests passing — v1.2

### Active

- [ ] NCCL integration for optimized multi-GPU collectives — Phase 13-14
- [ ] Extended NCCL collectives with unified fallback — Phase 15
- [ ] Tensor parallelism for large layer support — Phase 16
- [ ] Pipeline parallelism for deep model support — Phase 17
- [ ] Distributed batch normalization — v2 (deferred)

### Out of Scope

- Distributed multi-node computation (multiple nodes, not just multiple GPUs) — future work
- Python bindings — separate project
- Real-time video processing pipeline — not in scope

## Context

**Project:** nova CUDA library at `https://github.com/pplmx/nova`
- **Current:** C++23, CUDA 20, CMake 4.0+
- Target architectures: 6.0, 7.0, 8.0, 9.0 (Pascal through Ampere)
- Five-layer architecture with clear separation of concerns
- **444 tests using Google Test v1.14.0**
- **v1.2 shipped:** Toolchain upgrade (C++23, CUDA 20, CMake 4.0)

**Current capabilities:**
- Device mesh detection and peer memory access between GPUs
- Multi-GPU collective operations (all-reduce, broadcast, all-gather, barrier)
- Distributed memory pool spanning multiple GPUs
- Multi-GPU matrix multiply with single-GPU fallback
- All v1.0 features: FFT, Ray Tracing, Graph Algorithms, Neural Net Primitives, Async/Streaming

## Constraints

- **Tech stack:** C++23, CUDA 20, CMake 4.0+ — current versions
- **Backward compatibility:** Existing API must not break
- **Testing:** All existing tests must pass after upgrade
- **Performance:** New implementations must not regress existing algorithms

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Foundation-first phasing | Quality foundations enable reliable feature work | ✓ v1.0 shipped |
| Streams for async | Native CUDA streams, not abstraction layer | ✓ Implemented |
| FFTW-style API | Familiar interface for signal processing users | ✓ Implemented |
| BVH helpers over full ray tracer | Focus on GPU compute primitives | ✓ Implemented |
| P2P ring-allreduce fallback | No NCCL dependency for v1.1 | ✓ Implemented |
| Row-wise split matmul | Simple, builds on existing infrastructure | ✓ Implemented |
| Device mesh singleton | Lazy initialization, single source of truth | ✓ Implemented |
| C++23 adoption | std::expected, constexpr, ranges for modern patterns | ✓ v1.2 shipped |
| CUDA 20 standard | Next-generation CUDA toolkit for new features | ✓ v1.2 shipped |
| CMake 4.0+ | Modern CMake features and policy support | ✓ v1.2 shipped |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state (users, feedback, metrics)

---
*Last updated: 2026-04-24 after v1.3 milestone started*
