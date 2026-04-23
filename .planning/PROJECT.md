# Nova CUDA Library Enhancement

## What This Is

A production-ready CUDA parallel algorithms library with a five-layer architecture, supporting education, extensibility, and production use cases. This project adds production-quality foundations and new algorithm capabilities.

## Core Value

A reliable, high-performance CUDA compute library that can be trusted in production environments, with comprehensive algorithms for scientific computing, image processing, and emerging workloads.

## Requirements

### Validated

- ✓ Five-layer CUDA architecture (memory → device → algo → api) — existing
- ✓ Memory management (Buffer, unique_ptr, MemoryPool) — existing
- ✓ Algorithm wrappers (reduce, scan, sort, histogram) — existing
- ✓ Image processing (blur, sobel, morphology, brightness) — existing
- ✓ Matrix operations (add, mult, ops) — existing
- ✓ 81+ tests across 13 test suites — existing
- ✓ CMake build with Google Test — existing

### Active

- [ ] Device capability-aware kernel launch configuration
- [ ] Memory metrics and profiling hooks
- [ ] Input validation with enhanced error context
- [ ] Benchmark suite for throughput/latency measurement
- [ ] Async operations with CUDA streams
- [ ] Pinned memory for faster host-device transfers
- [ ] Memory pool improvements (defragmentation, metrics)
- [ ] Multi-GPU support foundation
- [ ] FFT (Fast Fourier Transform)
- [ ] Ray tracing primitives (ray-box, ray-sphere, BVH helpers)
- [ ] Graph algorithms (BFS, PageRank on GPU)
- [ ] Neural network primitives (matmul, softmax, ReLU, layer norm)

### Out of Scope

- Real-time video processing pipeline — requires streams first
- Distributed multi-node computation — beyond single-GPU scope
- Python bindings — separate project
- GPU memory allocator tuning beyond pool improvements

## Context

**Existing project:** nova CUDA library at `https://github.com/pplmx/nova`
- C++20, CUDA 17, CMake 3.25+
- Target architectures: 6.0, 7.0, 8.0, 9.0 (Pascal through Ampere)
- Five-layer architecture with clear separation of concerns
- 81 tests using Google Test v1.14.0

**Known limitations from codebase map:**
- Hardcoded block sizes (256) — not device-aware
- No memory usage metrics
- No async/streaming operations
- No benchmark suite
- Memory pool lacks defragmentation

## Constraints

- **Tech stack:** C++20, CUDA 17, CMake 3.25+ — must maintain compatibility
- **Backward compatibility:** Existing API must not break
- **Testing:** All new code requires tests, maintain 80%+ coverage
- **Performance:** New implementations must not regress existing algorithms

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Foundation-first phasing | Quality foundations enable reliable feature work | — Pending |
| Streams for async | Native CUDA streams, not abstraction layer | — Pending |
| FFTW-style API | Familiar interface for signal processing users | — Pending |
| BVH helpers over full ray tracer | Focus on GPU compute primitives | — Pending |

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
*Last updated: 2026-04-23 after initialization*
