# Roadmap: Nova CUDA Library Enhancement

**Created:** 2026-04-23
**Granularity:** Standard (6 phases, 3-5 plans each)

## Phase Summary

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Performance Foundations | Device-aware kernels, memory metrics, validation, benchmarks | PERF-01 to PERF-06, BMCH-01 to BMCH-04 | 10 criteria |
| 2 | Async & Streaming | CUDA streams, pinned memory, pool improvements | ASYNC-01 to ASYNC-04, POOL-01 to POOL-04 | 8 criteria |
| 3 | FFT Module | Fast Fourier Transform implementation | FFT-01 to FFT-04 | 4 criteria |
| 4 | Ray Tracing Primitives | Intersection tests and BVH helpers | RAY-01 to RAY-04 | 4 criteria |
| 5 | Graph Algorithms | BFS and PageRank on GPU | GRAPH-01 to GRAPH-04 | 4 criteria |
| 6 | Neural Net Primitives | Matmul, softmax, ReLU, layer norm | NN-01 to NN-04 | 4 criteria |

---

## Phase 1: Performance Foundations

**Goal:** Make the library device-aware, measurable, and production-ready

**Requirements:**
- PERF-01: Device capability queries
- PERF-02: Auto block size selection
- PERF-03: Memory pool statistics
- PERF-04: Memory usage interface
- PERF-05: Input validation
- PERF-06: Enhanced error messages
- BMCH-01: Warm-up benchmark runs
- BMCH-02: Variance reporting
- BMCH-03: Throughput/latency metrics
- BMCH-04: Regression detection

**Success Criteria:**
1. Device info service queries compute capability and memory bandwidth
2. Kernel launcher accepts device-aware block size parameter
3. Memory pool exposes hits/misses/fragmentation via metrics interface
4. `cuda::memory::available()` and `cuda::memory::used()` functions work
5. Public APIs assert valid input ranges with clear error messages
6. Exceptions include operation name, dimensions, and device info
7. Benchmark runner executes warm-up iterations before measurement
8. Benchmark results include mean ± stddev
9. Throughput reported in GB/s for memory operations
10. New benchmark run compares against baseline, reports delta %

**Key Files:**
- `include/cuda/performance/device_info.h` (new)
- `include/cuda/performance/memory_metrics.h` (new)
- `include/cuda/benchmark/benchmark.h` (new)
- `include/cuda/device/error.h` (modify)
- `include/cuda/algo/kernel_launcher.h` (modify)
- `include/cuda/memory/memory_pool.h` (modify)
- `tests/benchmark/*.cpp` (new)

**Plans:** 3 plans
Plans:
- [ ] 01-01-PLAN.md — Device Info and Kernel Launcher Enhancement (PERF-01, PERF-02)
- [ ] 01-02-PLAN.md — Memory Metrics and Error Enhancement (PERF-03, PERF-04, PERF-05, PERF-06)
- [ ] 01-03-PLAN.md — Benchmark Suite (BMCH-01, BMCH-02, BMCH-03, BMCH-04)

---

## Phase 2: Async & Streaming

**Goal:** Enable non-blocking GPU operations and efficient memory transfers

**Requirements:**
- ASYNC-01: Stream manager with priorities
- ASYNC-02: Pinned memory allocator
- ASYNC-03: Async copy primitives
- ASYNC-04: Event-based synchronization
- POOL-01: Pool defragmentation
- POOL-02: Fragmentation percentage reporting
- POOL-03: Multi-stream pool integration
- POOL-04: Graceful pool limits

**Success Criteria:**
1. Stream manager creates streams with configurable priority
2. Pinned memory allocator returns page-locked host pointers
3. Async copy accepts stream parameter, returns immediately
4. Events record and synchronize between operations
5. Memory pool defrag() compacts free blocks
6. Pool reports fragmentation % via metrics interface
7. Pool tracks allocations per-stream
8. Pool allocation returns null/throws when limit reached

**Key Files:**
- `include/cuda/async/stream_manager.h` (new)
- `include/cuda/async/pinned_memory.h` (new)
- `include/cuda/async/async_copy.h` (new)
- `include/cuda/memory/memory_pool.h` (modify)
- `tests/async/*.cpp` (new)

**Plans:** 2 plans
Plans:
- [ ] 02-01-PLAN.md — Stream Manager and Pinned Memory (ASYNC-01, ASYNC-02)
- [ ] 02-02-PLAN.md — Async Copy and Memory Pool v2 (ASYNC-03, POOL-03, POOL-04)

---

## Phase 3: FFT Module

**Goal:** Fast Fourier Transform for signal and image processing

**Requirements:**
- FFT-01: FFT plan creation
- FFT-02: Forward transform
- FFT-03: Inverse transform
- FFT-04: Plan destruction

**Success Criteria:**
1. `FFTPlan` constructor accepts size and direction parameters
2. Forward FFT produces correct frequency domain output (verified against reference)
3. Inverse FFT recovers original signal within numerical tolerance (1e-5)
4. FFTPlan destructor frees all associated resources

**Key Files:**
- `include/cuda/fft/fft.h` (new)
- `include/cuda/fft/fft_types.h` (new)
- `src/cuda/fft/fft.cu` (new)
- `tests/fft/*_test.cu` (new)

**Plans:** 1 plan
Plans:
- [ ] 03-01-PLAN.md — FFT Module Implementation (FFT-01, FFT-02, FFT-03, FFT-04)

---

## Phase 4: Ray Tracing Primitives

**Goal:** GPU-accelerated ray intersection and BVH construction

**Requirements:**
- RAY-01: Ray-box intersection
- RAY-02: Ray-sphere intersection
- RAY-03: BVH construction
- RAY-04: Ray-BVH traversal

**Success Criteria:**
1. Ray-box intersection returns tNear, tFar, and hit normal
2. Ray-sphere intersection handles miss, hit, and inside cases
3. BVH construction reduces primitive tests by >50% vs linear search
4. Ray-BVH traversal visits only nodes along ray path

**Key Files:**
- `include/cuda/raytrace/primitives.h` (new)
- `include/cuda/raytrace/bvh.h` (new)
- `src/cuda/raytrace/*.cu` (new)
- `tests/raytrace/*_test.cu` (new)

---

## Phase 5: Graph Algorithms

**Goal:** GPU-accelerated graph processing with BFS and PageRank

**Requirements:**
- GRAPH-01: BFS distance computation
- GRAPH-02: Disconnected component handling
- GRAPH-03: PageRank convergence
- GRAPH-04: CSR format storage

**Success Criteria:**
1. BFS correctly computes distance from source to all reachable vertices
2. BFS terminates early on disconnected components, marks unreachable as -1
3. PageRank converges to 1e-6 tolerance within 50 iterations
4. CSR format stores graph with O(V+E) memory for V vertices, E edges

**Key Files:**
- `include/cuda/graph/bfs.h` (new)
- `include/cuda/graph/pagerank.h` (new)
- `src/cuda/graph/*.cu` (new)
- `tests/graph/*_test.cu` (new)

---

## Phase 6: Neural Net Primitives

**Goal:** GPU kernels for common deep learning operations

**Requirements:**
- NN-01: Matrix multiply correctness
- NN-02: Numerically stable softmax
- NN-03: Leaky ReLU
- NN-04: Layer normalization

**Success Criteria:**
1. Matmul output matches cuBLAS reference within 1e-3 absolute error
2. Softmax produces no NaN values for any input range
3. Leaky ReLU supports configurable alpha (default 0.01)
4. Layer normalization produces correct mean (≈0) and variance (≈1)

**Key Files:**
- `include/cuda/neural/matmul.h` (new)
- `include/cuda/neural/softmax.h` (new)
- `include/cuda/neural/activations.h` (new)
- `include/cuda/neural/layer_norm.h` (new)
- `src/cuda/neural/*.cu` (new)
- `tests/neural/*_test.cu` (new)

---

## Milestone

| Milestone | Phases | Target |
|-----------|--------|--------|
| Production Ready | 1-2 | Core infrastructure complete |
| Feature Complete | 1-6 | All algorithms implemented |
| Stable Release | 1-6 + testing | Full test suite passing |

---

*Roadmap created: 2026-04-23*
