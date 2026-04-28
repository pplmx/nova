# Requirements: v2.4 Production Hardening

**Project:** Nova CUDA Library Enhancement
**Milestone:** v2.4 Production Hardening
**Status:** Draft
**Date:** 2026-04-28

---

## Overview

Add production-grade hardening features: CUDA graphs, performance optimization, observability improvements, and stress testing infrastructure.

---

## Phase 59: CUDA Graphs Foundation

### GRAPH-01: GraphExecutor Core

Implement `cuda/production/graph_executor.h` for CUDA Graphs capture and replay.

**Acceptance criteria:**
- `GraphExecutor::capture()` starts graph capture on a stream
- `GraphExecutor::update()` updates parameters without full rebuild
- `GraphExecutor::launch()` replays the graph
- Supports up to 64 parameter nodes

**Rationale:** CUDA Graphs reduce kernel launch overhead by 10-50x for batch workloads.

---

### GRAPH-02: Memory Node Integration

Add memory allocation nodes to `GraphExecutor`.

**Acceptance criteria:**
- Memory nodes for device, host-pinned, and managed memory
- Automatic dependency management between memory and compute nodes
- Proper cleanup on graph destruction

**Rationale:** Memory operations must be captured to avoid implicit synchronization.

---

### GRAPH-03: Algorithm Graph Wrappers

Wrap existing algorithms for graph execution.

**Acceptance criteria:**
- `wrap_for_graph(reduce)` - capture reduce operation
- `wrap_for_graph(scan)` - capture scan operation
- `wrap_for_graph(sort)` - capture sort operation
- Integration with existing algo namespace

**Rationale:** Core algorithms benefit most from graph capture.

---

## Phase 60: Performance Optimization

### PERF-01: L2 Cache Persistence

Implement `cuda/production/l2_persistence.h` for L2 cache control.

**Acceptance criteria:**
- `L2PersistenceManager` class with `set_persistence_size()` and `restore_defaults()`
- Configurable cache budget (0-100% of available L2)
- RAII pattern for automatic cleanup

**Rationale:** L2 cache persistence improves performance for iterative algorithms working within a working set.

---

### PERF-02: Priority Stream Pool

Extend stream management with priority queues.

**Acceptance criteria:**
- `PriorityStreamPool` with low/medium/high priority streams
- `acquire_priority(Priority)` method
- Stream recycling with priority-aware eviction
- Integration with existing StreamManager

**Rationale:** Priority scheduling allows latency-sensitive operations to bypass queued work.

---

### PERF-03: NVBench Integration

Add GPU-native microbenchmarking with NVBench.

**Acceptance criteria:**
- CMake integration with FetchContent NVBench
- `BENCHMARK_KERNEL(name, func, args)` macro
- Memory bandwidth benchmark suite
- Compute throughput benchmark suite

**Rationale:** NVBench provides GPU-specific measurements unavailable in Google Benchmark.

---

## Phase 61: Observability & Monitoring

### OBS-01: NVTX Domain Extensions

Extend NVTX annotations with per-layer domains.

**Acceptance criteria:**
- Domain for each layer: memory, device, algo, api, production
- Range helpers: `NVTXScopedRange`, `NVTXAutoPush`
- Compile-time toggle via `NOVA_NVTX_ENABLED`
- Performance impact < 1% when disabled

**Rationale:** Per-layer domains improve timeline readability in Nsight Visual Studio Edition.

---

### OBS-02: Async Error Tracker

Implement async error propagation for deferred CUDA errors.

**Acceptance criteria:**
- `AsyncErrorTracker` class with `record()` and `check()` methods
- Automatic tracking after kernel launches
- Aggregated error reporting with context

**Rationale:** CUDA kernel errors are deferred until synchronization; tracking enables proactive detection.

---

### OBS-03: Health Metrics Dashboard

Add production health monitoring endpoints.

**Acceptance criteria:**
- `HealthMetrics` struct with utilization, memory pressure, error rates
- `get_health_snapshot()` returning current system state
- JSON serialization for monitoring integration

**Rationale:** Production deployments need health monitoring for alerting and dashboards.

---

## Phase 62: Stress Testing

### STRESS-01: Error Injection Framework

Implement chaos testing primitives.

**Acceptance criteria:**
- `ErrorInjector` class with `inject_once()` and `inject_always()` modes
- Target selection: allocation, launch, collective
- Statistics tracking: injection count, affected operations

**Rationale:** Error injection validates fault tolerance without real failures.

---

### STRESS-02: Memory Pressure Tests

Add tests under constrained memory conditions.

**Acceptance criteria:**
- `MemoryPressureTest` fixture with configurable limits
- Allocation failure injection
- Recovery path testing
- Integration with existing fuzz framework

**Rationale:** Production environments may hit memory limits; recovery must be tested.

---

### STRESS-03: Concurrent Stream Tests

Stress test with multiple concurrent stream operations.

**Acceptance criteria:**
- Cross-stream synchronization edge cases
- Priority inversion scenarios
- Deadlock detection
- Timeout-based test termination

**Rationale:** Concurrent streams expose race conditions and synchronization bugs.

---

## Phase 63: Integration & Polish

### INT-01: Build System Integration

Ensure all new components build correctly.

**Acceptance criteria:**
- CMake targets: `nova_production`, `nova_production_tests`
- CUDA 20 required features: managed memory, graphs
- Test coverage > 80% for new code

**Rationale:** Integration validates that components work together.

---

### INT-02: Documentation

Document production hardening features.

**Acceptance criteria:**
- Production hardening guide (PRODUCTION.md)
- API reference for GraphExecutor, L2PersistenceManager, etc.
- Migration guide from v2.3 to v2.4

**Rationale:** Users need documentation to adopt new features.

---

### INT-03: Regression Testing

Validate no performance regression from new components.

**Acceptance criteria:**
- Benchmark suite comparison against v2.3 baseline
- Memory overhead validation
- Initialization time impact < 10ms

**Rationale:** New features must not degrade existing performance.

---

## Requirement Summary

| ID | Requirement | Phase | Priority |
|----|-------------|-------|----------|
| GRAPH-01 | GraphExecutor Core | 59 | P0 |
| GRAPH-02 | Memory Node Integration | 59 | P0 |
| GRAPH-03 | Algorithm Graph Wrappers | 59 | P1 |
| PERF-01 | L2 Cache Persistence | 60 | P1 |
| PERF-02 | Priority Stream Pool | 60 | P1 |
| PERF-03 | NVBench Integration | 60 | P2 |
| OBS-01 | NVTX Domain Extensions | 61 | P1 |
| OBS-02 | Async Error Tracker | 61 | P1 |
| OBS-03 | Health Metrics Dashboard | 61 | P2 |
| STRESS-01 | Error Injection Framework | 62 | P0 |
| STRESS-02 | Memory Pressure Tests | 62 | P1 |
| STRESS-03 | Concurrent Stream Tests | 62 | P1 |
| INT-01 | Build System Integration | 63 | P0 |
| INT-02 | Documentation | 63 | P1 |
| INT-03 | Regression Testing | 63 | P0 |

**Total:** 15 requirements across 5 phases

---

*Requirements defined: 2026-04-28*
