# Phase 26: Performance Profiling - Context

**Phase:** 26
**Goal:** Add profiling infrastructure for kernel and collective performance
**Requirements:** PROF-01 to PROF-03

## Requirements Analysis

### PROF-01: Kernel-level profiling with CUDA profiling tools integration

Integration points:
- CUDA events for timing
- nvprof/nsys markers
- Custom annotation macros

### PROF-02: Memory bandwidth and compute throughput metrics

Metrics to track:
- Memory throughput (GB/s)
- Compute throughput (TFLOPS)
- Kernel occupancy
- Latency per operation

### PROF-03: Collective operation latency tracking

Track:
- All-reduce latency
- Broadcast latency
- All-gather latency
- Barrier latency

## Design Decisions

### D-01: Profiling Interface

**Decision:** Scoped timer RAII pattern

**Rationale:**
- Automatic start/stop via destructor
- Stack-allocated for minimal overhead
- Works with existing code without major refactoring

### D-02: Metrics Collection

**Decision:** In-memory metrics with optional export

**Rationale:**
- Minimal I/O overhead during profiling
- Export on demand to JSON/text
- Thread-safe for multi-threaded use

### D-03: Overhead Control

**Decision:** Compile-time enable/disable

**Rationale:**
- Zero overhead when disabled
- Simple macro-based API
- CUDA events are lightweight

## Architecture

```
PerformanceProfiler
├── Timer
│   ├── start()
│   └── stop()
├── MetricsCollector
│   ├── record_kernel_time()
│   ├── record_memory_bandwidth()
│   └── record_collective_latency()
├── ReportGenerator
│   ├── generate_summary()
│   └── export_json()
└── Global singleton
```

## Implementation Plan

1. **Files to create:**
   - `include/cuda/performance/profiler.h`
   - `src/cuda/performance/profiler.cpp`

2. **Dependencies:**
   - Existing performance/benchmark code for reference

## Success Criteria

1. Kernel timing accurate to microsecond precision
2. Memory bandwidth reported for memory operations
3. Collective latencies tracked per operation type
4. Zero overhead when profiling disabled

## References

- CUDA Profiling Tools Interface (CUPTI)
- NVIDIANsight Systems
- nvprof documentation
