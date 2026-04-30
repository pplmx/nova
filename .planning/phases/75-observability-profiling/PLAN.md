# Phase 75: Observability & Profiling - Plan

**Phase:** 75
**Status:** Planned

## Requirements

| ID | Requirement |
|----|-------------|
| OBS-01 | User can export timeline visualizations in Chrome trace format from NVTX annotations |
| OBS-02 | User can measure memory bandwidth (H2D/D2H/D2D) via NVbandwidth integration |
| OBS-03 | User can collect kernel statistics (latency, throughput, occupancy) per kernel launch |
| OBS-04 | User can analyze real-time occupancy and receive feedback on block size selection |

## Implementation

### 1. Chrome Trace Timeline Export (OBS-01)

**Files:**
- `cuda/observability/timeline.hpp` - Timeline exporter
- `cuda/observability/timeline.cpp` - Implementation
- `tests/observability/timeline_test.cpp` - Tests

**Implementation:**
```cpp
// Timeline export to Chrome trace format
// JSON structure: {traceEvents: [...]}
// Events include: name, cat (category), ts (timestamp), dur (duration), pid, tid
```

**Success criteria:**
- User can export Chrome trace format files from NVTX annotations
- Files load in chrome://tracing

### 2. Memory Bandwidth Measurement (OBS-02)

**Files:**
- `cuda/observability/bandwidth_tracker.hpp` - Bandwidth measurement API
- `cuda/observability/bandwidth_tracker.cpp` - Implementation
- `tests/observability/bandwidth_tracker_test.cpp` - Tests

**Implementation:**
```cpp
// Wrapper around NVbandwidth or custom timing kernels
// Measure H2D, D2H, D2D bandwidth
// Provide DeviceMemoryBandwidth class
```

**Success criteria:**
- User can measure H2D, D2H, D2D memory bandwidth

### 3. Kernel Statistics Collection (OBS-03)

**Files:**
- `cuda/observability/kernel_stats.hpp` - Stats collector
- `cuda/observability/kernel_stats.cpp` - Implementation
- `tests/observability/kernel_stats_test.cpp` - Tests

**Implementation:**
```cpp
// Per-kernel timing and statistics
// cudaEvent_t for latency, throughput calculation
// Achieved occupancy via cudaOccupancy* APIs
```

**Success criteria:**
- User can collect per-kernel statistics (latency, throughput, occupancy)

### 4. Occupancy Analyzer (OBS-04)

**Files:**
- `cuda/observability/occupancy_analyzer.hpp` - Occupancy feedback
- `cuda/observability/occupancy_analyzer.cpp` - Implementation
- `tests/observability/occupancy_analyzer_test.cpp` - Tests

**Implementation:**
```cpp
// cudaOccupancyMaxPotentialBlockSize for feedback
// Register/shared memory pressure analysis
// Block size recommendations
```

**Success criteria:**
- User can receive real-time feedback on block size selection

## Files to Create

| File | Purpose |
|------|---------|
| `cuda/observability/timeline.hpp` | Chrome trace timeline export |
| `cuda/observability/bandwidth_tracker.hpp` | Memory bandwidth measurement |
| `cuda/observability/kernel_stats.hpp` | Kernel statistics collection |
| `cuda/observability/occupancy_analyzer.hpp` | Occupancy feedback |
| `tests/observability/*_test.cpp` | Tests for each component |
| `cuda/observability/CMakeLists.txt` | Build configuration |

## Test Plan

1. Timeline export generates valid JSON loadable in chrome://tracing
2. Bandwidth tracker measures within 5% of nvbandwidth CLI
3. Kernel stats collect accurate latency/throughput
4. Occupancy analyzer recommends appropriate block sizes
