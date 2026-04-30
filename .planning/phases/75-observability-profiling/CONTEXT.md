# Phase 75: Observability & Profiling - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

## Phase Boundary

Users can visualize kernel execution timelines, measure memory bandwidth, collect kernel statistics, and analyze occupancy.

**Requirements:**
- OBS-01: Timeline visualization with Chrome trace format export
- OBS-02: Memory bandwidth measurement via NVbandwidth integration
- OBS-03: Kernel stats collection (latency, throughput, occupancy)
- OBS-04: Real-time occupancy analyzer with feedback

## Implementation Decisions

### Timeline Visualization (OBS-01)
- Export NVTX annotations to Chrome trace format
- Use existing NVTX infrastructure from v1.7/v2.4
- Support chrome://tracing import for visualization

### Memory Bandwidth Measurement (OBS-02)
- Integrate NVbandwidth CLI tool via CMake detection
- Provide C++ wrapper API for programmatic measurement
- Support H2D, D2H, D2D transfer types

### Kernel Statistics (OBS-03)
- Extend existing CUDA event-based timing from v1.6
- Collect per-kernel latency, throughput, achieved occupancy
- Integrate with NVTX domains for labeling

### Occupancy Analyzer (OBS-04)
- Use cudaOccupancyMaxPotentialBlockSize for feedback
- Provide real-time block size recommendations
- Include register/shared memory pressure analysis

### Stack Addition
From research: CCCL migration (CUB → CCCL) required. NVIDIA tools (Compute Sanitizer, Nsight) are free with CUDA Toolkit.

## Existing Code Insights

From codebase:
- NVTX infrastructure exists in observability/ directory (v1.7, v2.4)
- CUDA event timing in device/ timing.hpp
- Health metrics dashboard in production/ (v2.4)
- NVTX domain extensions per layer

## Specific Ideas

Based on ROADMAP success criteria:
1. Chrome trace format export functionality
2. NVbandwidth integration wrapper
3. Kernel stats collection per launch
4. Occupancy calculator feedback API
