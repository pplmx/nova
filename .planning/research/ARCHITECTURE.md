# Architecture Research

**Domain:** CUDA Library Enhancement Architecture

## Component Boundaries

### Foundation Layer (Phase 1)

```
┌─────────────────────────────────────────────┐
│  Device Capability Service                   │
│  - Query device properties                   │
│  - Compute optimal block sizes               │
│  - Occupancy calculator                      │
└─────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────┐
│  Memory Metrics Service                      │
│  - Memory usage tracking                     │
│  - Pool statistics                           │
│  - Allocation profiling                      │
└─────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────┐
│  Validation Layer                            │
│  - Input validation                          │
│  - Error context enrichment                  │
└─────────────────────────────────────────────┘
```

### Async Layer (Phase 2)

```
┌─────────────────────────────────────────────┐
│  Stream Manager                              │
│  - Stream creation/destruction               │
│  - Priority streams                          │
│  - Stream synchronization                    │
└─────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────┐
│  Pinned Memory Service                       │
│  - Pinned allocation                         │
│  - Async copy primitives                     │
│  - Transfer queue                            │
└─────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────┐
│  Memory Pool v2                              │
│  - Defragmentation                           │
│  - Multi-stream support                      │
│  - Metrics integration                       │
└─────────────────────────────────────────────┘
```

### Algorithm Layer (Phase 3-6)

```
┌─────────────────────────────────────────────┐
│  Algorithm Modules                           │
│  ├── FFT (cuFFT-based)                       │
│  ├── Ray Tracing Primitives                  │
│  ├── Graph Algorithms                        │
│  └── Neural Net Primitives                   │
└─────────────────────────────────────────────┘
```

## Data Flow Patterns

### Synchronous (Current)

```
Host → Buffer → Kernel → Buffer → Host
         (blocking)
```

### Asynchronous (Target)

```
Host → PinnedBuffer → Stream → DeviceBuffer → Kernel
   ↑_____________________event_______________|
```

### Memory Pool Flow

```
Request → Pool.lookup() → Return buffer or allocate
                              ↓
                         Track stats
                              ↓
                         Return to pool on release
```

## Suggested Build Order

| Phase | Component | Dependencies |
|-------|-----------|--------------|
| 1 | Device capability service | None |
| 1 | Memory metrics service | None |
| 1 | Validation layer | Error handling patterns |
| 1 | Benchmark framework | Tests foundation |
| 2 | Stream manager | Device capability |
| 2 | Pinned memory service | Stream manager |
| 2 | Memory pool v2 | Memory metrics, stream manager |
| 3 | FFT module | cuFFT, memory pool |
| 4 | Ray tracing primitives | Memory pool |
| 5 | Graph algorithms | Memory pool |
| 6 | Neural net primitives | cuBLAS, memory pool |

## Abstraction Levels

| Level | Abstraction | Example |
|-------|-------------|---------|
| **Host API** | User-facing | `cuda::fft::transform(plan, in, out)` |
| **Wrapper** | Internal | `fft_execute(plan)` |
| **Primitive** | Device kernels | `fft_kernel<<<grid, block>>>(...)` |

## Integration Points

### With Existing Layers

- Layer 0 (memory): Extend `Buffer<T>` with metrics, async copy
- Layer 1 (device): Add device-aware kernel launch utilities
- Layer 2 (algo): Wrap new algorithms consistently
- Layer 3 (api): Integrate into `DeviceVector` and streams

### Module Structure

```
include/cuda/
├── performance/           # NEW: Profiling and metrics
│   ├── device_info.h     # Device capability queries
│   └── memory_metrics.h  # Memory tracking
├── async/                 # NEW: Async operations
│   ├── stream.h          # Stream management
│   ├── pinned_memory.h   # Pinned buffer
│   └── event.h           # Event synchronization
├── fft/                   # NEW: FFT module
│   └── fft.h             # FFT plan and execute
├── raytrace/              # NEW: Ray tracing
│   ├── primitives.h      # Ray, intersection
│   └── bvh.h             # BVH helpers
├── graph/                 # NEW: Graph algorithms
│   ├── bfs.h             # Breadth-first search
│   └── pagerank.h        # PageRank
└── neural/                # NEW: Neural net primitives
    ├── matmul.h          # Matrix multiply
    ├── softmax.h         # Softmax
    └── layer_norm.h      # Layer normalization
```
