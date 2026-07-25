# ADR-001: Five-Layer Architecture

**Date:** 2026-07-25
**Status:** Accepted
**Context version:** v2.11

## Context

Nova is a CUDA library spanning multiple concerns: memory management,
raw kernel execution, algorithm orchestration, high-level STL-style
containers, and domain-specific operations (sparse, quantize, GNN).

A flat or single-layer design would mix these concerns, making it hard
to:
- Swap memory allocation strategies without touching kernel code
- Test kernels in isolation
- Expose a clean API to external consumers
- Add new domain modules without affecting existing code

## Decision

Adopt a five-layer architecture:

| Layer | Namespace | Responsibility |
|-------|-----------|----------------|
| 0 | `cuda::memory` | RAII buffers, memory pools, distributed pools |
| 1 | `cuda::device` | Pure CUDA kernels, warp primitives |
| 2 | `cuda::algo` | Algorithm orchestration, device memory management |
| 3 | `cuda::api` | STL-style containers (DeviceVector, Stream) |
| 4 | `cuda::sparse/quantize/gnn` | Domain-specific operations |

Dependencies flow strictly upward: Layer N may only depend on Layers
0..N-1. No layer may depend on a higher layer.

## Consequences

- **Positive**: Each layer can be tested and benchmarked independently.
  Kernel correctness tests live at layer 1 without layer 3
  infrastructure. Memory pool strategies can be swapped at layer 0
  without touching algorithm code.
- **Positive**: New domain modules (layer 4) can be added without
  modifying lower layers — only layer 3 API surface needs to be
  sufficient.
- **Negative**: Five layers impose ceremony. A simple reduce operation
  touches layers 0→1→2→3. For internal tools, this overhead is
  acceptable; for external consumers, layer 3 is the recommended entry
  point.

## Alternatives considered

- **Flat namespace**: Rejected — would mix kernel code, memory
  management, and application logic in a single namespace, making
  dependency boundaries invisible.
- **Two-layer (core + app)**: Rejected — insufficient granularity.
  Memory pools and kernels have different stability guarantees and
  should not share a layer.
- **Plugin-based**: Rejected as over-engineered for a library that is
  compiled into the consuming binary. Layers provide compile-time
  enforcement without runtime plugin overhead.

## See also

- `docs/ARCHITECTURE.md` — detailed layer descriptions and directory structure
- `docs/PRODUCTION.md` — how layer 4 modules integrate with production hardening
