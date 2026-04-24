# Phase 13: NCCL Foundation - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Set up NCCL integration infrastructure with proper error handling. Deliverables:
- NCCL library detection and CMake integration
- NcclContext with dependency injection pattern
- Communicator initialization and lifecycle management
- Async error polling infrastructure

This is foundation for Phase 14 (Core Collectives) which uses these primitives.

</domain>

<decisions>
## Implementation Decisions

### Context Management
- **D-01:** Dependency injection with singleton fallback — Primary: pass NcclContext as parameter for testability; convenience: `NcclContext::instance()` available for simple cases

### Error Handling
- **D-02:** Wrapper with automatic polling — `safe_nccl_call()` wrapper that polls `ncclCommGetAsyncError()` after each collective operation and throws on error

### CMake Integration
- **D-03:** Optional with P2P fallback — CMake finds NCCL if available; code compiles with P2P fallback when NCCL unavailable; `NCCL_ENABLE` CMake option to control

### Communicator Lifecycle
- **D-04:** Per-device singleton caching — One NCCL communicator per device, cached in NcclContext, reused across operations for efficiency

</decisions>

<canonical_refs>
## Canonical References

### Architecture Research
- `.planning/research/ARCHITECTURE.md` — NCCL integration architecture, dependency injection vs singleton analysis
- `.planning/research/STACK.md` — NCCL 2.25+ version requirements
- `.planning/research/PITFALLS.md` — Async error handling pitfalls, NCCL initialization failures

### Requirements
- `.planning/REQUIREMENTS.md` — NCCL-01 through NCCL-05 requirements

### Existing Code
- `include/cuda/mesh/device_mesh.h` — DeviceMesh that NcclContext integrates with
- `include/cuda/distributed/reduce.h` — Existing distributed reduce to extend with NCCL backend
- `include/cuda/device/error.h` — Existing error handling patterns
- `include/cuda/async/stream_manager.h` — Stream management for async collectives

</canonical_refs>

<codebase_context>
## Existing Code Insights

### Reusable Assets
- DeviceMesh: Can extend to provide NCCL topology info
- StreamManager: Use for async collective scheduling
- Error handling: Existing patterns to follow for NCCL errors

### Established Patterns
- Error reporting via exceptions with descriptive messages
- Device-scoped operations (ScopedDevice pattern exists)
- Memory pool for buffer management

### Integration Points
- Extend DeviceMesh to create NcclContext
- Add NCCL backend to existing distributed operations
- Use existing stream infrastructure for async collectives

</codebase_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard NCCL library patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-nccl-foundation*
*Context gathered: 2026-04-24*
