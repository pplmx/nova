# ADR-002: Production Hardening Strategy

**Date:** 2026-07-25
**Status:** Accepted
**Context version:** v2.11

## Context

CUDA operations can fail for reasons beyond application logic: GPU OOM,
driver timeouts, transient hardware errors, thermal throttling, and
peer-to-peer communication failures in multi-GPU setups. A library that
only propagates `cudaError_t` to callers forces every consumer to
implement their own retry, timeout, and degradation logic.

Nova targets education, research, AND production use. Production users
need these resilience patterns built in — not bolted on per-deployment.

## Decision

Include three production-hardening primitives as first-class library
features:

1. **Timeout management** (`cuda::error::timeout_manager`):
   RAII `timeout_guard` with background watchdog thread. Configurable
   per-operation timeouts with callbacks.

2. **Retry with exponential backoff** (`cuda::error::retry_executor`):
   Configurable base delay, multiplier, max delay, max attempts, and
   jitter. Pluggable into any callable.

3. **Circuit breaker** (`cuda::error::circuit_breaker`):
   Failure threshold → open → half-open probe → close. Per-operation
   type configuration. Prevents cascading GPU failures.

4. **Graceful degradation** (`cuda::error::degradation_manager`):
   Precision levels (high/medium/low). Quality thresholds with
   auto-degradation triggers (e.g., memory pressure).

All primitives use `std::error_code` for error propagation, not
exceptions, to avoid the exception-safety challenges of CUDA host code.

## Consequences

- **Positive**: Production users get resilience without writing their
  own infrastructure. Research users can ignore these features
  (zero-cost when not used).
- **Positive**: The circuit breaker prevents a single GPU failure from
  taking down an entire multi-GPU pipeline.
- **Negative**: The timeout watchdog thread adds a background thread
  even when timeouts are not configured. Mitigated by lazy
  initialization — the thread starts only on first `timeout_guard`
  construction.
- **Negative**: Retry with jitter is not free — each retry pays
  `cudaMalloc`/`cudaMemcpy` cost for the operation being retried.
  Callers must set `max_attempts` appropriately.

## Alternatives considered

- **Delegate to caller**: Rejected — every consumer reimplements
  slightly different retry/timeout logic, leading to inconsistent
  behavior and duplicated bugs.
- **Exception-based error handling**: Rejected — CUDA host code and
  exceptions have well-known pitfalls (device code cannot throw,
  `__host__ __device__` functions need dual error paths).
- **External observability tool (e.g., only NVBlox metrics)**:
  Rejected — metrics tell you what failed, but resilience primitives
  prevent the failure from propagating.

## See also

- `docs/PRODUCTION.md` — usage guide and configuration defaults
- ADR-001 — five-layer architecture (error handling lives at layer 2)
