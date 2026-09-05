#pragma once

/**
 * @file nccl_recovery.h
 * @brief Comm-poisoning seam for collectives that launch NCCL directly
 *
 * The comm-error layer (safe_nccl_call / safe_stream_wait) aborts a dead
 * communicator and notifies its owning NcclContext via mark_comm_aborted, so
 * has_nccl() goes false and later collectives fail fast instead of hanging —
 * and the owning context's re-initialize() path self-heals (v2.18 P3 /
 * TASK-003). Header-only code that launches NCCL directly instead of going
 * through that layer (sequence-parallel collectives, the ring KV P2P exchange)
 * must route its failures through the same seam, or a broken communicator is
 * silently reused and every later NCCL user in the process hangs.
 */

/* NcclContext is intentionally left as an opaque handle (void*): header-only
   callers only need to hand the owner pointer back on failure; they never need
   the full type. The definition lives in src/cuda/nccl/nccl_context.cpp. */

namespace cuda::nccl {

/**
 * @brief Abort a broken communicator and flag it dead on its owning context
 *
 * Mirrors what safe_nccl_call does on an async error / timeout: abort the comm
 * first (the error layer's cleanup), then notify the owning NcclContext so
 * has_nccl() flips false and subsequent collectives fail fast instead of
 * silently reusing a dead communicator and hanging. The self-healing recovery
 * path (initialize() on the owning context) then re-establishes NCCL.
 *
 * @param comm         The broken NCCL communicator (null is a no-op).
 * @param nccl_context Opaque pointer to the owning NcclContext (may be null:
 *                     the comm is then aborted but no owner is notified, so
 *                     has_nccl() stays untouched).
 *
 * @note noexcept: this runs on error paths where there is nothing left to do
 *       but clean up and propagate; it must never throw into a half-torn-down
 *       collective.
 */
void poison_failed_comm(void* comm, void* nccl_context) noexcept;

/**
 * @brief Ask whether an owning NcclContext is still healthy (has_nccl() true)
 *
 * The counterpart to poison_failed_comm: before calling NCCL through a
 * caller-supplied communicator, consult the owning context if one was wired.
 * After a prior collective poisoned the context (a broken comm -> has_nccl()
 * false), the caller's `comm` points at a dead communicator — using it is
 * undefined (observed to SEGV). This lets header-only callers fail fast and
 * route into the self-healing re-initialize() path instead.
 *
 * @param nccl_context Opaque pointer to the owning NcclContext (may be null).
 * @return true when the context is healthy (or no context was wired — the
 *         caller owns the comm lifecycle then); false when the wired context
 *         reports has_nccl() false.
 */
bool comm_context_healthy(void* nccl_context) noexcept;

}  // namespace cuda::nccl
