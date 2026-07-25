# Architecture Decision Records (ADRs)

This directory contains ADRs documenting significant architectural
decisions in Nova.

## Index

### Foundational

- [ADR-001](ADR-001-five-layer-architecture.md) — Five-layer architecture
- [ADR-002](ADR-002-production-hardening.md) — Production hardening: circuit breaker, retry, graceful degradation
- [ADR-003](ADR-003-quantization-dual-strategy.md) — INT8 + FP8 dual quantization strategy

## Conventions

Each ADR follows the format:

```
# ADR-NNN: Title

**Date:** YYYY-MM-DD
**Status:** Accepted | Superseded | Deprecated
**Context version:** vX.Y

## Context
## Decision
## Consequences
## Alternatives considered
## See also
```

ADRs are **immutable once accepted**. Superseding decisions write a new
ADR that references the old one.

## Adding a new ADR

1. Pick the next available number
2. Copy an existing ADR as a template
3. Fill in all sections
4. Add an entry to this index
5. Commit: `docs(adr): ADR-NNN <topic>`

## See also

- `docs/ARCHITECTURE.md` — high-level architecture overview
- `docs/PRODUCTION.md` — production hardening guide
- `docs/QUANTIZATION.md` — quantization guide
