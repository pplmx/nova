# Nova v1.9 Documentation — Requirements

**Milestone:** v1.9 Documentation
**Status:** Draft

## Overview

This milestone adds comprehensive documentation to Nova:

1. **API Reference** — Auto-generated documentation from source code
2. **Tutorials** — Step-by-step guides for common use cases
3. **Examples** — Sample code demonstrating key features

## Requirements

### Phase 37: API Reference

- [ ] **API-01**: Doxygen configuration generates HTML documentation from source code
- [ ] **API-02**: All public headers have documented function signatures
- [ ] **API-03**: Grouped documentation by module (memory, device, algo, api)
- [ ] **API-04**: Cross-references link related functions and types

### Phase 38: Tutorials

- [ ] **TUT-01**: Quick start guide (5-minute to first CUDA program)
- [ ] **TUT-02**: Multi-GPU tutorial with device mesh example
- [ ] **TUT-03**: Checkpoint and restore tutorial
- [ ] **TUT-04**: Performance profiling guide using benchmarks

### Phase 39: Examples

- [ ] **EX-01**: Image processing example with blur/sobel/morphology
- [ ] **EX-02**: Graph algorithm example (BFS/PageRank)
- [ ] **EX-03**: Neural net primitives example (matmul, softmax)
- [ ] **EX-04**: Distributed training example with NCCL

## Future Requirements (Deferred)

- Interactive API playground
- Video tutorials
- Community-contributed examples

## Out of Scope

| Feature | Reason |
|---------|--------|
| Python documentation | Separate project |
| Video content | Requires different tooling |
| Translated documentation | English-only for v1.9 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| API-01 to API-04 | Phase 37 | — |
| TUT-01 to TUT-04 | Phase 38 | — |
| EX-01 to EX-04 | Phase 39 | — |

---

*Requirements defined: 2026-04-26*
*Total: 12 requirements (API-01..API-04, TUT-01..TUT-04, EX-01..EX-04)*
