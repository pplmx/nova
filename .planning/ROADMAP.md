# Nova v1.9 Documentation — Roadmap

**Milestone:** v1.9 Documentation
**Created:** 2026-04-26
**Status:** Planning
**Granularity:** Standard (3 phases)

## Overview

This milestone adds comprehensive documentation to Nova:

1. **API Reference** — Doxygen/Sphinx auto-generated documentation
2. **Tutorials** — Step-by-step guides for common use cases
3. **Examples** — Sample code demonstrating key features

## Phases

- [ ] **Phase 37: API Reference** — Doxygen configuration and documentation generation
- [ ] **Phase 38: Tutorials** — Quick start, multi-GPU, checkpoint, profiling guides
- [ ] **Phase 39: Examples** — Image processing, graph, neural net, distributed examples

---

## Phase Details

### Phase 37: API Reference

**Goal:** Generate comprehensive API documentation from source code

**Depends on:** Nothing (first phase)

**Requirements:** API-01, API-02, API-03, API-04

**Success Criteria** (what must be TRUE):

1. Developer can run `doxygen` and generate HTML documentation
2. All public functions in headers have Doxygen comments
3. Documentation is grouped by module (memory, device, algo, api)
4. Cross-references link related functions and types

**Plans:** TBD

### Phase 38: Tutorials

**Goal:** Provide step-by-step guides for common use cases

**Depends on:** Phase 37 (API reference provides context)

**Requirements:** TUT-01, TUT-02, TUT-03, TUT-04

**Success Criteria** (what must be TRUE):

1. Developer can complete quick start in 5 minutes and run first CUDA program
2. Developer can implement multi-GPU data parallelism following tutorial
3. Developer can save and restore checkpoint following tutorial
4. Developer can profile application using benchmarks following guide

**Plans:** TBD

### Phase 39: Examples

**Goal:** Provide runnable examples demonstrating key features

**Depends on:** Phase 37 (API reference for implementation details)

**Requirements:** EX-01, EX-02, EX-03, EX-04

**Success Criteria** (what must be TRUE):

1. Developer can run image processing example with blur/sobel/morphology
2. Developer can run graph algorithm example (BFS or PageRank)
3. Developer can run neural net primitives example (matmul, softmax)
4. Developer can run distributed training example with NCCL

**Plans:** TBD

---

## Coverage

| Requirement | Phase | Description |
|-------------|-------|-------------|
| API-01 | Phase 37 | Doxygen configuration generates HTML documentation |
| API-02 | Phase 37 | All public headers have documented function signatures |
| API-03 | Phase 37 | Grouped documentation by module |
| API-04 | Phase 37 | Cross-references link related functions and types |
| TUT-01 | Phase 38 | Quick start guide (5-minute to first CUDA program) |
| TUT-02 | Phase 38 | Multi-GPU tutorial with device mesh example |
| TUT-03 | Phase 38 | Checkpoint and restore tutorial |
| TUT-04 | Phase 38 | Performance profiling guide using benchmarks |
| EX-01 | Phase 39 | Image processing example with blur/sobel/morphology |
| EX-02 | Phase 39 | Graph algorithm example (BFS/PageRank) |
| EX-03 | Phase 39 | Neural net primitives example (matmul, softmax) |
| EX-04 | Phase 39 | Distributed training example with NCCL |

**Coverage:** 12/12 requirements mapped

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 37. API Reference | 0/4 | Not started | — |
| 38. Tutorials | 0/4 | Not started | — |
| 39. Examples | 0/4 | Not started | — |

---

## Previous Milestone

**v1.8 Developer Experience** — SHIPPED 2026-04-26

- Phase 33: Error Message Framework
- Phase 34: CMake Package Export
- Phase 35: IDE Configuration
- Phase 36: Build Performance

---

*Roadmap created: 2026-04-26*
*Next: /gsd-plan-phase 37*
