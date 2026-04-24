# Requirements: Nova CUDA Library Enhancement

**Defined:** 2026-04-24
**Milestone:** v1.4 Multi-Node Support
**Core Value:** Enable efficient multi-node training with MPI-based NCCL initialization, topology-aware collective selection, and cross-node communicator management.

## v1.4 Requirements

Requirements for v1.4 milestone. Each maps to roadmap phases.

### MPI Integration (Phase 1)

- [ ] **MULN-01**: MPI library detection and version validation via CMake find module
- [ ] **MULN-02**: MpiContext with rank/node discovery and NCCL bootstrapping
- [ ] **MULN-03**: MPI init/finalize lifecycle management with RAII semantics
- [ ] **MULN-04**: Cross-node device assignment (local_rank calculation)
- [ ] **MULN-05**: Environment variable and config file options for MPI parameters

### Topology-Aware Collectives (Phase 2)

- [ ] **TOPO-01**: Node topology detection (intra-node vs inter-node paths)
- [ ] **TOPO-02**: Network interface card (NIC) enumeration and selection
- [ ] **TOPO-03**: Topology-aware NCCL communicator splitting by NIC
- [ ] **TOPO-04**: Bandwidth-aware collective algorithm selection (ring vs tree vs collnet)
- [ ] **TOPO-05**: Topology validation with performance profiling

### Cross-Node Communicators (Phase 3)

- [ ] **CNOD-01**: MultiNodeContext extending NcclContext for cluster scale
- [ ] **CNOD-02**: Intra-node NCCL communicator (per-node GPU groups)
- [ ] **CNOD-03**: Inter-node NCCL communicator (cross-node GPU groups)
- [ ] **CNOD-04**: Hierarchical collectives (node-local then cross-node)
- [ ] **CNOD-05**: Graceful degradation when MPI/NCCL-NET unavailable

## v2 Requirements

Deferred to future release.

### Distributed Batch Normalization

- **DBN-01**: Cross-GPU batch statistics aggregation
- **DBN-02**: Distributed sync BatchNorm layer
- **DBN-03**: Memory-efficient distributed BatchNorm with tensor parallelism

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| MPI collective algorithms | NCCL handles collective optimization; MPI only for init |
| RDMA/InfiniBand specifics | Abstraction via NCCL-NET; platform-specific tuning separate |
| Kubernetes/job scheduler integration | Separate project (nova-cluster) |
| Fault tolerance/recovery | v1.5 scope |
| Python bindings | Separate project |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| MULN-01 | Phase 1 | Pending |
| MULN-02 | Phase 1 | Pending |
| MULN-03 | Phase 1 | Pending |
| MULN-04 | Phase 1 | Pending |
| MULN-05 | Phase 1 | Pending |
| TOPO-01 | Phase 2 | Pending |
| TOPO-02 | Phase 2 | Pending |
| TOPO-03 | Phase 2 | Pending |
| TOPO-04 | Phase 2 | Pending |
| TOPO-05 | Phase 2 | Pending |
| CNOD-01 | Phase 3 | Pending |
| CNOD-02 | Phase 3 | Pending |
| CNOD-03 | Phase 3 | Pending |
| CNOD-04 | Phase 3 | Pending |
| CNOD-05 | Phase 3 | Pending |

**Coverage:**
- v1.4 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-04-24*
*Last updated: 2026-04-24 after v1.4 requirements definition*
