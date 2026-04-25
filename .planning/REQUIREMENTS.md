# Requirements: Nova CUDA Library Enhancement

**Defined:** 2026-04-26
**Milestone:** v1.5 Fault Tolerance
**Core Value:** Enable production-grade fault tolerance with checkpoint/restart, error recovery, and graceful handling of cluster preemption.

## v1.5 Requirements

Requirements for v1.5 milestone. Each maps to roadmap phases.

### Checkpoint/Restart (Phase 1)

- [ ] **CKPT-01**: CheckpointManager with async writes and configurable interval
- [ ] **CKPT-02**: Full state serialization (weights, optimizer states, RNG state)
- [ ] **CKPT-03**: Storage backend abstraction (filesystem, object store paths)
- [ ] **CKPT-04**: Incremental checkpoint support for reduced I/O overhead
- [ ] **CKPT-05**: Automatic checkpoint on error detection before recovery

### Communication Error Recovery (Phase 2)

- [ ] **COMM-01**: NCCL timeout detection with configurable thresholds
- [ ] **COMM-02**: Health monitoring for collective operations (watchdog thread)
- [ ] **COMM-03**: Automatic retry with exponential backoff for transient errors
- [ ] **COMM-04**: Cross-node connection repair and communicator recreation
- [ ] **COMM-05**: Error classification (transient vs permanent) for retry decisions

### Memory Error Detection (Phase 3)

- [ ] **MEM-01**: CUDA error detection and classification via cudaDeviceGetErrorString
- [ ] **MEM-02**: ECC error callback registration and handling infrastructure
- [ ] **MEM-03**: Device health monitoring with periodic checks during idle
- [ ] **MEM-04**: Graceful degradation strategies (reduce TP degree, fall back to CPU)
- [ ] **MEM-05**: Memory error telemetry and logging for diagnostics

### Job Preemption Handling (Phase 4)

- [ ] **PEMP-01**: Signal handlers for SIGTERM/SIGUSR1 with graceful shutdown
- [ ] **PEMP-02**: Training state preservation sequence on preemption
- [ ] **PEMP-03**: Resume-from-checkpoint validation and recovery
- [ ] **PEMP-04**: Configurable shutdown timeout (default 30s, extendable)
- [ ] **PEMP-05**: Coordinated checkpoint across multi-node ranks

## v1.4 Requirements

Previously completed.

### MPI Integration

- [x] **MULN-01**: MPI library detection and version validation via CMake find module
- [x] **MULN-02**: MpiContext with rank/node discovery and NCCL bootstrapping
- [x] **MULN-03**: MPI init/finalize lifecycle management with RAII semantics
- [x] **MULN-04**: Cross-node device assignment (local_rank calculation)
- [x] **MULN-05**: Environment variable and config file options for MPI parameters

### Topology-Aware Collectives

- [x] **TOPO-01**: Node topology detection (intra-node vs inter-node paths)
- [x] **TOPO-02**: Network interface card (NIC) enumeration and selection
- [x] **TOPO-03**: Topology-aware NCCL communicator splitting by NIC
- [x] **TOPO-04**: Bandwidth-aware collective algorithm selection (ring vs tree vs collnet)
- [x] **TOPO-05**: Topology validation with performance profiling

### Cross-Node Communicators

- [x] **CNOD-01**: MultiNodeContext extending NcclContext for cluster scale
- [x] **CNOD-02**: Intra-node NCCL communicator (per-node GPU groups)
- [x] **CNOD-03**: Inter-node NCCL communicator (cross-node GPU groups)
- [x] **CNOD-04**: Hierarchical collectives (node-local then cross-node)
- [x] **CNOD-05**: Graceful degradation when MPI/NCCL-NET unavailable

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CKPT-01 | Phase 1 | Pending |
| CKPT-02 | Phase 1 | Pending |
| CKPT-03 | Phase 1 | Pending |
| CKPT-04 | Phase 1 | Pending |
| CKPT-05 | Phase 1 | Pending |
| COMM-01 | Phase 2 | Pending |
| COMM-02 | Phase 2 | Pending |
| COMM-03 | Phase 2 | Pending |
| COMM-04 | Phase 2 | Pending |
| COMM-05 | Phase 2 | Pending |
| MEM-01 | Phase 3 | Pending |
| MEM-02 | Phase 3 | Pending |
| MEM-03 | Phase 3 | Pending |
| MEM-04 | Phase 3 | Pending |
| MEM-05 | Phase 3 | Pending |
| PEMP-01 | Phase 4 | Pending |
| PEMP-02 | Phase 4 | Pending |
| PEMP-03 | Phase 4 | Pending |
| PEMP-04 | Phase 4 | Pending |
| PEMP-05 | Phase 4 | Pending |

**Coverage:**
- v1.5 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0

---
*Requirements defined: 2026-04-26*
*Last updated: 2026-04-26 after v1.5 requirements definition*
