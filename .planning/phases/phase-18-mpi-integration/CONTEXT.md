# Phase 18: MPI Integration

**Phase:** 18 of v1.4 Multi-Node Support
**Started:** 2026-04-24
**Goal:** Set up MPI integration for multi-node NCCL bootstrapping and rank discovery.

## Context

- Existing NCCL single-node implementation (v1.3)
- CMake build system with optional dependency pattern (NCCL)
- Need MPI for cluster-scale initialization

## Requirements

- MULN-01: MPI library detection and version validation via CMake find module
- MULN-02: MpiContext with rank/node discovery and NCCL bootstrapping
- MULN-03: MPI init/finalize lifecycle management with RAII semantics
- MULN-04: Cross-node device assignment (local_rank calculation)
- MULN-05: Environment variable and config file options for MPI parameters

## Success Criteria

1. CMake detects OpenMPI or MPICH when available
2. MpiContext provides world_rank, world_size, local_rank, local_size
3. RAII ensures MPI_Finalize is called on scope exit
4. Local GPU assignment matches CUDA_VISIBLE_DEVICES ordering
5. Config options override MPI environment variables

## Pitfalls

- MPI version mismatches (OpenMPI 4.x vs MPICH 3.x)
- Missing MPI when multi-node not needed (optional dependency)
- Device assignment race conditions with multiple processes
