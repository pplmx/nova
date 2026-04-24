# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **v1.4 Multi-Node Support**: Cluster-scale training infrastructure
  - MPI integration (MpiContext, rank discovery)
  - Topology detection (NIC enumeration, RDMA capability)
  - Cross-node communicators (MultiNodeContext, HierarchicalAllReduce)
- **Build System**: Multi-core optimization and Ninja support
  - Auto-detect CPU cores for parallel builds
  - Ninja generator support for faster builds
  - Unity build for compilation speedup
  - CPU-aware test parallelism (capped at 16 for GPU memory)

- **Five-Layer Architecture**: Production-ready layered architecture
  - Layer 0: `cuda::memory` - Buffer<T>, unique_ptr<T>, MemoryPool, Allocator concepts
  - Layer 1: `cuda::device` - Pure device kernels, CUDA_CHECK, ReduceOp, warp_reduce
  - Layer 2: `cuda::algo` - Algorithm wrappers, DeviceBuffer aliases
  - Layer 3: `cuda::api` - DeviceVector, Stream, Event, Config objects
- **Memory Pool**: Efficient device memory allocation with caching
- **Stream/Event API**: RAII wrappers for CUDA streams and events
- **Configuration Objects**: ReduceConfig, ScanConfig, SortConfig, MatrixMultConfig

### Changed

- **CMake**: Modernized with INTERFACE libraries (cuda_memory, cuda_device, cuda_algo, cuda_api)
- **CMake**: Auto-detect CPU cores, Unity build, Ninja generator support
- **Namespace Renaming**: `cuda::kernel` → `cuda::device`
- **Directory Structure**: All CUDA headers in `include/cuda/` subdirectories

### Removed

- **Backward Compatibility**: Removed `include/cuda/kernel/` forwarding headers
- **Legacy Aliases**: All code now uses new architecture directly

### Documentation

- Updated README with five-layer architecture diagram
- Updated architecture specifications in docs/superpowers/specs/
- Added Layer 0 (memory) documentation

## [0.1.0] - 2025-04-22

### Added

- **Layered Architecture**: Three-layer architecture (kernel/algo/api)
  - Layer 1: Pure CUDA kernels with no memory management
  - Layer 2: Algorithm wrappers with DeviceBuffer RAII memory
  - Layer 3: STL-style DeviceVector container
- **DeviceBuffer**: RAII wrapper for CUDA device memory management
- **DeviceVector**: High-level STL-style container for device memory
- **Image Processing Module**: types, brightness, gaussian_blur, sobel_edge
- **Parallel Primitives Module**: scan (prefix sum), sort (odd-even, bitonic)
- **Matrix Operations Module**: add, mult (naive/tiled/cuBLAS)
- **AGENTS.md**: Agent guidelines for development workflow

### Changed

- **CMake**: Modernized with INTERFACE libraries (cuda_kernel, cuda_algo, cuda_api)
- **Directory Structure**: Organized by functional modules (image/, parallel/, matrix/)
- **API Namespaces**: New API uses `cuda::kernel`, `cuda::algo`, `cuda::api`
- **API**: Removed backward compatibility forwarding headers

### Deprecated

- Legacy flat include structure (use module paths instead)

### Removed

- **my_lib**: Example utility files deleted
- **Forwarding headers**: Removed for cleaner API

### Documentation

- Updated README with complete directory structure and module descriptions
- Added implementation plan to docs/superpowers/
- Added design specification to docs/superpowers/
