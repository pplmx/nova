# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
