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
- **Forwarding Headers**: Backward compatibility for existing code
- **AGENTS.md**: Agent guidelines for development workflow

### Changed

- **CMake**: Modernized with INTERFACE libraries (cuda_kernel, cuda_algo, cuda_api)
- **Directory Structure**: Flat structure reorganized into layered architecture
- **API Namespaces**: New API uses `cuda::kernel`, `cuda::algo`, `cuda::api`

### Fixed

- Test linking to proper layered libraries
- Pre-existing CUBLAS_CHECK macro issue

### Documentation

- Updated README with architecture diagrams and usage examples
- Added implementation plan to docs/superpowers/
- Added design specification to docs/superpowers/
