# Phase 34: CMake Package Export

**Milestone:** v1.8 Developer Experience
**Status:** Discussing approach

## Goal

Downstream projects can discover and link Nova via CMake

## Requirements

| ID | Description |
|----|-------------|
| CMK-01 | find_package(nova REQUIRED) support |
| CMK-02 | Relocatable exported targets with generator expressions |
| CMK-03 | Feature matrix display during CMake configure |
| CMK-04 | Version file matching installed version |

## Success Criteria

1. Developer can run `find_package(nova REQUIRED)` successfully after running `cmake --install`
2. Developer can link `Nova::nova` and `Nova::cuda` imported targets in their CMake project
3. Developer sees feature matrix output during CMake configure showing NCCL, MPI status
4. Developer can relocate installed package to different directory without breaking find_package

## Existing Build System

### Current State

- CMake 4.0+ with CUDA 20, C++23
- CMAKE_EXPORT_COMPILE_COMMANDS is ON (line 53)
- No `install(EXPORT)` currently
- Uses FindCUDAToolkit for `CUDA::*` imported targets

### What Needs to be Added

1. **Config file template** (`cmake/NovaConfig.cmake.in`)
2. **Install export rules** in CMakeLists.txt
3. **Version file** via `write_basic_package_version_file()`
4. **Feature matrix** in ConfigOutputOptions

## Implementation Approach

Based on research, use modern CMake config-file package pattern:

```cmake
# In NovaConfig.cmake.in
@PACKAGE_INIT@

include("${CMAKE_CURRENT_LIST_DIR}/NovaTargets.cmake")

# Feature detection
check_required_components(nova)

# Print feature matrix
if(NOVA_NCCL_ENABLED)
    message(STATUS "Nova: NCCL [ON]")
endif()
```

Install rules:
```cmake
install(EXPORT NovaTargets 
        FILE NovaTargets.cmake
        NAMESPACE Nova::
        DESTINATION lib/cmake/nova)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NovaConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/NovaConfigVersion.cmake
        DESTINATION lib/cmake/nova)
```

## Key Decisions Needed

1. Installation directory structure (lib vs lib/cmake/nova)
2. Whether to install headers separately
3. How to expose optional components (NCCL, MPI)

---
*Context created: 2026-04-26*
*Ready for planning*
