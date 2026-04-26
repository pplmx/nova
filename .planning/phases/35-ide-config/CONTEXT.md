# Phase 35: IDE Configuration

**Milestone:** v1.8 Developer Experience
**Status:** Planning

## Goal

Developers can use clangd or VS Code with full CUDA support

## Requirements

| ID | Description |
|----|-------------|
| IDE-01 | .clangd/config.yaml for clangd CUDA parsing |
| IDE-02 | .vscode/settings.json for VS Code clangd integration |
| IDE-03 | compile_commands.json symlink at project root |
| IDE-04 | docs/ide-setup.md documentation |

## Success Criteria

1. Developer can open Nova in any editor with clangd and see zero spurious errors for `.cu` files
2. Developer using VS Code sees clangd integration working with real-time diagnostics
3. Developer finds `compile_commands.json` symlinked at project root
4. Developer can follow `docs/ide-setup.md` to configure their IDE in under 5 minutes

## Implementation

### 1. `.clangd/config.yaml`

```yaml
CompileFlags:
  Add:
    - "-xcuda"
    - "--cuda-gpu-arch=sm_80"
    - "-I/usr/local/cuda/include"
  Remove:
    - "-gencode*"
    - "--threads*"
Diagnostics:
  Clang: Off
  Unused: Off
```

### 2. `.vscode/settings.json`

```json
{
  "clangd.arguments": [
    "--background-index",
    "--compile-commands-dir=${workspaceFolder}/build"
  ],
  "clangd.path": "clangd"
}
```

### 3. Symlink

```bash
ln -sf build/compile_commands.json compile_commands.json
```

Add to CMakeLists.txt post-install step or use cmake link script.

### 4. `docs/ide-setup.md`

Quick guide for:
- Installing clangd
- VS Code clangd extension
- CMake Tools configuration
- Building and indexing

---
*Context created: 2026-04-26*
