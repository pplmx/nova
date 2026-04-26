# IDE Setup Guide for Nova

This guide helps you set up your IDE for Nova CUDA development.

## Prerequisites

- CUDA Toolkit 12.0+
- CMake 4.0+
- A C++ IDE (VS Code, CLion, or similar)

## VS Code Setup

### 1. Install Extensions

Install these VS Code extensions:

- **clangd** (llvm-vs-code-extensions.vscode-clangd) - Language server for C++
- **CMake Tools** (ms-vscode.cmake-tools) - CMake integration
- **C/C++** (ms-vscode.cpptools) - IntelliSense and debugging

### 2. Open Nova in VS Code

```bash
cd /path/to/nova
code .
```

### 3. Configure Build

1. Press `Ctrl+Shift+P` and run **CMake: Configure**
2. Select Ninja as the generator
3. Choose your build options:
   - `NOVA_ENABLE_NCCL` - Enable NCCL support (default: ON)
   - `NOVA_ENABLE_MPI` - Enable MPI support (default: OFF)

### 4. Build the Project

```bash
cmake --build build --parallel
```

This generates `compile_commands.json` which clangd uses for code intelligence.

### 5. Enable Code Intelligence

After building, clangd will automatically index your code using `compile_commands.json`.

## clangd Setup (Other Editors)

### 1. Install clangd

```bash
# Ubuntu/Debian
sudo apt install clangd

# macOS
brew install clangd

# Or download from https://clangd.llvm.org/
```

### 2. Configure clangd

Copy `.clangd/config.yaml` from this repository to your project root.

Key settings:
- Uses `-xcuda` for CUDA file parsing
- Targets `sm_80` (Ampere) architecture
- Filters out nvcc-specific flags that clangd can't understand

### 3. Generate compile_commands.json

```bash
cmake -G Ninja -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
```

## CLion Setup

1. Open Nova as a CMake project
2. Go to **File > Settings > Build, Execution, Deployment > CMake**
3. Set Generator: **Ninja**
4. Add CMake options:
   ```
   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   ```
5. Apply and reload CMake

## Troubleshooting

### clangd shows errors for CUDA files

Ensure `.clangd/config.yaml` is present and you've built the project to generate `compile_commands.json`.

### IntelliSense not working

Check that `compile_commands.json` exists at the project root:

```bash
ls -la compile_commands.json
```

If it doesn't exist, rebuild the project:

```bash
cmake --build build
```

### Missing includes

Verify the CUDA include path in `.vscode/c_cpp_properties.json` or `.clangd/config.yaml` matches your CUDA installation.

## Quick Start Commands

```bash
# Configure with compile_commands.json export
cmake -G Ninja -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
cmake --build build --parallel

# Run tests
ctest --test-dir build
```

## Further Reading

- [clangd documentation](https://clangd.llvm.org/)
- [CMake Tools documentation](https://github.com/microsoft/vscode-cmake-tools)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide/)
