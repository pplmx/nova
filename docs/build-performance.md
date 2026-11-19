# Build Performance Guide

This guide helps you optimize Nova's build time using CMake presets, ccache, and unity builds.

## Quick Start

The fastest way to build Nova:

```bash
cmake --preset dev
cmake --build --preset dev --parallel
```

## CMake Presets

Nova includes CMake presets for common build configurations:

| Preset    | Build Type | Purpose                           |
| --------- | ---------- | --------------------------------- |
| `dev`     | Debug      | Development with ccache enabled   |
| `release` | Release    | Optimized build with unity builds |
| `ci`      | Release    | CI-optimized, no ccache           |

### Using Presets

```bash
# Configure with a preset
cmake --preset dev

# Build with a preset
cmake --build --preset dev

# Run tests with a preset
ctest --preset dev
```

### Customizing Presets

Edit `CMakePresets.json` to add your own presets or modify existing ones.

## ccache

ccache is a compiler cache that speeds up rebuilds by caching previous compilations.

### Installation

```bash
# Ubuntu/Debian
sudo apt install ccache

# macOS
brew install ccache

# Verify installation
ccache --version
```

### Enabling ccache

#### Option 1: CMake Preset (recommended)

```bash
cmake --preset dev  # dev preset has ccache enabled by default
```

#### Option 2: Manual configuration

```bash
cmake -B build -DNOVA_USE_CCACHE=ON
cmake --build build
```

### Verifying ccache is Working

```bash
# After building, check cache stats
ccache -s

# Example output:
# cache size: 1.2 GB
# hits: 156
# misses: 42
# hit rate: 78.8%
```

### Expected Performance

- First build: Same as normal (cache population)
- Subsequent builds: 50-80% faster
- Clean builds: Same as normal (cache miss)

## Unity Builds

Unity builds combine multiple source files into single compilation units, significantly reducing build time.

### Enabling Unity Builds

#### Option 1: CMake Preset (recommended)

```bash
cmake --preset release  # release preset has unity builds enabled
```

#### Option 2: Manual configuration

```bash
cmake -B build -DNOVA_ENABLE_UNITY_BUILD=ON
cmake --build build
```

### Unity Build Batch Size

Nova automatically adjusts batch size based on CPU cores:

| CPU Cores | Batch Size  |
| --------- | ----------- |
| 64+       | 64          |
| 32+       | 32          |
| <32       | 4 (default) |

### Known Limitations

Unity builds may cause:

- Longer link times
- Reduced parallel compilation
- Occasional symbol collisions (rare)

If you encounter issues, disable unity builds:

```bash
cmake -B build -DNOVA_ENABLE_UNITY_BUILD=OFF
```

## Ninja Generator

Ninja is faster than Make for most builds.

### Using Ninja

```bash
cmake -G Ninja -B build
cmake --build build
```

Or use the preset:

```bash
cmake --preset dev -G Ninja
```

## Build Performance Tips

1. **Use ccache** - 50-80% faster rebuilds
2. **Use Ninja** - Better parallelization than Make
3. **Use unity builds** - 20-40% faster compilation
4. **Use presets** - Consistent, reproducible builds
5. **Parallel builds** - Use `-j$(nproc)` for maximum parallelism

## Benchmark Results

Typical build times on a 32-core machine:

| Configuration            | Initial Build | Rebuild |
| ------------------------ | ------------- | ------- |
| Release (no cache)       | 5 min         | 5 min   |
| Release (ccache)         | 5 min         | 1 min   |
| Release (unity)          | 4 min         | 4 min   |
| Release (ccache + unity) | 4 min         | 45 sec  |

## CI Integration

For CI systems, use the `ci` preset:

```bash
cmake --preset ci
cmake --build --preset ci
ctest --preset ci
```

The `ci` preset:

- Disables ccache (unnecessary in CI)
- Enables unity builds
- Fails on missing tests

## Troubleshooting

### ccache not found

```bash
# Install ccache
sudo apt install ccache

# Or set NOVA_USE_CCACHE=OFF
cmake -B build -DNOVA_USE_CCACHE=OFF
```

### Low ccache hit rate

Check for:

- Different compiler flags between builds
- Missing `CCACHE_BASEDIR` environment variable
- Source file modifications

```bash
export CCACHE_BASEDIR=$PWD
ccache -C  # Clear and restart
```

### Unity build errors

```bash
# Disable unity builds
cmake -B build -DNOVA_ENABLE_UNITY_BUILD=OFF
cmake --build build
```

## Further Reading

- [ccache documentation](https://ccache.dev/documentation.html)
- [CMake Unity Builds](https://cmake.org/cmake/help/latest/prop_tgt/UNITY_BUILD.html)
- [Ninja documentation](https://ninja-build.org/manual.html)
