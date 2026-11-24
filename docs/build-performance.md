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

| Preset    | Build Type | Purpose                                     |
| --------- | ---------- | ------------------------------------------- |
| `dev`     | Debug      | Development with unity builds disabled      |
| `release` | Release    | Optimized build with unity builds           |
| `ci`      | Release    | CI-optimized, unity builds                  |

Compiler caching and a faster linker are enabled by default for every preset
(see below), so the table only lists what differs between them.

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

## Compiler cache

A compiler cache stores the object files produced by previous compilations and
replays them when the same input comes back. Nova supports two, and picks
whichever is installed: **sccache** first, then **ccache**. Both are on by
default (`NOVA_USE_CCACHE=ON`); with neither installed the build is unchanged
and configure prints a line saying so.

sccache is preferred when both are present because it is the only one of the
two that can share its cache across machines, which is what makes it useful in
CI. Locally the difference does not matter.

### Installation

```bash
# Ubuntu/Debian
sudo apt install ccache

# macOS
brew install ccache
```

sccache ships with `cargo install sccache` or a release binary. Verify either
with `ccache --version` / `sccache --version`.

### Enabling it

It is already on. To turn it off:

```bash
cmake -B build -DNOVA_USE_CCACHE=OFF
```

### Verifying it is working

```bash
ccache -s          # or: sccache --show-stats
```

On a clean build tree the second build should show a high hit count. Measured
on the reference machine for the `nova` target alone: 76.8 s cold cache versus
30.6 s warm cache (759 hits), about 2.5x.

### Expected performance

- First build: same as normal (cache population)
- Subsequent builds after `ccache -C` or a fresh build directory: 2-3x faster
- Editing one file: only that file recompiles, as usual

## Faster linker

Link time is the part a compiler cache cannot help with: cached object files
still have to be linked. Nova links with **mold** when it is available,
falling back to **lld** and then to the toolchain default.
`CMAKE_LINKER_TYPE` is only set after `check_linker_flag` confirms the compiler
can actually drive that linker, so a missing or unsupported linker keeps the
default instead of failing at link time.

```bash
# Ubuntu/Debian
sudo apt install mold
```

Configure prints one of `Linker: mold`, `Linker: lld not usable by this
compiler, keeping the default`, or `Linker: default for this toolchain`.
Disable with `-DNOVA_USE_MOLD=OFF` and `-DNOVA_USE_LLD=OFF`.

On a CUDA build the device link still runs through nvcc; mold handles the host
link step, which is where a large static library costs the most.

### What it is worth here

Measured on the reference machine for the `nova` target: relinking with GNU ld
takes 224 ms, with mold 105 ms. That is a real halving of the link step, but
the link step is under 0.5% of a full build (50.8 s versus 49.5 s end to end,
about 2.5%), because Nova is compile-bound, not link-bound. mold is left on
because it costs nothing and helps more as the number of targets grows, but do
not expect a full build to get noticeably faster from it. If link time is
what hurts, look at unity builds and the compiler cache first.

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

1. **Use a compiler cache** - 2-3x faster rebuilds; on by default
2. **Use mold** - about halves link time; on by default, but link is a small
   share of this project's build
3. **Use Ninja** - Better parallelization than Make
4. **Use unity builds** - 20-40% faster compilation
5. **Use presets** - Consistent, reproducible builds
6. **Parallel builds** - Use `-j$(nproc)` for maximum parallelism

## Benchmark Results

Typical build times on a 32-core machine:

| Configuration            | Initial Build | Rebuild |
| ------------------------ | ------------- | ------- |
| Release (no cache)       | 5 min         | 5 min   |
| Release (ccache)         | 5 min         | 1 min   |
| Release (unity)          | 4 min         | 4 min   |
| Release (ccache + unity) | 4 min         | 45 sec  |

## CI Integration

For CI systems, use the `ci` preset -- but note it does not build on this tree yet:

```bash
cmake --preset ci
cmake --build --preset ci   # fails, see below
ctest --preset ci
```

The `ci` preset:

- Enables unity builds
- Keeps the compiler cache on, so a runner that installs sccache reuses it
- Fails on missing tests

### The `ci` preset currently fails to build

`ci` sets `CMAKE_UNITY_BUILD=ON`, which concatenates several .cpp files into one translation unit.
Two things that compile fine separately collide when merged:

```text
tensor_parallel_layers.cpp:39:  error: redefinition of
    'void cuda::neural::{anonymous}::throw_shape_error(const char*)'
tensor_parallel_attention.cpp:36: error: redefinition of
    'void cuda::neural::{anonymous}::throw_no_weight(const char*)'
```

Both names are defined in an anonymous namespace in two files each. The anonymous namespace makes
them internal to a translation unit, which is enough when every .cpp is its own unit -- and not enough
when unity builds merge those units.

Behind those, unity also surfaces missing includes that the per-file build was masking:
`src/matrix/add.cu` and `src/convolution/conv2d.cu` use `cuda::algo::detail::KernelLauncher` and
`calc_grid_2d` without including `cuda/algo/kernel_launcher.h`, which compiled only because some other
file in the same unity blob supplied it.

Renaming the two helpers fixes the first two errors; the include gaps are a separate, wider change.
`.github/workflows/` does not invoke this preset -- its jobs use their own CMake invocations -- so the
default build is unaffected. Use the `dev` preset or a plain `cmake -B build` until this is resolved.

To make the cache actually pay off in CI, give sccache a shared backend so
runners do not each start cold:

```yaml
env:
  SCCACHE_BUCKET: ${{ vars.SCCACHE_BUCKET }}
  SCCACHE_REGION: ${{ vars.SCCACHE_REGION }}
```

Without a backend sccache still caches, but only within a single job.

## Troubleshooting

### No compiler cache found

Configure prints `Compiler cache: none` and the build proceeds unchanged.
Install one to enable it:

```bash
sudo apt install ccache      # or: cargo install sccache
```

To silence even the detection step:

```bash
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
- [sccache documentation](https://github.com/mozilla/sccache)
- [mold documentation](https://github.com/rui314/mold)
- [CMake Unity Builds](https://cmake.org/cmake/help/latest/prop_tgt/UNITY_BUILD.html)
- [CMake CMAKE_LINKER_TYPE](https://cmake.org/cmake/help/latest/variable/CMAKE_LINKER_TYPE.html)
- [Ninja documentation](https://ninja-build.org/manual.html)
