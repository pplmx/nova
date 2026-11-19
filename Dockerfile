# Build and run the nova demo binary.
#
# The previous version could not build: it started from rockylinux:9, which has
# no CUDA toolkit, while CMakeLists.txt declares `project(... LANGUAGES CXX
# CUDA)` and calls find_package(CUDAToolkit REQUIRED); its `dnf install cmake`
# was also 3.31.8 against a cmake_minimum_required(VERSION 4.0). It then copied
# /app/build/nova, but the project writes binaries to build/bin.
#
# Two stages, matching the CUDA image flavours: `devel` has nvcc and the
# headers, `runtime` has the math libraries the binary links against
# (libcublas, libcusolver, libcufft) without the toolchain.
#
# This image needs an NVIDIA GPU and the NVIDIA container runtime to do
# anything: the demo allocates device memory and aborts with "no
# CUDA-capable device is detected" without one. It therefore cannot be smoke
# tested on a GitHub-hosted runner, which is why docker.yml no longer runs it.

ARG CUDA_VERSION=12.6.2
ARG UBUNTU_VERSION=24.04

# --------------------------------------------------------------------------
# Build stage
# --------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive

# cmake from apt on 24.04 is 3.28, below the 4.0 this project requires, so
# take CMake from PyPI like ci.yml does. ninja and gcc come from apt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        g++ \
        git \
        libssl-dev \
        ninja-build \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages "cmake>=4.0" \
    && cmake --version

WORKDIR /src

# COPY .. is normalised by the builder to the build context root (BuildKit
# strips the parent-directory traversal), so this copies the repository.
COPY .. .

# NCCL and MPI off: neither is present in this image, and the demo needs
# neither. NCCL=OFF keeps configure from looking for NCCL. Tests and
# benchmarks are off because only the `nova` target is wanted here: those
# subdirectories clone GoogleTest and Google Benchmark at configure time, which
# would add a network fetch and a large build for nothing.
RUN cmake -G Ninja -B build \
        -DNOVA_ENABLE_NCCL=OFF \
        -DNOVA_ENABLE_MPI=OFF \
        -DNOVA_BUILD_TESTS_AND_BENCHMARKS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target nova --parallel

# --------------------------------------------------------------------------
# Runtime stage
# --------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# CMAKE_RUNTIME_OUTPUT_DIRECTORY puts the demo at build/bin, not build/.
COPY --from=builder /src/build/bin/nova /usr/local/bin/nova

# The demo is interactive-ish: it prints a benchmark table and exits. It is
# run with --help style probing in mind, so keep the entrypoint simple.
ENTRYPOINT ["/usr/local/bin/nova"]
