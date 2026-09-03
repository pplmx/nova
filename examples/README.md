# Nova Examples

This directory contains runnable example programs demonstrating Nova's current
public API. Each example is a standalone executable; the targets are always
registered, so they build when the main build runs:

```bash
cmake -S . -B build
cmake --build build --parallel

# Or build a specific example:
cmake --build build --target graph_algorithms
```

The available targets are `neural_net`, `image_processing`, `graph_algorithms`,
and `distributed_training`. All four have been ported to the current
`cuda::*` namespaces (the old pre-v2 `nova::*` APIs they referenced no longer
exist).

## Neural Network Primitives

**File:** `neural_net.cpp`

Demonstrates CUDA neural network operations with:

- Fused matrix multiply + bias + ReLU (`cuda::neural::fusion::FusedMatmulBiasAct`)
- Layer normalization
- Softmax (verified: every output row sums to 1)

```bash
./build/bin/neural_net --batch 32 --seq_len 16 --hidden 128
```

Options: `--batch`, `--seq_len`, `--hidden`, `--help`.

## Image Processing

**File:** `image_processing.cpp`

Demonstrates CUDA image kernels with embedded PGM (P5 binary) IO:

- Sobel edge detection (`cuda::image::sobelEdgeDetection`, RGB; a grayscale PGM
  is adapted via the R channel)
- Gaussian blur (`cuda::algo::gaussianBlur`)
- Morphological dilation (`dilateImage`)

```bash
# Process an existing PGM:
./build/bin/image_processing --input in.pgm --output out.pgm --kernel sobel
./build/bin/image_processing --input in.pgm --output out.pgm --kernel blur --iterations 2
```

Options: `--kernel sobel|blur|morphology`, `--input`, `--output`, `--iterations`,
`--help`. An unknown `--kernel` is rejected before any device work (the old
example silently no-op'ed and saved garbage). The example embeds its own
minimal PGM (P5 binary) reader/writer.

## Graph Algorithms

**File:** `graph_algorithms.cpp`

Demonstrates GPU graph processing with:

- BFS (`cuda::graph::bfs`) over a generated random edge list
- PageRank (`cuda::graph::pagerank`), reporting iterations and final delta

```bash
./build/bin/graph_algorithms --algorithm bfs --nodes 10000 --edges 50000
./build/bin/graph_algorithms --algorithm pagerank --nodes 10000 --iterations 20
```

Options: `--algorithm bfs|pagerank`, `--nodes`, `--edges`, `--source`, `--iterations`,

`--damping`, `--tolerance`, `--help`.

## Distributed Training

**File:** `distributed_training.cpp`

Demonstrates multi-GPU distributed all-reduce with the current device-mesh
architecture (no MPI):

- Shared `cuda::nccl::NcclContext` singleton discovers the visible GPUs
- `cuda::distributed::DistributedReduce::all_reduce(Sum)` across the whole group
- One thread per device (the collective contract the multi-GPU suites use);
  every rank verifies it ends with the group sum

```bash
# Needs >= 2 visible GPUs for a real NCCL collective:
CUDA_VISIBLE_DEVICES=1,2 ./build/bin/distributed_training --chunk 1024
```

On a single visible GPU the operation degenerates to a local copy (identity) and
the demo still verifies the result. Options: `--chunk`, `--help`.

## Compilation

Examples are built by the CMake targets above — no manual `g++`/`mpicc`
invocation is required. For a manual build (shared lib), link against
`cuda_impl` and the CUDA runtime:

```bash
g++ -std=c++23 \
    -I include \
    -I /usr/local/cuda/include \
    examples/neural_net.cpp \
    -L build/lib -lcuda_impl \
    -L /usr/local/cuda/lib64 -lcudart \
    -o neural_net
```

The `distributed_training` example does **not** require MPI; it is a plain
single-process program (the library owns NCCL initialization).

## Requirements

- CUDA Toolkit 12.0+ (with NCCL for the distributed example)
- CMake 4.0+
- C++23 compiler

## Troubleshooting

### "No CUDA-capable device"

Ensure `CUDA_VISIBLE_DEVICES` is set to a device that is actually free:

```bash
nvidia-smi                                # pick a free device
CUDA_VISIBLE_DEVICES=1 ./build/bin/image_processing --generate ...
```

### The distributed example exits with an NCCL error

NCCL must see the group of GPUs on one machine and the devices must be free:

```bash
nvidia-smi  # Verify >= 2 GPUs are visible and idle
CUDA_VISIBLE_DEVICES=1,2 ./build/bin/distributed_training
```

### Build errors

Clean and rebuild:

```bash
rm -rf build
cmake -S . -B build
cmake --build build --parallel
```
