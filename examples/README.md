# Nova Examples

This directory contains runnable example programs demonstrating Nova features.

## Building Examples

```bash
# All examples are built as part of the main build
cmake -G Ninja -B build
cmake --build build --parallel

# Or build specific examples
cmake --build build --target image_processing
```

## Available Examples

### Image Processing

**File:** `image_processing.cpp`

Demonstrates CUDA image processing with:
- Sobel edge detection
- Gaussian blur
- Morphological operations

```bash
./build/bin/image_processing --input image.pgm --output result.pgm --kernel sobel
```

### Graph Algorithms

**File:** `graph_algorithms.cpp`

Demonstrates GPU graph processing with:
- BFS (Breadth-First Search)
- PageRank

```bash
./build/bin/graph_algorithms --algorithm bfs --nodes 10000 --edges 50000
./build/bin/graph_algorithms --algorithm pagerank --nodes 10000 --iterations 20
```

### Neural Network Primitives

**File:** `neural_net.cpp`

Demonstrates CUDA neural network operations with:
- Matrix multiply with bias
- ReLU activation
- Layer normalization
- Softmax

```bash
./build/bin/neural_net --batch 32 --seq_len 128 --hidden 512
```

### Distributed Training

**File:** `distributed_training.cpp`

Demonstrates multi-GPU distributed training with:
- NCCL collectives
- All-reduce gradients
- Multi-node support via MPI

```bash
# Requires MPI and NCCL
mpirun -n 2 --allow-run-as-root ./distributed_training --batch 64 --epochs 10
```

## Compilation

### Single-GPU Examples

```bash
g++ -std=c++23 \
    -I /path/to/nova/include \
    -I /usr/local/cuda/include \
    examples/image_processing.cpp \
    -L /path/to/nova/build/lib -lcuda_impl \
    -L /usr/local/cuda/lib64 -lcudart \
    -o image_processing
```

### Distributed Example

```bash
mpicc -std=c++23 \
    -I /path/to/nova/include \
    -I /usr/local/cuda/include \
    examples/distributed_training.cpp \
    -L /path/to/nova/build/lib -lcuda_impl \
    -L /usr/local/cuda/lib64 -lcudart -lnccl \
    -o distributed_training
```

## Requirements

- CUDA Toolkit 12.0+
- CMake 4.0+
- C++23 compiler

For distributed examples:
- MPI implementation (OpenMPI, MPICH)
- NCCL library

## Troubleshooting

### "No CUDA-capable device"

Ensure `CUDA_VISIBLE_DEVICES` is set correctly:
```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/image_processing
```

### NCCL initialization failed

Ensure NCCL is installed and CUDA can see multiple GPUs:
```bash
nvidia-smi  # Verify GPUs are visible
```

### Build errors

Clean and rebuild:
```bash
rm -rf build
cmake -G Ninja -B build
cmake --build build --parallel
```
