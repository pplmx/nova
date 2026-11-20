# Multi-GPU Tutorial

How to use Nova's DeviceMesh for multi-GPU CUDA programming.

## Overview

Nova provides `DeviceMesh` for easy multi-GPU coordination:

```cpp
#include <cuda/mesh/device_mesh.h>
#include <cuda/distributed/reduce.h>
```

## Prerequisites

- Multiple NVIDIA GPUs (or single GPU for basic testing)
- NCCL library (optional, for optimized collectives)

## Basic Multi-GPU Setup

```cpp
#include <cuda/mesh/device_mesh.h>

int main() {
    // Initialize device mesh
    auto& mesh = nova::DeviceMesh::instance();

    // Get local device count
    int device_count = mesh.device_count();
    int local_rank = mesh.local_rank();

    printf("Using %d GPU(s), this is GPU %d\n", device_count, local_rank);

    // Set device for current rank
    mesh.set_device(local_rank);

    // Your computation here...

    return 0;
}
```

## Multi-GPU Reduction

```cpp
#include <cuda/distributed/reduce.h>

// Each GPU has partial data
nova::memory::Buffer<float> local_data(size);
initialize_data(local_data);

// All-reduce across GPUs
nova::distributed::all_reduce(
    local_data.device_data(),  // Input/output buffer
    size,                      // Element count
    0,                         // Root rank (or any for broadcast result)
    mesh.get_stream()           // CUDA stream for async ops
);

// Now all GPUs have the same summed result
```

## Peer Memory Access

Enable direct GPU-to-GPU transfers:

```cpp
// Check if peer access is available
if (mesh.can_access_peer(src_device, dst_device)) {
    // Enable peer access
    mesh.enable_peer_access(src_device, dst_device);

    // Perform direct copy
    mesh.peer_copy(dst_device, dst, src_device, src, size);

    // Disable when done
    mesh.disable_peer_access(src_device, dst_device);
}
```

## Multi-GPU Matmul

```cpp
#include <cuda/distributed/matmul.h>

// Split matrix across GPUs by rows
int rows_per_gpu = matrix_height / device_count;
int local_start_row = local_rank * rows_per_gpu;

nova::distributed::DistributedMatmul matmul;
matmul.split_rows(rows_per_gpu, local_start_row);
matmul.execute(local_A, local_B, local_C);

// Results are distributed across GPUs
```

## Best Practices

1. **Minimize host-device transfers** - Keep data on GPU when possible
2. **Use streams** - Overlap computation and communication
3. **Enable peer access early** - Before first transfer
4. **Synchronize properly** - Use events for coordination

## Complete Example

```cpp
#include <cuda/mesh/device_mesh.h>
#include <cuda/distributed/reduce.h>
#include <cuda/algo/reduce.h>
#include <stdio.h>

int main() {
    auto& mesh = nova::DeviceMesh::instance();
    int local_rank = mesh.local_rank();
    int device_count = mesh.device_count();

    const size_t size = 1024 * 1024;

    // Each GPU computes partial sum
    nova::memory::Buffer<float> local_data(size);
    initialize_data(local_data);

    nova::memory::Buffer<float> partial_sum(1);
    nova::algo::reduce(local_data.device_data(), 
                       partial_sum.device_data(), 
                       size);

    // All-reduce to get global sum
    nova::distributed::all_reduce(
        partial_sum.device_data(), 1, 0, mesh.get_stream()
    );

    // Print result (only on root GPU)
    if (local_rank == 0) {
        partial_sum.sync_to_host();
        printf("Global sum: %f\n", partial_sum.host_data()[0]);
    }

    return 0;
}
```

## Troubleshooting

### "Peer access not available"

- Check GPU topology: `nvidia-smi topo -m`
- Ensure GPUs are on same PCIe switch
- Some cloud instances don't support P2P

### Slow multi-GPU communication

- Enable peer access for direct transfers
- Use NCCL for optimized collectives
- Check network bandwidth for multi-node

## Next Steps

- [Checkpoint Tutorial](03-checkpoint.md) - Save distributed training state
- [Architecture](../ARCHITECTURE.md) - Layer and module reference
