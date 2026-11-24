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
#include <cuda/device/error.h>
#include <cuda/mesh/device_mesh.h>
#include <stdio.h>

int main() {
    // The mesh is a process-wide singleton and lives in cuda::mesh, not nova::.
    // DeviceMesh describes the topology; it does not own a rank or set the current device.
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    const int device_count = mesh.device_count();
    printf("Mesh has %d device(s)\n", device_count);

    // Enumerate what the mesh knows about each device.
    for (const auto& peer : mesh.get_mesh_devices()) {
        printf("  device %d\n", peer.device_id);
    }

    // Bind this thread to a device before any CUDA call. That is a CUDA runtime call, not a
    // DeviceMesh one: the mesh reports topology, choosing a rank is the launcher's job.
    CUDA_CHECK(cudaSetDevice(0));

    // Your computation here...

    return 0;
}
```

## Multi-GPU Reduction

```cpp
#include <cuda/distributed/reduce.h>

// Each GPU has partial data
cuda::memory::Buffer<float> local_data(size);
initialize_data(local_data);

// All-reduce across GPUs. all_reduce is a static method on DistributedReduce, not a free function
// in cuda::distributed, and it takes (send, recv, count, op, dtype) -- there is no root rank.
cuda::distributed::DistributedReduce::all_reduce(
    local_data.device_data(),                  // send buffer
    local_data.device_data(),                  // recv buffer (in place)
    size,                                      // element count
    cuda::distributed::ReductionOp::Sum        // reduction operator
);
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

// Row-wise split is handled by DistributedMatmul itself: it splits the output rows across the
// devices it sees and gathers the result. There is no split_rows()/execute() pair to drive.
cuda::distributed::DistributedMatmul matmul;
cuda::distributed::DistributedMatmulOptions opts;
matmul.matmul(local_A, local_B, local_C, m, n, k, opts);

// Results are distributed across GPUs
```

## Best Practices

1. **Minimize host-device transfers** - Keep data on GPU when possible
2. **Use streams** - Overlap computation and communication
3. **Enable peer access early** - Before first transfer
4. **Synchronize properly** - Use events for coordination

## Complete Example

```cpp
#include <cuda/algo/reduce.h>
#include <cuda/distributed/reduce.h>
#include <cuda/mesh/device_mesh.h>
#include <cuda/memory/buffer.h>
#include <stdio.h>
#include <vector>

int main() {
    const int device_count = cuda::mesh::DeviceMesh::instance().device_count();
    const size_t size = 1024 * 1024;

    // Each rank reduces its own shard first. reduce_sum() takes a device pointer and a count and
    // returns the scalar on the host; it does not take an output buffer.
    std::vector<float> host_shard(size, 1.0f);
    cuda::memory::Buffer<float> local_shard(size);
    local_shard.copy_from(host_shard.data(), size);
    cuda::memory::Buffer<float> partial(1);
    partial.copy_from(std::vector<float> {cuda::algo::reduce_sum(local_shard.data(), size)}.data(), 1);

    // All-reduce across ranks. all_reduce takes (send, recv, count, op); there is no root-rank
    // argument and no stream argument on the synchronous form. Both pointers must be device
    // allocations, and passing the same pointer twice performs the operation in place.
    cuda::distributed::DistributedReduce::all_reduce(partial.data(), partial.data(), 1, cuda::distributed::ReductionOp::Sum);

    // Every rank now holds the global sum, so no rank check is needed to print it.
    std::vector<float> total(1);
    partial.copy_to(total.data(), 1);
    printf("Global sum from %d device(s): %f\n", device_count, total[0]);

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
