# Checkpoint and Restore Tutorial

How to save and restore training state with Nova's CheckpointManager.

## Overview

Nova provides `CheckpointManager` for saving:

- Model weights
- Optimizer states
- RNG state (for reproducibility)
- Custom user data

## Basic Usage

```cpp
#include <cuda/checkpoint/checkpoint_manager.h>

// Create checkpoint manager
nova::checkpoint::CheckpointManager manager;
manager.set_storage_backend("checkpoints/");
manager.set_interval(std::chrono::minutes(10));  // Auto-save every 10 minutes
```

## Saving State

```cpp
// Define what to save
struct TrainingState {
    std::vector<float> weights;
    std::vector<float> optimizer_state;
    int epoch;
    float loss;
};

// Create state
TrainingState state = load_or_init_state();

// Save checkpoint
manager.save("training_checkpoint", state, []() {
    // This lambda is called to serialize state
    return serialize_to_bytes(state);
});
```

## Loading State

```cpp
// Load checkpoint
auto bytes = manager.load("training_checkpoint");
TrainingState state = deserialize_from_bytes<TrainingState>(bytes);
```

## Complete Training Loop Example

```cpp
#include <cuda/checkpoint/checkpoint_manager.h>
#include <cuda/algo/reduce.h>
#include <cuda/memory/buffer.h>
#include <iostream>

class Trainer {
public:
    Trainer() {
        manager.set_storage_backend("checkpoints/");
        manager.set_interval(std::chrono::minutes(5));
    }

    void train(int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Training step
            float loss = training_step(epoch);
            std::cout << "Epoch " << epoch << " loss: " << loss << "\n";

            // Auto-save if interval reached
            if (manager.should_checkpoint()) {
                save_checkpoint(epoch, loss);
            }
        }
    }

    void resume() {
        if (manager.checkpoint_exists("latest")) {
            std::cout << "Loading checkpoint...\n";
            auto state = load_checkpoint();
            restore_state(state);
        }
    }

private:
    nova::checkpoint::CheckpointManager manager;
    // ... model state
};
```

## File Storage Backend

```cpp
#include <cuda/checkpoint/checkpoint_manager.h>  // FileStorageBackend lives here

// Use file-based storage
auto backend = std::make_unique<nova::checkpoint::FileStorageBackend>();
backend->set_base_path("/tmp/nova_checkpoints");
backend->set_atomic_writes(true);  // Safe writes

manager.set_storage_backend(std::move(backend));
```

## Key Features

### Atomic Writes

Checkpoints are written atomically to prevent corruption:

```cpp
manager.set_atomic_writes(true);
```

### Compression

Not available. CheckpointManager writes the tensor blobs uncompressed, and there is no
enable_compression() method or CompressionType enum in this tree. Compressing a
checkpoint directory means compressing the files it writes, outside the manager.

### Async Saves

Non-blocking checkpoint saves:

```cpp
manager.save_async("checkpoint_name", state, callback);
```

## Checkpoint Metadata

Automatic metadata tracking:

```cpp
// Access checkpoint info
auto info = manager.get_checkpoint_info("training_v1");
std::cout << "Created: " << info.created_at << "\n";
std::cout << "Size: " << info.size_bytes << " bytes\n";
std::cout << "Epoch: " << info.metadata["epoch"] << "\n";
```

## Best Practices

1. **Regular saves** - Every N minutes or N batches
2. **Multiple versions** - Keep last 3-5 checkpoints
3. **Validate on load** - Check integrity before using
4. **Compression for large models** - Save disk space

## Troubleshooting

### "Checkpoint file not found"

Check:

1. Storage path is correct
2. File permissions allow read
3. Path exists: `ls -la checkpoints/`

### "Corrupt checkpoint"

- Enable atomic writes
- Use checksums: `manager.enable_checksum()`
- Keep previous checkpoint as backup

## Next Steps

- [Profiling Guide](04-profiling.md) - Benchmark your training
- [Architecture](../ARCHITECTURE.md) - Layer and module reference
