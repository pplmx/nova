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

// CheckpointManager is a singleton -- its constructor is private. Storage is supplied through
// initialize(backend, stream); the interval is in STEPS, not a duration.
auto& manager = nova::checkpoint::CheckpointManager::instance();
auto backend = std::make_shared<nova::checkpoint::FileStorageBackend>("checkpoints/");
manager.initialize(backend, nullptr);
manager.set_checkpoint_interval(1000);  // auto-save every 1000 steps
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
#include <iostream>
#include <map>
#include <string>
#include <vector>

class Trainer {
public:
    Trainer()
        : manager(nova::checkpoint::CheckpointManager::instance()) {
        // initialize() takes the backend; the interval is a step count, not a duration.
        auto backend = std::make_shared<nova::checkpoint::FileStorageBackend>("checkpoints/");
        manager.initialize(backend, nullptr);
        manager.set_checkpoint_interval(500);
    }

    void train(int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // ... one training step, updating model_state / optimizer_state ...

            // on_step_complete() decides whether to save; there is no should_checkpoint().
            manager.on_step_complete(epoch);
        }
    }

    void resume() {
        // load_latest_checkpoint() returns false when there is nothing to load.
        if (manager.load_latest_checkpoint(model_state, optimizer_state)) {
            std::cout << "Loaded the latest checkpoint\n";
        }
    }

private:
    nova::checkpoint::CheckpointManager& manager;
    std::map<std::string, std::vector<float>> model_state;
    std::map<std::string, std::vector<float>> optimizer_state;
};
```

## File Storage Backend

```cpp
#include <cuda/checkpoint/checkpoint_manager.h>  // FileStorageBackend lives here

// The base path is a constructor argument; there is no set_base_path() and no set_atomic_writes().
auto backend = std::make_shared<nova::checkpoint::FileStorageBackend>("/tmp/nova_checkpoints");

manager.initialize(backend, nullptr);
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
