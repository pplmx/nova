# Transformer Implementation Guide

## Overview

This guide walks through implementing transformer models using Nova's neural network primitives.

## Multi-Head Attention

```cpp
#include <cuda/neural/transformer/attention.h>

// Configure attention
cuda::neural::transformer::MultiHeadAttentionConfig config;
config.num_heads = 8;
config.head_dim = 64;
config.dropout_rate = 0.1f;

cuda::neural::transformer::MultiHeadAttention attention(config);

// Forward pass
attention.forward_self_attention(
    input.data(), output.data(),
    batch_size, seq_len, hidden_dim, stream
);
```

## Positional Encoding

```cpp
#include <cuda/neural/transformer/attention.h>

cuda::neural::transformer::PositionalEncodingConfig config;
config.type = cuda::neural::transformer::PositionalEncodingType::Sinusoidal;
config.max_seq_len = 512;
config.embed_dim = 512;

cuda::neural::transformer::PositionalEncoding pos_enc(config);

// Apply to input
pos_enc.forward(input.data(), output.data(), batch_size, seq_len, stream);
```

## Loss Functions

### Cross-Entropy Loss

```cpp
#include <cuda/neural/loss/loss_functions.h>
#include <vector>

cuda::neural::loss::CrossEntropyConfig config;
config.num_classes = 10;
config.reduction_mean = true;

std::vector<float> output(batch_size);
// targets is const int* (class indices), not float: the signature is
// cross_entropy_loss(const float* predictions, const int* targets, float* output, ...)
float loss = cuda::neural::loss::cross_entropy_loss(
    predictions.data(), targets.data(), output.data(),
    batch_size, 10, config, stream
);
```

### Focal Loss

```cpp
#include <cuda/neural/loss/loss_functions.h>

cuda::neural::loss::FocalLossConfig config;
config.alpha = 1.0f;
config.gamma = 2.0f;

float loss = cuda::neural::loss::focal_loss(
    predictions.data(), targets.data(), output.data(),
    batch_size, 5, config, stream
);
```

## Optimizers

### AdamW

```cpp
#include <cuda/neural/optimizers/optimizers.h>

cuda::neural::optimizers::OptimizerConfig config;
config.learning_rate = 0.001f;
config.weight_decay = 0.01f;

cuda::neural::optimizers::AdamWOptimizer optimizer(config);
optimizer.step(params.data(), grads.data(), num_params, step, stream);
```

### LAMB

```cpp
#include <cuda/neural/optimizers/optimizers.h>

cuda::neural::optimizers::LAMBConfig config;
config.learning_rate = 0.001f;
config.use_layer_adaptation = true;

cuda::neural::optimizers::LAMBOptimizer optimizer(config);
optimizer.step(params.data(), grads.data(), num_params, step,
               &layer_norm_1, &layer_norm_2, stream);
```

### Gradient Clipping

```cpp
#include <cuda/neural/optimizers/optimizers.h>

cuda::neural::optimizers::GradientClipConfig clip_config;
clip_config.max_norm = 1.0f;
clip_config.norm_type = cuda::neural::optimizers::GradientClipConfig::NormType::L2;

cuda::neural::optimizers::GradientClipper clipper(clip_config);
float norm = clipper.clip(grads.data(), num_grads, stream);
```

## Complete Example

```cpp
#include <cuda/neural/transformer/attention.h>
#include <cuda/neural/loss/loss_functions.h>
#include <cuda/neural/optimizers/optimizers.h>

// Setup
MultiHeadAttention attention({8, 64, 0.1f});
PositionalEncoding pos_enc({PositionalEncodingType::Sinusoidal, 512, 512});
CrossEntropyLossFunction loss_fn({10});
AdamWOptimizer optimizer({0.001f, 0.01f});

// Training loop
for (int step = 0; step < max_steps; ++step) {
    // Forward
    attention.forward_self_attention(x, attn_out, B, S, H, stream);
    pos_enc.forward(attn_out, x_pos, B, S, stream);

    // Loss
    float loss = loss_fn.forward(logits, targets, output.data(), B, stream);

    // Backward + clip
    backward(loss);
    clipper.clip(grads.data(), num_grads, stream);

    // Optimizer step
    optimizer.step(params.data(), grads.data(), num_params, step, stream);
}
```

## Performance Tips

1. **Use kernel fusion** for matmul + bias + activation chains
2. **Enable autotuning** to find optimal block sizes
3. **Profile with TimelineVisualizer** to identify bottlenecks
4. **Use memory pool** for frequent allocations
