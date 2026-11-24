# Performance Tooling Guide

**Version:** v2.11
**Date:** 2026-05-02

This guide covers the performance tooling infrastructure added in v2.11 for the Nova CUDA library.

## Overview

The performance tooling module provides:

1. **NVBlox Metrics** - Kernel-level profiling and custom metrics
2. **Kernel Fusion Analysis** - Detection of fusion opportunities
3. **Memory Bandwidth Analysis** - Roofline model and utilization tracking
4. **Dashboard Export** - Unified data export for visualization

## Quick Start

```cpp
#include <cuda/performance/nvblox_metrics.h>
#include <cuda/performance/bandwidth/roofline_model.h>
#include <cuda/performance/dashboard/dashboard_exporter.h>
#include <string>

// Collect kernel metrics
auto& collector = cuda::performance::NVBloxMetricsCollector::instance();
collector.register_metric("latency", cuda::performance::MetricType::Latency);
collector.add_sample("latency", 1.5);

// Analyze performance
cuda::performance::bandwidth::RooflineModel roofline;
roofline.add_point("matmul", 1024, 64, 1000);

// Export dashboard
cuda::performance::dashboard::DashboardExporter exporter;
exporter.add_roofline_data(roofline);
std::string json = exporter.to_json();```

## NVBlox Metrics

### KernelMetricsCollector

Thread-safe metric collection with JSON/CSV export.

```cpp
#include <cuda/performance/nvblox_metrics.h>

#include <string>
using namespace cuda::performance;

auto& collector = NVBloxMetricsCollector::instance();

// Register metrics
collector.register_metric("latency", MetricType::Latency);
collector.register_metric("throughput", MetricType::Throughput);

// Add samples
collector.add_sample("latency", 1.0);
collector.add_sample("throughput", 100.0);

// Record kernel data
KernelMetrics km;
km.name = "matmul_kernel";
km.latency_ns = 1000;
km.throughput_gflops = 500.0;
collector.record_kernel(km);

// Export
std::string json = collector.to_json();
std::string csv = collector.to_csv();```

### KernelProfiler

CUDA event-based kernel profiling.

```cpp
#include <cuda/performance/kernel_profiler.h>

using namespace cuda::performance;

auto& profiler = KernelProfiler::instance();
profiler.enable();

// Profile a kernel
profiler.record_start("my_kernel", stream);
kernel<<<...>>>(...);
profiler.record_end("my_kernel", stream);

// Get latency
uint64_t latency = profiler.get_kernel_latency_ns("my_kernel");

// Estimate occupancy
float occ = profiler.estimate_occupancy(256, 4);```

## Kernel Fusion Analysis

### KernelFusionAnalyzer

Detects fusion opportunities in operation graphs.

```cpp
#include <cuda/performance/fusion/kernel_fusion_analyzer.h>

#include <iostream>
using namespace cuda::performance::fusion;

KernelFusionAnalyzer analyzer;

// Add operations
analyzer.add_operation({"matmul", OpType::Matmul, 100, 1024, 64});
analyzer.add_operation({"bias", OpType::ElementWise, 50, 64, 64});

// Detect opportunities
auto opportunities = analyzer.detect_opportunities();

for (const auto& opp : opportunities) {
    std::cout << opp.pattern().name << "\n";
    std::cout << "Potential latency saved: " << opp.potential_latency_saved_ns() << "ns\n";
}```

### FusionRecommendationEngine

Generates actionable recommendations with confidence levels.

```cpp
#include <cuda/performance/fusion/fusion_profitability.h>

#include <iostream>
using namespace cuda::performance::fusion;

FusionProfitabilityConfig config;
config.launch_overhead_us = 100.0;  // 100us launch overhead threshold

FusionRecommendationEngine engine(config);
auto recommendations = engine.generate_recommendations(opportunities);

for (const auto& rec : recommendations) {
    std::cout << "Pattern: " << rec.pattern_name() << "\n";
    std::cout << "Confidence: " << to_string(rec.confidence()) << "\n";
    std::cout << "Speedup: " << rec.speedup_factor() << "x\n";
    std::cout << "Suggestion: " << rec.suggestion() << "\n";
}```

### Known Fusion Patterns

| Pattern              | Description          | Confidence |
| -------------------- | -------------------- | ---------- |
| matmul_bias_act_relu | Matmul + bias + ReLU | HIGH       |
| matmul_bias_act_gelu | Matmul + bias + GELU | HIGH       |
| conv_bias_act_relu   | Conv + bias + ReLU   | HIGH       |
| relu_pool            | ReLU + pooling       | HIGH       |
| elementwise_chain    | Element-wise chain   | HIGH       |

## Memory Bandwidth Optimization

### RooflineModel

Memory bandwidth roofline analysis.

```cpp
#include <cuda/performance/bandwidth/roofline_model.h>

#include <iostream>
#include <string>
using namespace cuda::performance::bandwidth;

RooflineModel model;

// Add kernel data
model.add_point("matmul", 1024, 64, 1000);  // flops, bytes, elapsed_ns
model.add_point("reduce", 64, 8192, 2000);

// Get device peaks
const auto& peaks = model.peaks();
std::cout << "Peak FP64: " << peaks.fp64_gflops << " GFLOPS\n";
std::cout << "Peak Bandwidth: " << peaks.hbm_bandwidth_gbs << " GB/s\n";
std::cout << "Ridge Point: " << model.ridge_point() << " FLOP/byte\n";

// Classify kernels
auto memory_bound = model.get_memory_bound_points();
auto compute_bound = model.get_compute_bound_points();

// Export
std::string json = model.to_json();
std::string csv = model.to_csv();```

### BandwidthUtilizationTracker

Track memory bandwidth utilization.

```cpp
#include <cuda/performance/bandwidth/roofline_model.h>

#include <iostream>
using namespace cuda::performance::bandwidth;

BandwidthUtilizationTracker tracker;
tracker.set_peak_bandwidth(20.0, 20.0, 900.0);  // H2D, D2H, D2D

// Add samples
tracker.add_h2d_sample(15.0, 1e9, 66e6);  // bandwidth, bytes, elapsed_ns
tracker.add_d2d_sample(700.0, 1e9, 1.4e6);

// Get utilization
std::cout << "Average: " << tracker.average_bandwidth_gbs() << " GB/s\n";
std::cout << "Utilization: " << tracker.utilization_percent() << "%\n";

// Warning check
if (tracker.has_low_utilization_warning()) {
    std::cout << "WARNING: Low utilization detected\n";
}```

### CacheAnalyzer

Cache hit rate analysis.

```cpp
#include <cuda/performance/bandwidth/cache_analyzer.h>

#include <iostream>
using namespace cuda::performance::bandwidth;

CacheAnalyzer analyzer;
auto metrics = analyzer.analyze(0);

if (metrics.available) {
    std::cout << "L1 hit rate: " << metrics.l1_hit_rate * 100 << "%\n";
    std::cout << "L2 hit rate: " << metrics.l2_hit_rate * 100 << "%\n";
}```

## Dashboard Export

### DashboardExporter

Unified dashboard data export.

```cpp
#include <cuda/performance/dashboard/dashboard_exporter.h>

#include <string>
using namespace cuda::performance::dashboard;

DashboardExporter exporter;
exporter.add_roofline_data(roofline);
exporter.add_fusion_data(recommendations);
exporter.add_bandwidth_data(tracker);
exporter.add_kernel_count(collector.total_kernel_count());

// Export
std::string json = exporter.to_json();
std::string csv = exporter.to_csv();```

### FlameGraphGenerator

Generate flame graphs from NVTX traces.

```cpp
#include <cuda/performance/dashboard/flame_graph.h>

#include <string>
using namespace cuda::performance::dashboard;

FlameGraphGenerator generator;

// Add Chrome trace events
ChromeTraceEvent event;
event.name = "kernel1";
event.category = "algo";
event.ph = "X";  // Complete event
event.ts = 1000;  // Start time
event.dur = 100;  // Duration
generator.add_event(event);

// Build flame graph
auto flame_graph = generator.build_flame_graph();

// Export
std::string json = generator.to_json();
std::string trace = generator.to_chrome_trace();```

## CMake Configuration

```cmake
# Enable performance tooling
set(NOVA_ENABLE_NVBLOX ON)

# Build with CUDA
find_package(CUDAToolkit REQUIRED)
target_link_libraries(your_target PRIVATE cuda_impl)
```

## NVTX Domains

New performance domains added in v2.11:

- `nova.performance` - General performance profiling
- `nova.performance.nvblox` - NVBlox-specific metrics
- `nova.performance.fusion` - Fusion analysis
- `nova.performance.bandwidth` - Bandwidth analysis

## Requirements Coverage

| Requirement  | Component                   | Phase |
| ------------ | --------------------------- | ----- |
| NVBlox-01    | NVBloxMetricsCollector      | 93    |
| NVBlox-02    | KernelProfiler              | 93    |
| NVBlox-03    | MetricAggregators           | 93    |
| FUSION-01    | KernelFusionAnalyzer        | 94    |
| FUSION-02    | FusionProfitabilityModel    | 94    |
| FUSION-03    | FusionRecommendationEngine  | 94    |
| BANDWIDTH-01 | RooflineModel               | 95    |
| BANDWIDTH-02 | BandwidthUtilizationTracker | 95    |
| BANDWIDTH-03 | CacheAnalyzer               | 95    |
| DASH-01      | DashboardExporter           | 96    |
| DASH-02      | Roofline export             | 96    |
| DASH-03      | FlameGraphGenerator         | 96    |
| INT-01       | nvbench integration         | 96    |
| INT-02       | NVTX domains                | 93    |

## See Also

- [Production Guide](./PRODUCTION.md)
- [Build Performance Guide](./build-performance.md)
- [API Reference](../include/cuda/performance/)
