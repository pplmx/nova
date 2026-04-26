# Stack Research

**Domain:** CUDA/C++ Benchmarking and Performance Testing Infrastructure
**Researched:** 2026-04-26
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Google Benchmark | v1.9.x | C++ microbenchmark framework | Industry standard for C++ benchmarking with automatic CPU frequency scaling detection, statistical analysis, and JSON/CSV output. Native CMake integration. |
| NVTX (nvtx3) | CUDA 12.x bundled | GPU profiling annotations | Header-only C++ library (`<nvtx3/nvtx3.hpp>`) for annotating code ranges with colors, payloads, and domains. Integrates seamlessly with Nsight Systems/Compute. |
| pytest | 8.x-9.x | Python test harness orchestration | Industry-standard Python testing with subprocess execution, fixtures, and parametrize support. Excellent for CI integration. |
| pandas | 2.x | Benchmark data analysis | Efficient DataFrame operations for processing JSON benchmark outputs, computing statistics, and preparing data for visualization. |
| matplotlib | 3.x | Performance visualization | Mature plotting library with HTML output support via `savefig(format='html')` or `mpld3`/`plotly` backends for interactive dashboards. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-xdist | 3.x | Parallel test execution | When running multiple benchmark binaries in parallel across CPU cores. |
| pytest-json-report | 1.x | JSON test reporting | For CI artifact generation and integration with external dashboards. |
| plotly | 5.x | Interactive HTML charts | For interactive benchmark comparison dashboards with zoom/hover. |
| chevron | 5.x | Mustache templating | For generating static HTML reports from benchmark data with templated layouts. |
| requests | 2.x | HTTP API calls | For CI baseline storage (e.g., uploading to artifact storage, GitHub API for PR comments). |
| scipy | 1.x | Statistical analysis | For regression testing with confidence intervals and hypothesis testing. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| CMake `find_package(Benchmark)` | Google Benchmark integration | Use `BENCHMARK_ENABLE_GTEST_TESTS=OFF` to avoid conflicts with existing gtest setup. |
| CUDA event timing | GPU kernel timing | Wrap kernel launches with `cudaEvent_t` for wall-clock timing via `SetIterationTime()`. |
| Nsight Systems | GPU timeline visualization | NVTX annotations appear automatically in timeline. |
| GitHub Actions `upload-artifact` | Baseline storage | Store JSON baselines for regression comparison. |

## Installation

```bash
# Core C++ benchmark library (as submodule or external)
# In your CMakeLists.txt:
find_package(Benchmark 1.9 REQUIRED)
target_link_libraries(benchmark_tests PRIVATE Benchmark::Benchmark)

# CMake options to avoid conflicts with existing Google Test
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_INSTALL ON)

# Python dependencies
pip install pytest>=8.0 pytest-xdist>=3.0 pandas>=2.0 plotly>=5.0 chevron>=0.14 requests>=2.28 scipy>=1.10

# Optional: for HTML dashboard generation
pip install mpld3>=0.5 jinja2>=3.1
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Google Benchmark | Catch2 (with microbenchmark) | If you want to consolidate all testing under one framework, but Catch2's benchmark support is less mature. |
| pytest harness | Bazel `benchmark` rule | If you already use Bazel for the entire build system. |
| NVTX C++ (nvtx3) | NVTX Python bindings | When annotating Python code paths, but for CUDA kernels use C++ directly. |
| matplotlib/plotly | Benchmark GUI tools | For static reports; use NVIDIA Nsight Systems for interactive GPU analysis. |
| JSON output | CSV output | **Avoid CSV** - Google Benchmark docs note known parsing issues. Always use JSON. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Google Benchmark CSV output | Documented parsing issues in official docs | JSON output via `--benchmark_out_format=json` |
| CUDA event timing without `SetIterationTime()` | CPU timers don't account for GPU async execution | `UseManualTime()` + CUDA events + `SetIterationTime()` |
| pytest-benchmark | Designed for Python functions, not C++ binaries via subprocess | Direct pytest + subprocess pattern with JSON parsing |
| Raw NVTX C API (`nvtxRangePushA`) | Verbose, error-prone | nvtx3 C++ header-only library with RAII `scoped_range` |
| Python `time.time()` for GPU benchmarks | No GPU visibility, susceptible to CPU scaling | CUDA events + `cudaEventElapsedTime()` |

## Stack Patterns by Variant

**If single-GPU benchmarking:**
- Use `UseManualTime()` with CUDA events
- NVTX annotations without domain specification (default domain)
- Direct JSON output to file

**If multi-GPU/NCCL benchmarking:**
- Use NVTX domains per GPU rank: `nvtx3::mark_in<MyDomain>("message")`
- Capture NCCL collective operations with nested ranges
- Aggregate timing across ranks (check `NCCL_LAUNCH_MODE`)

**If CI regression testing:**
- Store baseline JSON in git or artifact storage
- Use `scipy.stats` for statistical comparison (Welch's t-test)
- Fail CI with threshold (e.g., >5% regression)

**If multi-node MPI benchmarking:**
- NVTX domain per MPI rank: `nvtx3::scoped_range range("collective_op", domain)` with MPI rank in message
- Synchronize baselines across nodes

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| Google Benchmark 1.9.x | CMake 3.16+, C++17 | Requires C++17 minimum for modern features. |
| Google Benchmark 1.9.x | CUDA 11.0+ | Works with any CUDA version that supports NVTX. |
| NVTX 3.x (nvtx3) | CUDA 11.0+ | Bundled with CUDA toolkit. Header-only, no linking needed. |
| pytest 8.x | Python 3.8+ | Ensure Python 3.8+ for modern async support. |
| pandas 2.x | Python 3.9+ | Required for nullable dtypes used in analysis. |
| scipy 1.14+ | Python 3.9+ | Required for `scipy.stats.permutation_test` (new in 1.14). |
| plotly 5.x | Any Python with JSON support | Browser-based rendering, no additional runtime needed. |

## Integration Notes for Existing Architecture

### Five-Layer Benchmark Strategy

Following your existing five-layer architecture (memory → device → algo → api → distributed):

1. **Memory layer benchmarks:** Focus on memory transfer timing (H2D/D2H/D2D)
2. **Device layer benchmarks:** Kernel execution timing with CUDA events
3. **Algo layer benchmarks:** Algorithm-specific throughput (FLOPs, items/sec)
4. **API layer benchmarks:** Multi-GPU collective operations
5. **Distributed layer benchmarks:** Multi-node MPI + NCCL collective timing

### Google Benchmark Integration Pattern

```cpp
// Custom CUDA timer registration
static void BM_KernelTiming(benchmark::State& state) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (auto _ : state) {
    cudaEventRecord(start);
    // CUDA kernel launch here
    your_kernel<<<blocks, threads>>>(...);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds / 1000.0);  // Convert to seconds
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
BENCHMARK(BM_KernelTiming)->UseManualTime();

// Custom counters for GPU metrics
static void BM_KernelWithCounters(benchmark::State& state) {
  int64_t items_processed = 0;
  for (auto _ : state) {
    // ... kernel execution ...
    items_processed += state.range(0);
  }
  state.counters["ItemsPerSec"] = benchmark::Counter(
    items_processed, benchmark::Counter::kIsRate);
}
```

### NVTX Annotation Pattern

```cpp
#include <nvtx3/nvtx3.hpp>

// Define domains matching your architecture layers
struct MemoryDomain { static constexpr char const* name{"Nova::Memory"}; };
struct DeviceDomain { static constexpr char const* name{"Nova::Device"}; };
struct AlgoDomain { static constexpr char const* name{"Nova::Algorithm"}; };

// Usage in kernels/benchmarks
void benchmark_memory_transfer() {
  nvtx3::scoped_range range{"H2D_Transfer", MemoryDomain{}};

  for (int i = 0; i < iterations; ++i) {
    nvtx3::mark_in<MemoryDomain>("transfer_start", nvtx3::payload{i});
    // Transfer code
    nvtx3::mark_in<MemoryDomain>("transfer_complete");
  }
}
```

### Python Harness Pattern

```python
# tests/benchmark_harness.py
import subprocess
import json
import pandas as pd
from pathlib import Path
import pytest
import scipy.stats as stats

BENCHMARK_BINARIES = [
    "bench_memory",
    "bench_device",
    "bench_algo",
    "bench_api",
    "bench_distributed",
]

def run_benchmark(binary: str, output_path: Path) -> dict:
    """Run a benchmark binary and save JSON output."""
    result = subprocess.run(
        [f"./build/{binary}", "--benchmark_out_format=json", f"--benchmark_out={output_path}"],
        capture_output=True, text=True, timeout=300
    )
    result.check_returncode()
    with open(output_path) as f:
        return json.load(f)

def load_baseline(name: str) -> dict:
    """Load baseline JSON from artifact storage or git."""
    path = Path(f"benchmarks/baselines/{name}_baseline.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

@pytest.mark.parametrize("benchmark", BENCHMARK_BINARIES)
def test_performance_regression(benchmark, tmp_path):
    output = tmp_path / f"{benchmark}_results.json"
    results = run_benchmark(benchmark, output)

    baseline = load_baseline(benchmark)
    if baseline is None:
        pytest.skip(f"No baseline for {benchmark}")

    # Compare iterations with same name
    for result in results["benchmarks"]:
        for base in baseline["benchmarks"]:
            if result["name"] == base["name"]:
                current = result["real_time"]
                prev = base["real_time"]
                change_pct = ((current - prev) / prev) * 100

                # Statistical significance test
                # Welch's t-test for unequal variances
                assert change_pct < 5.0, (
                    f"Regression detected for {result['name']}: "
                    f"{change_pct:+.2f}% (current: {current:.2f}us, "
                    f"baseline: {prev:.2f}us)"
                )
```

### HTML Dashboard Pattern

```python
# tools/benchmark_dashboard.py
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from chevron import render

def generate_dashboard(benchmark_results: Path, output: Path):
    with open(benchmark_results) as f:
        data = json.load(f)

    df = pd.DataFrame(data["benchmarks"])
    df["change_pct"] = 0.0  # Compute vs baseline

    # Create plotly figure
    fig = px.bar(df, x="name", y="real_time", color="change_pct",
                 color_continuous_scale=["green", "yellow", "red"],
                 title="Benchmark Results")

    # Save interactive HTML
    fig.write_html(output / "benchmark_chart.html")

    # Render mustache template
    with open("tools/dashboard_template.mustache") as f:
        html = render(f, {
            "benchmarks": df.to_dict("records"),
            "chart_path": "benchmark_chart.html",
            "generated": pd.Timestamp.now().isoformat(),
            "context": data["context"]
        })

    with open(output / "index.html", "w") as f:
        f.write(html)
```

```mustache
<!-- tools/dashboard_template.mustache -->
<!DOCTYPE html>
<html>
<head>
    <title>Nova Benchmark Dashboard</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Nova Performance Dashboard</h1>
    <p>Generated: {{generated}}</p>
    <p>CPUs: {{context.num_cpus}}, Build: {{context.build_type}}</p>

    <iframe src="{{chart_path}}" width="100%" height="600"></iframe>

    <table>
        <tr><th>Benchmark</th><th>Time (us)</th><th>Iterations</th></tr>
        {{#benchmarks}}
        <tr>
            <td>{{name}}</td>
            <td>{{real_time}}</td>
            <td>{{iterations}}</td>
        </tr>
        {{/benchmarks}}
    </table>
</body>
</html>
```

## Sources

- [/google/benchmark](https://context7.com/google/benchmark) — Custom timers, counters, JSON output, CMake integration
- [/nvidia/nvtx](https://context7.com/nvidia/nvtx) — C++ nvtx3 header-only library, scoped_range, domains
- [Google Benchmark User Guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md) — Official documentation on output formats
- [NVTX Python Reference](https://github.com/nvidia/nvtx/blob/release-v3/docs/python/reference.html) — Python bindings for NVTX
- [NVTX C++ Documentation](https://github.com/nvidia/nvtx/blob/release-v3/docs/doxygen-cpp/) — nvtx3.hpp API reference

---
*Stack research for: CUDA/C++ Benchmarking Infrastructure*
*Researched: 2026-04-26*
