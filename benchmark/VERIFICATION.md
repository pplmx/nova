# Benchmark Infrastructure Verification

## Quick Verification

After building, verify Phase 29 implementation:

### 1. NVTX Header with Toggle

```bash
# With NVTX enabled (overhead present)
grep -l "NOVA_NVTX_ENABLED" build/*.cmake 2>/dev/null || \
  cmake -B build -DNOVA_NVTX_ENABLED=1 && cmake --build build --target nova_benchmarks

# With NVTX disabled (zero overhead)
cmake -B build -DNOVA_NVTX_ENABLED=0 && cmake --build build --target nova_benchmarks
```

### 2. Python Harness

```bash
# List available benchmarks
python scripts/benchmark/run_benchmarks.py --list

# Run all benchmarks
python scripts/benchmark/run_benchmarks.py --all --verbose

# Run specific benchmark
python scripts/benchmark/run_benchmarks.py --filter "*Reduce*"
```

### 3. Verify Requirements

| Requirement | Verification |
|-------------|--------------|
| BENCH-01 | `python scripts/benchmark/run_benchmarks.py --all` completes |
| BENCH-02 | CUDA events used in `include/cuda/benchmark/benchmark.h:66-84` |
| BENCH-03 | Warmup iterations in `benchmark.h:128-131` |
| BENCH-04 | NVTX toggle in `include/cuda/benchmark/nvtx.h:16-20` |
| BENCH-05 | NVTX macros are no-ops when disabled (`nvtx.h:130-160`) |

### 4. Expected Output

```
GPU: NVIDIA A100-SXM4-40GB
Driver: 535.54.03
Memory: 40.0 GB

Running benchmark_kernels... ✓ 8 benchmarks
Running nova_benchmarks... ✓ 12 benchmarks

Results saved to: results/benchmark_results_standard_2026-04-26.json

================================================================================
BENCHMARK RESULTS
================================================================================

Name                                              Time (ms)    Throughput
--------------------------------------------------------------------------------
BM_MemoryH2D/1024                                 0.0012       0.85 GB/s
BM_MemoryH2D/65536                                0.0045       14.62 GB/s
...
================================================================================
```

## CI Integration

To add to GitHub Actions:

```yaml
- name: Run benchmarks
  run: |
    python scripts/benchmark/run_benchmarks.py --all --baseline v1.7.0 --tolerance 10
```

## Notes

- First run will be slower due to JIT compilation
- GPU frequency scaling affects results; use warmup
- For stable baselines, run on dedicated GPU hardware
