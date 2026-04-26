#!/usr/bin/env python3
"""
Nova Benchmark Harness

Runs CUDA benchmarks and collects results for regression testing.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_BUILD_DIR = Path(__file__).parent.parent / "build"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "results"


def get_args():
    parser = argparse.ArgumentParser(description="Nova CUDA Benchmark Harness")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="Build directory containing benchmark binaries",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write benchmark results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["standard", "quick", "regression"],
        help="Benchmark configuration to use",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="*",
        help="Benchmark filter (e.g., '*Reduce*', '*Matmul*')",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device to use",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks and exit",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Compare against baseline version (e.g., 'v1.7.0')",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Regression tolerance percentage",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def get_benchmark_binaries(build_dir: Path) -> list[Path]:
    """Find all benchmark binaries in the build directory."""
    if not build_dir.exists():
        return []

    binaries = []
    for bin_dir in ["bin", "tests"]:
        bin_path = build_dir / bin_dir
        if bin_path.exists():
            for f in bin_path.iterdir():
                if f.is_file() and (
                    f.name.startswith("benchmark_") or f.name == "nova_benchmarks"
                ):
                    binaries.append(f)
    return binaries


def get_device_info() -> dict:
    """Get GPU device information using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(", ")
                return {
                    "gpu": parts[0] if len(parts) > 0 else "Unknown",
                    "driver": parts[1] if len(parts) > 1 else "Unknown",
                    "memory": parts[2] if len(parts) > 2 else "Unknown",
                }
    except Exception:
        pass
    return {"gpu": "Unknown", "driver": "Unknown", "memory": "Unknown"}


def run_benchmark(
    binary: Path,
    filter_pattern: str = "*",
    gpu: int = 0,
    verbose: bool = False,
) -> dict | None:
    """Run a single benchmark binary and return results."""
    if not binary.exists():
        return None

    args = [
        str(binary),
        f"--benchmark_filter={filter_pattern}",
        f"--benchmark_format=json",
        f"--benchmark_out_format=json",
        f"--benchmark_out={binary.parent / 'benchmark_results.json'}",
        f"--cuda_device={gpu}",
    ]

    if verbose:
        print(f"Running: {' '.join(args)}")

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=binary.parent,
        )

        if result.stdout:
            if verbose:
                print(result.stdout[:500])

        results_file = binary.parent / "benchmark_results.json"
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
        else:
            if result.returncode != 0:
                print(f"Error running {binary.name}: {result.stderr[:200]}", file=sys.stderr)
            return None

    except subprocess.TimeoutExpired:
        print(f"Timeout running {binary.name}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error running {binary.name}: {e}", file=sys.stderr)
        return None


def load_baseline(baseline_version: str) -> dict | None:
    """Load baseline results for comparison."""
    baseline_dir = Path(__file__).parent / "baselines" / baseline_version
    if not baseline_dir.exists():
        print(f"Baseline not found: {baseline_dir}", file=sys.stderr)
        return None

    baseline_results = {"benchmarks": []}
    for json_file in baseline_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                if "benchmarks" in data:
                    baseline_results["benchmarks"].extend(data["benchmarks"])
        except Exception as e:
            print(f"Error loading baseline {json_file}: {e}", file=sys.stderr)

    return baseline_results if baseline_results["benchmarks"] else None


def compare_results(current: dict, baseline: dict, tolerance: float) -> dict:
    """Compare current results against baseline."""
    comparison = {"regressions": [], "improvements": [], "stable": []}

    if "benchmarks" not in baseline:
        return comparison

    baseline_map = {b["name"]: b for b in baseline["benchmarks"]}

    for result in current.get("benchmarks", []):
        name = result.get("name", "")
        if name not in baseline_map:
            continue

        base = baseline_map[name]
        current_time = result.get("real_time", 0)
        base_time = base.get("real_time", 0)

        if base_time > 0:
            delta_pct = ((current_time - base_time) / base_time) * 100

            entry = {
                "name": name,
                "current": current_time,
                "baseline": base_time,
                "delta_pct": delta_pct,
            }

            if delta_pct > tolerance:
                comparison["regressions"].append(entry)
            elif delta_pct < -tolerance:
                comparison["improvements"].append(entry)
            else:
                comparison["stable"].append(entry)

    return comparison


def save_results(output_dir: Path, results: dict, config: str) -> Path:
    """Save benchmark results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_dir / f"benchmark_results_{config}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return output_file


def format_results_table(results: dict, comparison: dict | None = None) -> str:
    """Format results as a table."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("BENCHMARK RESULTS")
    lines.append("=" * 80)

    benchmarks = results.get("benchmarks", [])
    if not benchmarks:
        lines.append("\nNo benchmark results.")
        return "\n".join(lines)

    lines.append(f"\n{'Name':<50} {'Time (ms)':<12} {'Throughput':<15}")
    lines.append("-" * 80)

    for bench in benchmarks:
        name = bench.get("name", "")[:48]
        time_ms = bench.get("real_time", 0)
        bps = bench.get("bytes_per_second", 0)
        throughput = f"{bps / 1e9:.2f} GB/s" if bps > 0 else "-"
        lines.append(f"{name:<50} {time_ms:<12.4f} {throughput:<15}")

    if comparison:
        lines.append("\n" + "=" * 80)
        lines.append("REGRESSION COMPARISON")
        lines.append("=" * 80)

        if comparison["regressions"]:
            lines.append("\n⚠ REGRESSIONS:")
            for r in comparison["regressions"]:
                lines.append(
                    f"  {r['name'][:48]:<48} {r['delta_pct']:+.2f}% "
                    f"(current: {r['current']:.4f}ms, baseline: {r['baseline']:.4f}ms)"
                )

        if comparison["improvements"]:
            lines.append("\n✓ IMPROVEMENTS:")
            for r in comparison["improvements"]:
                lines.append(
                    f"  {r['name'][:48]:<48} {r['delta_pct']:+.2f}% "
                    f"(current: {r['current']:.4f}ms, baseline: {r['baseline']:.4f}ms)"
                )

        if comparison["stable"]:
            lines.append(f"\n~ STABLE: {len(comparison['stable'])} benchmarks within tolerance")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main():
    args = get_args()

    device_info = get_device_info()
    if args.verbose:
        print(f"GPU: {device_info['gpu']}")
        print(f"Driver: {device_info['driver']}")
        print(f"Memory: {device_info['memory']}")

    binaries = get_benchmark_binaries(args.build_dir)
    if not binaries:
        print(f"No benchmark binaries found in {args.build_dir}", file=sys.stderr)
        print("Run 'cmake --build build' first to build benchmarks.", file=sys.stderr)
        return 1

    if args.list:
        print("Available benchmark binaries:")
        for b in binaries:
            print(f"  - {b.name}")
        return 0

    filter_pattern = "*" if args.all else args.filter

    all_results = {
        "context": {
            "date": datetime.now().isoformat(),
            "config": args.config,
            "filter": filter_pattern,
            "gpu": device_info["gpu"],
            "driver": device_info["driver"],
            "memory": device_info["memory"],
        },
        "benchmarks": [],
    }

    for binary in binaries:
        print(f"Running {binary.name}...", end=" ", flush=True)
        result = run_benchmark(binary, filter_pattern, args.gpu, args.verbose)
        if result and "benchmarks" in result:
            all_results["benchmarks"].extend(result["benchmarks"])
            print(f"✓ {len(result['benchmarks'])} benchmarks")
        else:
            print("✗")

    if not all_results["benchmarks"]:
        print("No benchmark results collected.", file=sys.stderr)
        return 1

    baseline_results = None
    comparison = None
    if args.baseline:
        print(f"\nLoading baseline: {args.baseline}")
        baseline_results = load_baseline(args.baseline)
        if baseline_results:
            comparison = compare_results(all_results, baseline_results, args.tolerance)

    output_file = save_results(args.output_dir, all_results, args.config)
    print(f"\nResults saved to: {output_file}")

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        print(format_results_table(all_results, comparison))

    if comparison and comparison["regressions"]:
        print(f"\n⚠ {len(comparison['regressions'])} regression(s) detected!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
