#!/usr/bin/env python3
"""
Generate HTML performance dashboard from benchmark results.

Usage:
    python scripts/benchmark/generate_dashboard.py --results results/latest --output reports/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Generate performance dashboard")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "reports",
        help="Output directory for dashboard",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Baseline version to compare against",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=Path(__file__).parent / "baselines",
        help="Directory containing baselines",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> dict | None:
    """Load the most recent benchmark results."""
    json_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        return None

    with open(json_files[0]) as f:
        return json.load(f)


def load_baseline(baseline_version: str, baselines_dir: Path) -> dict | None:
    """Load baseline results for comparison."""
    baseline_dir = baselines_dir / baseline_version
    if not baseline_dir.exists():
        return None

    baseline_results = {"benchmarks": []}
    for json_file in baseline_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                if "benchmarks" in data:
                    baseline_results["benchmarks"].extend(data["benchmarks"])
        except Exception:
            pass

    return baseline_results if baseline_results["benchmarks"] else None


def compute_comparison(current: dict, baseline: dict, tolerance: float = 10.0) -> list:
    """Compute comparison between current and baseline."""
    if not baseline or "benchmarks" not in baseline:
        return []

    baseline_map = {b["name"]: b for b in baseline["benchmarks"]}
    comparison = []

    for result in current.get("benchmarks", []):
        name = result.get("name", "")
        if name not in baseline_map:
            continue

        base = baseline_map[name]
        current_time = result.get("real_time", 0)
        base_time = base.get("real_time", 0)

        if base_time > 0:
            delta_pct = ((current_time - base_time) / base_time) * 100

            if delta_pct > tolerance:
                status = "regression"
                status_color = "#ef4444"
            elif delta_pct < -tolerance:
                status = "improvement"
                status_color = "#22c55e"
            else:
                status = "stable"
                status_color = "#6b7280"

            comparison.append({
                "name": name,
                "current": current_time,
                "baseline": base_time,
                "delta_pct": delta_pct,
                "status": status,
                "status_color": status_color,
                "throughput": result.get("bytes_per_second", 0),
            })

    return comparison


def generate_chart_data(comparison: list) -> dict:
    """Generate chart data for plotly."""
    names = [c["name"][:40] for c in comparison]
    currents = [c["current"] for c in comparison]
    baselines = [c["baseline"] for c in comparison]
    deltas = [c["delta_pct"] for c in comparison]

    return {
        "names": names,
        "currents": currents,
        "baselines": baselines,
        "deltas": deltas,
    }


def generate_html(
    results: dict,
    comparison: list,
    output_dir: Path,
    baseline_version: str | None = None,
):
    """Generate HTML dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)

    context = results.get("context", {})
    benchmarks = results.get("benchmarks", [])

    regression_count = sum(1 for c in comparison if c["status"] == "regression")
    improvement_count = sum(1 for c in comparison if c["status"] == "improvement")
    stable_count = sum(1 for c in comparison if c["status"] == "stable")

    chart_data = generate_chart_data(comparison)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nova Performance Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f9fafb; color: #1f2937; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        header {{ background: white; border-bottom: 1px solid #e5e7eb; padding: 1.5rem 2rem; margin-bottom: 2rem; border-radius: 8px; }}
        h1 {{ font-size: 1.5rem; font-weight: 600; color: #111827; }}
        .subtitle {{ color: #6b7280; margin-top: 0.25rem; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .stat-card {{ background: white; padding: 1.5rem; border-radius: 8px; border: 1px solid #e5e7eb; }}
        .stat-value {{ font-size: 2rem; font-weight: 700; }}
        .stat-label {{ color: #6b7280; font-size: 0.875rem; margin-top: 0.25rem; }}
        .stat-card.regression .stat-value {{ color: #ef4444; }}
        .stat-card.improvement .stat-value {{ color: #22c55e; }}
        .stat-card.stable .stat-value {{ color: #6b7280; }}
        .card {{ background: white; border-radius: 8px; border: 1px solid #e5e7eb; margin-bottom: 1.5rem; overflow: hidden; }}
        .card-header {{ padding: 1rem 1.5rem; border-bottom: 1px solid #e5e7eb; font-weight: 600; }}
        .card-body {{ padding: 1.5rem; }}
        .chart {{ width: 100%; height: 500px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 0.75rem 1rem; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f9fafb; font-weight: 600; font-size: 0.875rem; color: #6b7280; text-transform: uppercase; }}
        tr:hover {{ background: #f9fafb; }}
        .status-badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500; }}
        .status-badge.regression {{ background: #fef2f2; color: #ef4444; }}
        .status-badge.improvement {{ background: #f0fdf4; color: #22c55e; }}
        .status-badge.stable {{ background: #f3f4f6; color: #6b7280; }}
        .context-info {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.5rem; font-size: 0.875rem; color: #6b7280; }}
        footer {{ text-align: center; color: #9ca3af; font-size: 0.875rem; margin-top: 2rem; padding-top: 2rem; border-top: 1px solid #e5e7eb; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Nova Performance Dashboard</h1>
            <p class="subtitle">Benchmark regression tracking and analysis</p>
        </header>

        <div class="stats">
            <div class="stat-card regression">
                <div class="stat-value">{regression_count}</div>
                <div class="stat-label">Regressions</div>
            </div>
            <div class="stat-card improvement">
                <div class="stat-value">{improvement_count}</div>
                <div class="stat-label">Improvements</div>
            </div>
            <div class="stat-card stable">
                <div class="stat-value">{stable_count}</div>
                <div class="stat-label">Stable</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(benchmarks)}</div>
                <div class="stat-label">Total Benchmarks</div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Comparison: Current vs Baseline</div>
            <div class="card-body">
                <div id="chart" class="chart"></div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Detailed Results</div>
            <div class="card-body">
                <table>
                    <thead>
                        <tr>
                            <th>Benchmark</th>
                            <th>Current (ms)</th>
                            <th>Baseline (ms)</th>
                            <th>Delta %</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for c in comparison:
        html_content += f"""
                        <tr>
                            <td>{c['name']}</td>
                            <td>{c['current']:.4f}</td>
                            <td>{c['baseline']:.4f}</td>
                            <td>{c['delta_pct']:+.2f}%</td>
                            <td><span class="status-badge {c['status']}">{c['status']}</span></td>
                        </tr>
"""

    html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Environment</div>
            <div class="card-body">
                <div class="context-info">
                    <div>GPU: {context.get('gpu', 'Unknown')}</div>
                    <div>Driver: {context.get('driver', 'Unknown')}</div>
                    <div>Memory: {context.get('memory', 'Unknown')}</div>
                    <div>Date: {context.get('date', datetime.now().isoformat())}</div>
                    <div>Config: {context.get('config', 'N/A')}</div>
                    <div>Baseline: {baseline_version or 'N/A'}</div>
                </div>
            </div>
        </div>

        <footer>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Nova CUDA Library - Benchmark Dashboard</p>
        </footer>
    </div>

    <script>
        var data = [
            {{
                x: {json.dumps(chart_data['names'])},
                y: {json.dumps(chart_data['currents'])},
                name: 'Current',
                type: 'bar',
                marker: {{ color: '#3b82f6' }}
            }},
            {{
                x: {json.dumps(chart_data['names'])},
                y: {json.dumps(chart_data['baselines'])},
                name: 'Baseline',
                type: 'bar',
                marker: {{ color: '#9ca3af' }}
            }}
        ];

        var layout = {{
            barmode: 'group',
            xaxis: {{ title: 'Benchmark', tickangle: -45, tickfont: {{ size: 10 }} }},
            yaxis: {{ title: 'Time (ms)', titlefont: {{ size: 12 }} }},
            margin: {{ b: 120, l: 80, r: 40, t: 40 }},
            legend: {{ x: 0.5, y: 1.15, xanchor: 'center', orientation: 'h' }},
            font: {{ family: '-apple-system, BlinkMacSystemFont, sans-serif' }}
        }};

        Plotly.newPlot('chart', data, layout, {{responsive: true}});
    </script>
</body>
</html>
"""

    dashboard_file = output_dir / "index.html"
    with open(dashboard_file, "w") as f:
        f.write(html_content)

    return dashboard_file


def main():
    args = get_args()

    if not args.results.exists():
        print(f"Results directory not found: {args.results}")
        return 1

    print(f"Loading results from: {args.results}")
    results = load_results(args.results)
    if not results:
        print("No results found")
        return 1

    baseline = None
    if args.baseline:
        print(f"Loading baseline: {args.baseline}")
        baseline = load_baseline(args.baseline, args.baselines_dir)

    comparison = compute_comparison(results, baseline) if baseline else []

    print(f"Generating dashboard to: {args.output}")
    dashboard_file = generate_html(results, comparison, args.output, args.baseline)

    print(f"Dashboard generated: {dashboard_file}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
