#!/usr/bin/env python3
"""
Analyze and compare latency benchmark results.

Reads CSV files from benchmarks/results/ (scanning all session subdirectories)
and generates comparison tables, charts, and key findings.

Requires: pip install pandas matplotlib

Usage:
    # Analyze all sessions
    python benchmarks/analyze_results.py

    # Analyze a specific session
    python benchmarks/analyze_results.py --session 2026-02-17_c922_full_osc

    # Filter by keyword across all sessions
    python benchmarks/analyze_results.py --filter c922

    # Save charts as PNG
    python benchmarks/analyze_results.py --save

    # Only console output, no charts
    python benchmarks/analyze_results.py --no-charts

    # List available sessions
    python benchmarks/analyze_results.py --list-sessions
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd
import numpy as np

# Directories relative to this script
BENCHMARKS_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARKS_DIR / "results"
CHARTS_DIR = BENCHMARKS_DIR / "charts"


# =========================================================================
# LOAD DATA
# =========================================================================

def list_sessions() -> list[Path]:
    """List all session directories that contain CSV results."""
    if not RESULTS_DIR.exists():
        return []

    sessions = []
    for item in sorted(RESULTS_DIR.iterdir()):
        if item.is_dir() and list(item.glob("latency_raw_*.csv")):
            sessions.append(item)

    # Also check for flat CSVs directly in results/ (legacy layout)
    if list(RESULTS_DIR.glob("latency_raw_*.csv")):
        sessions.insert(0, RESULTS_DIR)

    return sessions


def load_all_raw(session: str = None,
                 name_filter: str = None) -> pd.DataFrame:
    """
    Load all latency_raw CSVs into a single DataFrame.

    Scans subdirectories recursively. Adds 'benchmark', 'session',
    and config metadata columns.

    Args:
        session: Only load from this session folder name.
        name_filter: Filter benchmarks by keyword in name.
    """
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    # Find all raw CSVs
    if session:
        session_path = RESULTS_DIR / session
        if not session_path.exists():
            print(f"Session not found: {session_path}")
            print(f"Available sessions:")
            for s in list_sessions():
                print(f"  {s.name}")
            sys.exit(1)
        raw_files = sorted(session_path.glob("latency_raw_*.csv"))
    else:
        raw_files = sorted(RESULTS_DIR.rglob("latency_raw_*.csv"))

    if not raw_files:
        print(f"No raw CSV files found in {RESULTS_DIR}/")
        sys.exit(1)

    frames = []
    for raw_path in raw_files:
        timestamp = raw_path.stem.replace("latency_raw_", "")
        summary_path = raw_path.parent / f"latency_summary_{timestamp}.csv"

        # Determine session name from parent directory
        if raw_path.parent == RESULTS_DIR:
            session_name = "(flat)"
        else:
            session_name = raw_path.parent.name

        # Extract benchmark name and config from summary
        meta = {}
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            for _, row in summary_df.iterrows():
                key = row["metric"]
                if key.startswith("config_"):
                    key = key[7:]
                meta[key] = row["value"]

        benchmark_name = meta.get("benchmark_name", raw_path.stem)

        # Apply filter
        if name_filter and name_filter.lower() not in benchmark_name.lower():
            continue

        # Load raw data
        df = pd.read_csv(raw_path)
        df["benchmark"] = benchmark_name
        df["session"] = session_name
        df["timestamp"] = timestamp

        # Add metadata columns for grouping
        df["backend"] = meta.get("backend", "?")
        df["camera"] = meta.get("camera_name", "unknown")
        df["resolution"] = meta.get("resolution", "unknown")
        df["pose_model"] = meta.get("pose_model_complexity", "?")
        df["output_mode"] = meta.get("output_mode", "?")
        df["osc_send_mode"] = meta.get("osc_send_mode", "")

        frames.append(df)

    if not frames:
        print(f"No benchmarks found" +
              (f" matching '{name_filter}'" if name_filter else ""))
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


# =========================================================================
# ANALYSIS
# =========================================================================

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table from raw frame data.

    Returns one row per benchmark with stats for each stage.
    """
    valid = df[df["pose_detected"] == True].copy()

    stage_cols = [c for c in valid.columns if c.endswith("_ms")]

    def agg_stats(group):
        stats = {}
        for col in stage_cols:
            vals = group[col].dropna()
            stats[f"{col}_mean"] = vals.mean()
            stats[f"{col}_p50"] = vals.quantile(0.50)
            stats[f"{col}_p95"] = vals.quantile(0.95)
            stats[f"{col}_p99"] = vals.quantile(0.99)
            stats[f"{col}_max"] = vals.max()

        stats["fps_mean"] = group["fps"].mean()
        stats["fps_min"] = group["fps"].min()
        stats["total_frames"] = len(df[df["benchmark"] == group.name])
        stats["valid_frames"] = len(group)
        stats["detection_pct"] = len(group) / len(
            df[df["benchmark"] == group.name]) * 100
        stats["pct_under_80ms"] = (group["total_ms"] <= 80).mean() * 100

        # Keep metadata from first row
        stats["backend"] = group["backend"].iloc[0]
        stats["camera"] = group["camera"].iloc[0]
        stats["resolution"] = group["resolution"].iloc[0]
        stats["pose_model"] = group["pose_model"].iloc[0]
        stats["output_mode"] = group["output_mode"].iloc[0]
        stats["session"] = group["session"].iloc[0]

        return pd.Series(stats)

    summary = valid.groupby("benchmark").apply(agg_stats, include_groups=False)
    summary = summary.sort_values("total_ms_mean")

    return summary


# =========================================================================
# CONSOLE OUTPUT
# =========================================================================

def print_comparison(summary: pd.DataFrame):
    """Print the main comparison table."""
    print(f"\n{'=' * 120}")
    print("  BENCHMARK COMPARISON (sorted by mean latency)")
    print(f"{'=' * 120}\n")

    show_session = summary["session"].nunique() > 1

    header = (f"  {'Name':<42} {'Backend':<8} {'Camera':<14} {'Res':<10} "
              f"{'Pose':<6} {'Out':<6} {'Mean':>8} {'P50':>8} {'P95':>8} "
              f"{'FPS':>6} {'<80ms':>7} {'Det':>5}")
    if show_session:
        header += f" {'Session':<24}"
    print(header)
    print("  " + "-" * (116 if show_session else 106))

    pose_labels = {"0": "Lite", "1": "Full", "2": "Heavy"}

    for name, row in summary.iterrows():
        pose = pose_labels.get(str(row["pose_model"]), str(row["pose_model"]))
        line = (f"  {name:<42} {row['backend']:<8} {row['camera']:<14} "
                f"{row['resolution']:<10} {pose:<6} {row['output_mode']:<6} "
                f"{row['total_ms_mean']:>7.1f}ms "
                f"{row['total_ms_p50']:>7.1f}ms "
                f"{row['total_ms_p95']:>7.1f}ms "
                f"{row['fps_mean']:>5.1f} "
                f"{row['pct_under_80ms']:>6.1f}% "
                f"{row['detection_pct']:>4.0f}%")
        if show_session:
            line += f" {row['session']:<24}"
        print(line)
    print()


def print_stage_breakdown(summary: pd.DataFrame):
    """Print per-stage mean latency breakdown."""
    stages = ["capture", "pose", "features", "send"]

    print(f"{'=' * 90}")
    print("  PER-STAGE BREAKDOWN (mean ms)")
    print(f"{'=' * 90}\n")

    print(f"  {'Name':<32} {'Capture':>10} {'Pose':>10} "
          f"{'Features':>10} {'Send':>10} {'Total':>10}")
    print("  " + "-" * 82)

    for name, row in summary.iterrows():
        vals = [f"{row.get(f'{s}_ms_mean', 0):>8.1f}ms" for s in stages]
        total = f"{row['total_ms_mean']:>8.1f}ms"
        print(f"  {name:<32} {vals[0]} {vals[1]} {vals[2]} {vals[3]} {total}")

    # Percentage breakdown
    print(f"\n  {'Name':<32} {'Capture':>10} {'Pose':>10} "
          f"{'Features':>10} {'Send':>10}")
    print("  " + "-" * 72)

    for name, row in summary.iterrows():
        total = row["total_ms_mean"]
        if total > 0:
            pcts = [
                f"{row.get(f'{s}_ms_mean', 0) / total * 100:>9.0f}%"
                for s in stages
            ]
            print(f"  {name:<32} {pcts[0]} {pcts[1]} {pcts[2]} {pcts[3]}")

    print()


def print_findings(df: pd.DataFrame, summary: pd.DataFrame):
    """Print automated key findings."""
    print(f"{'=' * 65}")
    print("  KEY FINDINGS")
    print(f"{'=' * 65}\n")

    valid = df[df["pose_detected"] == True]

    # Best / worst
    best = summary.index[0]
    worst = summary.index[-1]
    print(f"  Fastest:  {best} ({summary.loc[best, 'total_ms_mean']:.1f}ms)")
    print(f"  Slowest:  {worst} ({summary.loc[worst, 'total_ms_mean']:.1f}ms)")
    diff = summary["total_ms_mean"].max() - summary["total_ms_mean"].min()
    print(f"  Range:    {diff:.1f}ms\n")

    # Target compliance
    all_pass = (summary["pct_under_80ms"] == 100).all()
    if all_pass:
        print("  All configurations meet the 80ms target.\n")
    else:
        failing = summary[summary["pct_under_80ms"] < 100]
        for name, row in failing.iterrows():
            print(f"  WARNING: {name} — {row['pct_under_80ms']:.1f}% under 80ms\n")

    # Pose model impact
    pose_means = valid.groupby("pose_model")["total_ms"].mean()
    if len(pose_means) > 1:
        pose_labels = {"0": "Lite", "1": "Full", "2": "Heavy"}
        print("  Pose model impact (mean total ms):")
        for model, mean in pose_means.sort_index().items():
            n = valid[valid["pose_model"] == model]["benchmark"].nunique()
            label = pose_labels.get(str(model), str(model))
            print(f"    {label} (complexity={model}): {mean:.1f}ms  "
                  f"({n} configs)")
        if "0" in pose_means.index and "1" in pose_means.index:
            diff_val = pose_means["1"] - pose_means["0"]
            print(f"    → Full costs {diff_val:+.1f}ms vs Lite")
        if "1" in pose_means.index and "2" in pose_means.index:
            diff_val = pose_means["2"] - pose_means["1"]
            print(f"    → Heavy costs {diff_val:+.1f}ms vs Full")
        print()

    # Camera comparison
    cam_means = valid.groupby("camera")["total_ms"].mean()
    if len(cam_means) > 1:
        print("  Camera comparison (mean total ms):")
        for cam, mean in cam_means.items():
            n = valid[valid["camera"] == cam]["benchmark"].nunique()
            print(f"    {cam}: {mean:.1f}ms  ({n} configs)")
        print()

    # Output mode comparison
    out_means = valid.groupby("output_mode")["total_ms"].mean()
    if len(out_means) > 1:
        print("  Output mode comparison (mean total ms):")
        for mode, mean in out_means.items():
            n = valid[valid["output_mode"] == mode]["benchmark"].nunique()
            print(f"    {mode}: {mean:.1f}ms  ({n} configs)")
        print()

    # Backend comparison
    backend_means = valid.groupby("backend")["total_ms"].mean()
    if len(backend_means) > 1:
        print("  Backend comparison (mean total ms):")
        for backend, mean in backend_means.items():
            n = valid[valid["backend"] == backend]["benchmark"].nunique()
            print(f"    {backend}: {mean:.1f}ms  ({n} configs)")
        if "cpu" in backend_means.index and "metal" in backend_means.index:
            diff_val = backend_means["metal"] - backend_means["cpu"]
            print(f"    → Metal vs CPU: {diff_val:+.1f}ms")
        if "cpu" in backend_means.index and "tensorrt" in backend_means.index:
            diff_val = backend_means["tensorrt"] - backend_means["cpu"]
            print(f"    → TensorRT vs CPU: {diff_val:+.1f}ms")
        print()

    # Bottleneck analysis
    stages = ["capture_ms", "pose_ms", "features_ms", "send_ms"]
    available = [s for s in stages if s in valid.columns]
    if available:
        stage_means = valid[available].mean()
        total = stage_means.sum()
        bottleneck = stage_means.idxmax().replace("_ms", "")
        pct = stage_means.max() / total * 100
        print(f"  Bottleneck: {bottleneck} ({pct:.0f}% of pipeline time)\n")


# =========================================================================
# CHARTS
# =========================================================================

def generate_charts(df: pd.DataFrame, summary: pd.DataFrame,
                    save: bool = False):
    """Generate publication-quality charts for the TFG."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        if save:
            matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not installed. Skipping charts.")
        return

    valid = df[df["pose_detected"] == True]
    names = summary.index.tolist()

    colors = {
        "capture": "#4C72B0",
        "pose": "#DD8452",
        "features": "#55A868",
        "send": "#C44E52",
    }
    stages = ["capture", "pose", "features", "send"]

    if save:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Chart 1: Stacked bar (per-stage mean) ---

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))

    bottom = np.zeros(len(names))
    for stage in stages:
        col = f"{stage}_ms_mean"
        vals = summary[col].values
        ax.bar(names, vals, bottom=bottom, label=stage.capitalize(),
               color=colors[stage], width=0.6)
        bottom += vals

    ax.axhline(y=80, color="red", linestyle="--", alpha=0.7,
               label="80ms target")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Cuerpo Sonoro — Per-Stage Latency by Configuration")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    if save:
        path = CHARTS_DIR / "chart_stages.png"
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
        plt.close()
    else:
        plt.show()

    # --- Chart 2: Box plot of total latency per benchmark ---

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))

    box_data = [
        valid[valid["benchmark"] == name]["total_ms"].values
        for name in names
    ]

    bp = ax.boxplot(box_data, labels=names, patch_artist=True,
                    showfliers=True, flierprops=dict(markersize=3, alpha=0.5))

    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.7)

    ax.axhline(y=80, color="red", linestyle="--", alpha=0.7,
               label="80ms target")
    ax.set_ylabel("Total Latency (ms)")
    ax.set_title("Cuerpo Sonoro — Latency Distribution by Configuration")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    if save:
        path = CHARTS_DIR / "chart_boxplot.png"
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
        plt.close()
    else:
        plt.show()

    # --- Chart 3: Pose model comparison (grouped bar) ---

    pose_labels = {"0": "Lite", "1": "Full", "2": "Heavy"}
    pose_models = sorted(valid["pose_model"].unique(),
                         key=lambda x: str(x))

    if len(pose_models) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))

        pose_stage_means = (
            valid.groupby("pose_model")[["capture_ms", "pose_ms",
                                         "features_ms", "send_ms"]]
            .mean()
        )

        x = np.arange(len(pose_models))
        width = 0.18

        for i, stage in enumerate(stages):
            col = f"{stage}_ms"
            vals = [pose_stage_means.loc[m, col] for m in pose_models]
            ax.bar(x + i * width, vals, width, label=stage.capitalize(),
                   color=colors[stage])

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([pose_labels.get(str(m), str(m))
                            for m in pose_models])
        ax.set_ylabel("Mean Latency (ms)")
        ax.set_title("Cuerpo Sonoro — Pose Model Complexity Impact")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save:
            path = CHARTS_DIR / "chart_pose_comparison.png"
            plt.savefig(path, dpi=150)
            print(f"  Saved: {path}")
            plt.close()
        else:
            plt.show()


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Cuerpo Sonoro benchmark results"
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Analyze only this session folder (e.g. '2026-02-17_c922_full_osc')"
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Filter benchmarks by keyword (e.g. 'c922', 'full')"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save charts as PNG to benchmarks/charts/"
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation"
    )
    parser.add_argument(
        "--list-sessions", action="store_true",
        help="List available sessions and exit"
    )
    args = parser.parse_args()

    # List sessions mode
    if args.list_sessions:
        sessions = list_sessions()
        if not sessions:
            print("No sessions found.")
        else:
            print(f"\n  Available sessions in {RESULTS_DIR}:\n")
            for s in sessions:
                n_raw = len(list(s.glob("latency_raw_*.csv")))
                label = "(flat — legacy)" if s == RESULTS_DIR else s.name
                print(f"    {label}  ({n_raw} benchmarks)")
        print()
        sys.exit(0)

    # Load data
    df = load_all_raw(session=args.session, name_filter=args.filter)

    sessions_loaded = df["session"].nunique()
    print(f"\n  Loaded {len(df)} frames from "
          f"{df['benchmark'].nunique()} benchmarks "
          f"across {sessions_loaded} session(s).\n")

    # Build summary
    summary = build_summary(df)

    # Console output
    print_comparison(summary)
    print_stage_breakdown(summary)
    print_findings(df, summary)

    # Charts
    if not args.no_charts:
        generate_charts(df, summary, save=args.save)


if __name__ == "__main__":
    main()
