#!/usr/bin/env python3
"""
Latency benchmark for Cuerpo Sonoro pipeline.

Generates all meaningful combinations of camera × resolution × pose model ×
output mode from config.yaml, and measures per-stage latency for each.

Runs WITHOUT OpenCV display to avoid contaminating measurements.

Usage:
    # Run all combinations
    python benchmarks/run_benchmark.py

    # Run only a specific camera
    python benchmarks/run_benchmark.py --camera-profile c922

    # Filter by output mode or pose model
    python benchmarks/run_benchmark.py --output osc --pose full

    # Custom session name
    python benchmarks/run_benchmark.py --session-name "pre-defense-final"

    # Custom frame count
    python benchmarks/run_benchmark.py --frames 500

    # List all combinations without running
    python benchmarks/run_benchmark.py --list

    # Show camera preview before each benchmark
    python benchmarks/run_benchmark.py --preview

Output:
    benchmarks/results/<session>/latency_raw_*.csv
    benchmarks/results/<session>/latency_summary_*.csv

    Session folder names are auto-generated from filters and date, e.g.:
        2026-02-17_c922_full_osc
        2026-02-17_all
        2026-02-17_pre-defense-final   (with --session-name)
"""

import sys
import os
import argparse
import platform
import statistics
from datetime import datetime

sys.path.insert(0, ".")

import cv2
from vision_processor.config import Config
from vision_processor.pose import PoseEstimator
from vision_processor.features import FeatureExtractor
from vision_processor.latency_logger import LatencyLogger

# Base directory for all benchmark results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# =========================================================================
# SESSION NAMING
# =========================================================================

def build_session_name(camera_filter: str = None,
                       output_filter: str = None,
                       pose_filter: str = None,
                       backend_filter: list[str] | None = None,
                       custom_name: str = None) -> str:
    """
    Generate a descriptive session folder name.

    Examples:
        No filters:                  2026-02-17_all
        --camera-profile c922:       2026-02-17_c922
        --pose full --output osc:    2026-02-17_full_osc
        --backend metal:             2026-02-17_metal
        --backend cpu metal:         2026-02-17_cpu-metal
        --session-name baseline:     2026-02-17_baseline
    """
    date = datetime.now().strftime("%Y-%m-%d")

    if custom_name:
        return f"{date}_{custom_name}"

    parts = []
    if backend_filter:
        parts.append("-".join(backend_filter))
    if camera_filter:
        parts.append(camera_filter)
    if pose_filter:
        parts.append(pose_filter)
    if output_filter:
        parts.append(output_filter)

    suffix = "_".join(parts) if parts else "all"
    return f"{date}_{suffix}"


def ensure_unique_session(base_path: str) -> str:
    """
    If the session folder already exists, append a counter.

    benchmarks/results/2026-02-17_all
    benchmarks/results/2026-02-17_all_2
    benchmarks/results/2026-02-17_all_3
    """
    if not os.path.exists(base_path):
        return base_path

    counter = 2
    while os.path.exists(f"{base_path}_{counter}"):
        counter += 1

    return f"{base_path}_{counter}"


# =========================================================================
# CAMERA PREVIEW
# =========================================================================

def run_preview(cap, pose_estimator, combo_name: str, timeout: int = 10):
    """
    Show a live camera preview with skeleton overlay.

    Lets the user confirm they're visible and properly positioned
    before the headless benchmark starts.

    Args:
        cap: Opened cv2.VideoCapture.
        pose_estimator: Initialized PoseEstimator.
        combo_name: Name of the current benchmark (shown on screen).
        timeout: Auto-close after this many seconds if no key pressed.
    """
    import time

    window_name = f"Preview — {combo_name}"
    print(f"  PREVIEW: Position yourself in front of the camera.")
    print(f"  Press 'q' or SPACE to start benchmark (auto-starts in {timeout}s)")

    start_time = time.time()

    while True:
        frame = cap.read()
        if frame is None:
            break

        # Run pose detection and draw skeleton
        results = pose_estimator.estimate(frame)
        frame = pose_estimator.draw_skeleton(frame, results)
        has_pose = results and results.pose_landmarks

        # Status overlay
        elapsed = time.time() - start_time
        remaining = max(0, timeout - int(elapsed))

        status = "POSE OK" if has_pose else "NO POSE — move into frame"
        color = (0, 255, 0) if has_pose else (0, 0, 255)

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Starting in {remaining}s (press q/SPACE to start now)",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, combo_name, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(window_name, frame)

        # Exit on key press or timeout
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord(' '), 27):  # q, space, or ESC
            break
        if elapsed >= timeout:
            break

    cv2.destroyWindow(window_name)
    # Small delay to let the window actually close before headless run
    cv2.waitKey(100)


# =========================================================================
# COMBINATION MATRIX
# =========================================================================

def generate_combinations(config: Config,
                          camera_filter: str = None,
                          output_filter: str = None,
                          pose_filter: str = None,
                          backend_filter: list[str] | None = None) -> list[dict]:
    """
    Generate all benchmark combinations from config.yaml.

    Each combination is a dict with all the parameters needed to
    configure and run one benchmark test.
    """
    cameras = config.camera_profiles
    resolutions = config.benchmark_resolutions
    pose_models = config.benchmark_pose_models
    output_modes = config.benchmark_output_modes
    backends = config.benchmark_backends

    # Apply filters
    if camera_filter:
        cameras = {k: v for k, v in cameras.items() if camera_filter in k}
    if output_filter:
        output_modes = [m for m in output_modes if output_filter in m["name"]]
    if pose_filter:
        pose_models = [p for p in pose_models if pose_filter in p["name"]]
    if backend_filter:
        backends = [b for b in backends if b["name"] in backend_filter]

    combinations = []

    for backend in backends:
        for cam_key, cam_profile in cameras.items():
            for res in resolutions:
                for pose in pose_models:
                    for output in output_modes:
                        name = (
                            f"{backend['name']}_"
                            f"{cam_key}_"
                            f"{res['name']}_"
                            f"{pose['name']}_"
                            f"{output['name']}"
                        )

                        combinations.append({
                            "name": name,
                            "backend": backend["name"],
                            "camera_profile": cam_key,
                            "camera_name": cam_profile.get("name", cam_key),
                            "camera_device_id": cam_profile["device_id"],
                            "width": res["width"],
                            "height": res["height"],
                            "resolution_name": res["name"],
                            "model_complexity": pose["model_complexity"],
                            "pose_name": pose["name"],
                            "output_mode": output["mode"],
                            "send_mode": output.get("send_mode"),
                            "output_name": output["name"],
                        })

    return combinations


def print_combination_table(combinations: list):
    """Print a formatted table of all combinations."""
    print(f"\n  {'#':<4} {'Name':<42} {'Backend':<8} {'Camera':<18} {'Res':<8} "
          f"{'Pose':<8} {'Output':<12}")
    print("  " + "-" * 102)

    for i, c in enumerate(combinations, 1):
        print(f"  {i:<4} {c['name']:<42} {c['backend']:<8} "
              f"{c['camera_name']:<18} {c['resolution_name']:<8} "
              f"{c['pose_name']:<8} {c['output_name']:<12}")


# =========================================================================
# BENCHMARK RUNNER
# =========================================================================

def run_single_benchmark(combo: dict, config: Config,
                         num_frames: int, warmup_frames: int,
                         session_dir: str,
                         preview: bool = False) -> LatencyLogger:
    """
    Run a single benchmark combination.

    Args:
        combo: Combination dict from generate_combinations().
        config: Base config (used for OSC/MIDI settings).
        num_frames: Frames to measure.
        warmup_frames: Frames to discard before measuring.
        session_dir: Directory to save CSV results.
        preview: Show camera preview before measuring.

    Returns:
        LatencyLogger with results, or None on failure.
    """
    name = combo["name"]

    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK: {name}")
    print(f"  backend={combo['backend']} | {combo['camera_name']} | "
          f"{combo['resolution_name']} | pose={combo['pose_name']} | "
          f"output={combo['output_name']}")
    print(f"{'=' * 60}")

    # --- Build config for this combination ---

    test_config = config.with_overrides(
        camera__device_id=combo["camera_device_id"],
        camera__width=combo["width"],
        camera__height=combo["height"],
        pose__model_complexity=combo["model_complexity"],
        pose__backend=combo["backend"],
        output__mode=combo["output_mode"],
    )
    if combo["send_mode"]:
        test_config.set("osc.send_mode", combo["send_mode"])

    # --- Initialize components ---

    print("  Opening camera...")
    cap = test_config.create_camera()
    if cap is None:
        print(f"  ERROR: Could not open camera (device {combo['camera_device_id']}). "
              f"Skipping.")
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera opened: {actual_w}x{actual_h}")

    pose_estimator = test_config.create_pose_estimator()
    feature_extractor = test_config.create_feature_extractor()
    sender = test_config.create_sender()

    # --- Preview ---

    if preview:
        run_preview(cap, pose_estimator, name)

    # --- Logger ---

    metadata = test_config.to_metadata()
    metadata["benchmark_name"] = name
    metadata["machine"] = get_machine_info()

    logger = LatencyLogger(config=metadata, output_dir=session_dir)

    # --- Warmup ---

    if warmup_frames > 0:
        print(f"  Warming up ({warmup_frames} frames)...")
        prev_landmarks = None
        for _ in range(warmup_frames):
            frame = cap.read()
            if frame is None:
                break
            results = pose_estimator.estimate(frame)
            landmarks = pose_estimator.get_landmarks(results)
            if landmarks:
                feature_extractor.calculate(landmarks, prev_landmarks)
                prev_landmarks = landmarks

    # --- Measure ---

    print(f"  Measuring ({num_frames} frames)...")

    prev_landmarks = None
    progress_interval = max(1, num_frames // 10)

    for i in range(num_frames):
        logger.start_frame()

        # Stage 1: Capture
        frame = cap.read()

        logger.mark("capture")

        if frame is None:
            print("  ERROR: Failed to read frame. Stopping.")
            break

        # Stage 2: Pose estimation
        results = pose_estimator.estimate(frame)
        landmarks = pose_estimator.get_landmarks(results)
        logger.mark("pose")

        # Stage 3: Feature extraction
        if landmarks:
            features = feature_extractor.calculate(landmarks, prev_landmarks)
            prev_landmarks = landmarks
        else:
            features = feature_extractor.calculate(None)
        logger.mark("features")

        # Stage 4: Send
        if sender and features:
            test_config.send_features(sender, features)
        logger.mark("send")

        logger.end_frame(pose_detected=landmarks is not None)

        # Progress
        if (i + 1) % progress_interval == 0:
            pct = (i + 1) / num_frames * 100
            last_ms = logger.get_last_total_ms()
            print(f"  [{pct:5.1f}%] frame {i + 1}/{num_frames} — {last_ms:.1f}ms")

    # --- Cleanup ---

    pose_estimator.release()
    cap.release()
    if sender and combo["output_mode"] == "midi":
        sender.close()

    # --- Results ---

    logger.print_summary()
    logger.save()

    return logger


def get_machine_info() -> str:
    """Gather basic machine info."""
    parts = [
        platform.node(),
        platform.machine(),
        platform.system(),
        platform.processor() or "unknown CPU",
    ]
    return " / ".join(parts)


# =========================================================================
# COMPARISON TABLE
# =========================================================================

def print_comparison(results: list[tuple[str, LatencyLogger]],
                     session_dir: str):
    """Print a final comparison table across all runs."""
    if len(results) < 2:
        return

    print(f"\n\n{'=' * 95}")
    print("  COMPARISON ACROSS ALL CONFIGURATIONS")
    print(f"{'=' * 95}")
    print(f"\n  {'Config':<42} {'Backend':<8} {'Mean':>8} {'P50':>8} "
          f"{'P95':>8} {'FPS':>7} {'<80ms':>7}")
    print("  " + "-" * 90)

    for name, logger in results:
        valid = [f for f in logger.frames if f.get("pose_detected", True)]
        totals = [f["total_ms"] for f in valid]
        fps_vals = [f["fps"] for f in valid if f.get("fps", 0) > 0]

        if not totals:
            continue

        mean = statistics.mean(totals)
        p50 = logger._percentile(totals, 50)
        p95 = logger._percentile(totals, 95)
        fps_mean = statistics.mean(fps_vals) if fps_vals else 0
        under_80 = sum(1 for v in totals if v <= 80) / len(totals) * 100
        backend = logger.config.get("backend", "?")

        print(f"  {name:<42} {backend:<8} {mean:>7.1f}ms {p50:>7.1f}ms "
              f"{p95:>7.1f}ms {fps_mean:>6.1f} {under_80:>6.1f}%")

    print(f"\n  Results saved in: {session_dir}")
    print("=" * 95)


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cuerpo Sonoro — Pipeline Latency Benchmark"
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="Frames per config (overrides config.yaml)"
    )
    parser.add_argument(
        "--warmup", type=int, default=None,
        help="Warmup frames (overrides config.yaml)"
    )
    parser.add_argument(
        "--camera-profile", type=str, default=None,
        help="Only test this camera profile (e.g. 'c922', 'macbook')"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Only test this output mode (e.g. 'osc', 'midi')"
    )
    parser.add_argument(
        "--pose", type=str, default=None,
        help="Only test this pose model (e.g. 'lite', 'full', 'heavy')"
    )
    parser.add_argument(
        "--backend",
        type=str,
        nargs="+",
        default=None,
        choices=["cpu", "metal", "tensorrt"],
        help="Pose estimation backend(s) to benchmark "
             "(e.g. --backend cpu metal). "
             "Default: all backends listed in config.yaml.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all combinations without running"
    )
    parser.add_argument(
        "--config-path", type=str, default=None,
        help="Path to config.yaml (default: project root)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show camera preview with skeleton before each benchmark"
    )
    parser.add_argument(
        "--session-name", type=str, default=None,
        help="Custom session name (e.g. 'pre-defense', 'baseline')"
    )
    args = parser.parse_args()

    # Load config
    config = Config(path=args.config_path)

    num_frames = args.frames or config.benchmark_frames
    warmup = args.warmup or config.benchmark_warmup

    # Generate combinations
    combinations = generate_combinations(
        config,
        camera_filter=args.camera_profile,
        output_filter=args.output,
        pose_filter=args.pose,
        backend_filter=args.backend,
    )

    if not combinations:
        print("No combinations match your filters.")
        sys.exit(1)

    # Build session directory
    session_name = build_session_name(
        camera_filter=args.camera_profile,
        output_filter=args.output,
        pose_filter=args.pose,
        backend_filter=args.backend,
        custom_name=args.session_name,
    )
    session_dir = ensure_unique_session(os.path.join(RESULTS_DIR, session_name))

    # Header
    print("=" * 60)
    print("  CUERPO SONORO — LATENCY BENCHMARK")
    print("=" * 60)
    print(f"\n  Machine:     {get_machine_info()}")
    print(f"  Configs:     {len(combinations)}")
    print(f"  Frames:      {num_frames} per config (+{warmup} warmup)")
    print(f"  Session:     {session_name}")
    print(f"  Output dir:  {session_dir}")

    print_combination_table(combinations)

    if args.list:
        print(f"\n  Total: {len(combinations)} combinations")
        sys.exit(0)

    # Create session directory
    os.makedirs(session_dir, exist_ok=True)

    # Confirm
    total_time_est = len(combinations) * (num_frames + warmup) / 30  # ~30fps
    print(f"\n  Estimated time: ~{total_time_est:.0f}s "
          f"({total_time_est / 60:.1f} min)")

    if not args.preview:
        input("\n  Press ENTER to start (stand in front of camera)...")
    else:
        print(f"\n  Preview mode: a camera window will open before each run.")

    # Run all combinations
    results = []
    for i, combo in enumerate(combinations, 1):
        print(f"\n  >>> Running {i}/{len(combinations)}: {combo['name']}")

        logger = run_single_benchmark(
            combo=combo,
            config=config,
            num_frames=num_frames,
            warmup_frames=warmup,
            session_dir=session_dir,
            preview=args.preview,
        )

        if logger and logger.frame_count > 0:
            results.append((combo["name"], logger))

        # Pause between camera switches to let USB stabilize
        if (i < len(combinations) and
                combo["camera_device_id"] !=
                combinations[i]["camera_device_id"]):
            print("\n  Camera switch detected. "
                  "Pausing 3s for USB to stabilize...")
            import time
            time.sleep(3)

    # Final comparison
    print_comparison(results, session_dir)


if __name__ == "__main__":
    main()
