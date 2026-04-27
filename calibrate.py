"""
Cuerpo Sonoro — Automatic Calibration

Runs test videos through the pose + feature pipeline (headless, no MIDI),
records per-frame feature values to CSV, then analyzes distributions to
recommend optimal blueprint thresholds.

Usage:
    # Analyse videos and print recommendations:
    python calibrate.py videos/slow_arms.mp4 videos/fast_arms.mp4 videos/still.mp4

    # Also write updated config.yaml automatically:
    python calibrate.py --apply videos/*.mp4

    # Label a video with the movement type it contains:
    python calibrate.py --label slow_arms videos/slow.mp4 --label fast_arms videos/fast.mp4

Expected labels (optional, but improve recommendations):
    still          - standing still, no intentional movement
    slow_arms      - gentle / slow arm movement
    fast_arms      - vigorous / fast arm movement
    pelvis         - pelvis thrust forward/backward
    lean           - spine lean forward/backward
    full_body      - mixed / free movement
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from vision_processor.capture import VideoFileCamera
from vision_processor.config import Config
from vision_processor.features import FeatureExtractor


# ── Feature keys we care about for calibration ──────────────────────────

FEATURE_KEYS = [
    "energy", "symmetry", "smoothness", "armAngle", "verticalExtension",
    "feetCenterX", "hipTilt", "kneeAngle",
    "rightHandY", "leftHandY",
    "rightHandJerk", "leftHandJerk",
    "rightArmVelocity", "leftArmVelocity",
    "rightElbowHipAngle", "leftElbowHipAngle",
    "headTilt",
    "pelvis_thrust", "spine_lean",
]


# ── Video processing ────────────────────────────────────────────────────

def process_video(video_path: str, config: Config) -> list[dict]:
    """Run a video through pose + features, return list of feature dicts."""
    camera = VideoFileCamera(path=video_path, loop=False)
    pose = config.create_pose_estimator()
    extractor = config.create_feature_extractor()

    frames = []
    prev_landmarks = None
    frame_num = 0

    while camera.is_open():
        frame = camera.read()
        if frame is None:
            break

        frame_num += 1
        results = pose.estimate(frame)
        landmarks = pose.get_landmarks(results)

        if landmarks:
            features = extractor.calculate(landmarks, prev_landmarks)
            features["_frame"] = frame_num
            features["_has_pose"] = True
            frames.append(features)
            prev_landmarks = landmarks
        else:
            frames.append({"_frame": frame_num, "_has_pose": False})
            prev_landmarks = None

    camera.release()
    pose.release()

    detected = sum(1 for f in frames if f.get("_has_pose"))
    print(f"  → {frame_num} frames, {detected} with pose ({100*detected/max(1,frame_num):.0f}%)")
    return frames


def save_csv(frames: list[dict], output_path: str):
    """Write feature data to CSV."""
    header = ["frame", "has_pose"] + FEATURE_KEYS
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in frames:
            if not row.get("_has_pose"):
                writer.writerow([row["_frame"], False] + [""] * len(FEATURE_KEYS))
            else:
                writer.writerow(
                    [row["_frame"], True] +
                    [f"{row.get(k, 0.0):.6f}" for k in FEATURE_KEYS]
                )


# ── Analysis ────────────────────────────────────────────────────────────

def analyse_features(all_data: dict[str, list[dict]]) -> dict:
    """
    Analyse feature distributions across labelled videos.

    Returns a dict of recommended config values with reasoning.
    """
    recommendations = {}

    # Collect arrays per feature across all videos
    per_video = {}
    for label, frames in all_data.items():
        arrays = {}
        pose_frames = [f for f in frames if f.get("_has_pose")]
        for key in FEATURE_KEYS:
            arrays[key] = np.array([f.get(key, 0.0) for f in pose_frames])
        per_video[label] = arrays

    # Merge all frames for global stats
    all_frames = []
    for frames in all_data.values():
        all_frames.extend(f for f in frames if f.get("_has_pose"))

    global_arrays = {}
    for key in FEATURE_KEYS:
        global_arrays[key] = np.array([f.get(key, 0.0) for f in all_frames])

    # ── Arm velocity threshold ──────────────────────────────────────
    r_vel = global_arrays["rightArmVelocity"]
    l_vel = global_arrays["leftArmVelocity"]
    max_vel = np.maximum(r_vel, l_vel)

    # Look for labelled still video
    still_labels = [l for l in all_data if "still" in l.lower()]
    if still_labels:
        still_arr = per_video[still_labels[0]]
        still_max_vel = np.maximum(
            still_arr["rightArmVelocity"],
            still_arr["leftArmVelocity"],
        )
        noise_floor = np.percentile(still_max_vel, 95)
    else:
        # Use the 10th percentile of all movement as noise proxy
        noise_floor = np.percentile(max_vel[max_vel > 0], 10) if np.any(max_vel > 0) else 0.02

    # Threshold = midpoint between noise ceiling and movement floor
    movement_labels = [l for l in all_data if "arm" in l.lower() or "full" in l.lower()]
    if movement_labels:
        move_vels = []
        for ml in movement_labels:
            arr = per_video[ml]
            move_vels.append(np.maximum(arr["rightArmVelocity"], arr["leftArmVelocity"]))
        move_vel = np.concatenate(move_vels)
        movement_floor = np.percentile(move_vel[move_vel > 0], 15) if np.any(move_vel > 0) else 0.05
    else:
        movement_floor = np.percentile(max_vel[max_vel > 0], 25) if np.any(max_vel > 0) else 0.05

    arm_thresh = round((noise_floor + movement_floor) / 2, 3)
    arm_thresh = max(0.03, min(0.20, arm_thresh))  # sanity clamp

    recommendations["arm_velocity_threshold"] = {
        "value": arm_thresh,
        "noise_p95": round(float(noise_floor), 4),
        "movement_p15": round(float(movement_floor), 4),
        "reason": f"Midpoint between noise ceiling ({noise_floor:.3f}) and movement floor ({movement_floor:.3f})",
    }

    # ── Melody cooldown ─────────────────────────────────────────────
    # Based on how often velocity crosses threshold
    above = max_vel > arm_thresh
    crossings = np.diff(above.astype(int))
    onsets = np.where(crossings == 1)[0]
    if len(onsets) > 1:
        intervals = np.diff(onsets) / 30.0  # assume ~30fps
        median_interval = float(np.median(intervals))
        cooldown = round(max(0.08, min(0.4, median_interval * 0.6)), 3)
    else:
        cooldown = 0.15

    recommendations["melody_cooldown"] = {
        "value": cooldown,
        "reason": f"60% of median onset interval ({median_interval:.3f}s)" if len(onsets) > 1 else "Default (not enough data)",
    }

    # ── Pelvis threshold ────────────────────────────────────────────
    pelvis = np.abs(global_arrays["pelvis_thrust"])

    pelvis_labels = [l for l in all_data if "pelvis" in l.lower()]
    if pelvis_labels:
        pelvis_arr = np.abs(per_video[pelvis_labels[0]]["pelvis_thrust"])
        pelvis_active = np.percentile(pelvis_arr[pelvis_arr > 0], 25) if np.any(pelvis_arr > 0) else 0.10
    else:
        pelvis_active = np.percentile(pelvis[pelvis > 0], 30) if np.any(pelvis > 0) else 0.10

    # Noise floor from still or global low percentile
    if still_labels:
        pelvis_noise = np.percentile(np.abs(per_video[still_labels[0]]["pelvis_thrust"]), 95)
    else:
        pelvis_noise = np.percentile(pelvis, 50)

    pelvis_thresh = round((pelvis_noise + pelvis_active) / 2, 3)
    pelvis_thresh = max(0.05, min(0.25, pelvis_thresh))

    recommendations["pelvis_threshold"] = {
        "value": pelvis_thresh,
        "reason": f"Midpoint between noise ({pelvis_noise:.3f}) and active movement ({pelvis_active:.3f})",
    }

    # ── Silence threshold ───────────────────────────────────────────
    velocity_keys = ["rightArmVelocity", "leftArmVelocity", "rightHandJerk", "leftHandJerk", "energy"]
    mean_vel = np.mean([global_arrays[k] for k in velocity_keys], axis=0)

    if still_labels:
        still_mean_vel = np.mean(
            [per_video[still_labels[0]][k] for k in velocity_keys], axis=0
        )
        silence_thresh = round(float(np.percentile(still_mean_vel, 90)), 4)
    else:
        silence_thresh = round(float(np.percentile(mean_vel, 10)), 4)

    silence_thresh = max(0.005, min(0.05, silence_thresh))

    recommendations["silence_threshold"] = {
        "value": silence_thresh,
        "reason": "90th percentile of still video mean velocity" if still_labels else "10th percentile of all mean velocity",
    }

    # ── Global stats summary ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FEATURE STATISTICS")
    print("=" * 70)

    for label, arrays in per_video.items():
        print(f"\n── {label} ({len(all_data[label])} frames) ──")
        for key in ["rightArmVelocity", "leftArmVelocity", "rightHandJerk",
                     "leftHandJerk", "pelvis_thrust", "spine_lean", "energy"]:
            arr = arrays[key]
            if len(arr) == 0:
                continue
            print(f"  {key:>22s}  "
                  f"min={np.min(arr):.4f}  "
                  f"median={np.median(arr):.4f}  "
                  f"p75={np.percentile(arr, 75):.4f}  "
                  f"p95={np.percentile(arr, 95):.4f}  "
                  f"max={np.max(arr):.4f}")

    return recommendations


def print_recommendations(recs: dict):
    """Print calibration recommendations."""
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIG (output.blueprint)")
    print("=" * 70)

    for key, info in recs.items():
        print(f"\n  {key}: {info['value']}")
        print(f"    Reason: {info['reason']}")


def apply_config(recs: dict, config_path: str):
    """Update config.yaml with recommended values."""
    with open(config_path, "r") as f:
        lines = f.readlines()

    updated = []
    for line in lines:
        replaced = False
        for key, info in recs.items():
            # Match lines like "    arm_velocity_threshold: 0.08 # comment"
            stripped = line.lstrip()
            if stripped.startswith(f"{key}:"):
                indent = line[:len(line) - len(stripped)]
                comment_idx = stripped.find("#")
                if comment_idx >= 0:
                    comment = stripped[comment_idx:]
                else:
                    comment = f"# calibrated"
                updated.append(f"{indent}{key}: {info['value']}  {comment}\n")
                replaced = True
                break
        if not replaced:
            updated.append(line)

    with open(config_path, "w") as f:
        f.writelines(updated)

    print(f"\n  ✓ Updated {config_path}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cuerpo Sonoro — Automatic Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "videos", nargs="*",
        help="Video files to process. Use --label to tag them.",
    )
    parser.add_argument(
        "--label", nargs=2, action="append", metavar=("LABEL", "VIDEO"),
        default=[],
        help="Label a video: --label still stand.mp4 --label fast_arms wave.mp4. "
             "Labels: still, slow_arms, fast_arms, pelvis, lean, full_body",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write recommended values to config.yaml.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="calibration",
        help="Directory to write CSV files (default: calibration/).",
    )
    args = parser.parse_args()

    # Build video list: labelled + unlabelled
    # Use path -> label so each file appears once (label can repeat)
    video_map: dict[str, str] = {}  # path -> label
    for label, path in args.label:
        video_map[path] = label

    # Unlabelled videos get their filename stem as label
    for path in (args.videos or []):
        if path not in video_map:
            video_map[path] = Path(path).stem

    if not video_map:
        parser.print_help()
        print("\nError: No videos provided.")
        sys.exit(1)

    # Ensure output dir exists
    out_dir = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Process each video
    config = Config()
    # Group frames by label (multiple videos can share a label)
    all_data: dict[str, list[dict]] = {}

    for path, label in video_map.items():
        print(f"\nProcessing: {path}  (label: {label})")
        if not os.path.isfile(path):
            print(f"  ✗ File not found: {path}")
            continue

        frames = process_video(path, config)

        if label in all_data:
            all_data[label].extend(frames)
        else:
            all_data[label] = frames

        stem = Path(path).stem
        csv_path = os.path.join(out_dir, f"{stem}.csv")
        save_csv(frames, csv_path)
        print(f"  → Saved {csv_path}")

    if not all_data:
        print("\nNo videos processed.")
        sys.exit(1)

    # Analyse and recommend
    recs = analyse_features(all_data)
    print_recommendations(recs)

    # Apply if requested
    if args.apply:
        config_path = os.path.join(ROOT_DIR, "config.yaml")
        apply_config(recs, config_path)
    else:
        print("\n  (Run with --apply to write these values to config.yaml)")


if __name__ == "__main__":
    main()
