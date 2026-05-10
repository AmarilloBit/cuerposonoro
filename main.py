"""
Cuerpo Sonoro — Main entry point.

Runs the full pipeline: camera → pose estimation → feature extraction → audio output.

Usage:
    # Live performance (webcam, output mode from config.yaml)
    python main.py

    # Feature validation with a pre-recorded video (loops automatically)
    python main.py --source tests/videos/my_test.mp4

    # Debug overlay showing live feature values
    python main.py --debug
    python main.py --source tests/videos/my_test.mp4 --debug

    # Override output mode at runtime
    python main.py --mode midi
    python main.py --mode osc
"""

import argparse
import sys
import time
import cv2

from vision_processor.config import Config


# =============================================================================
# Debug overlay
# =============================================================================

def _draw_debug_overlay(frame, features: dict, midi_state: dict | None = None):
    """
    Draw feature values and active MIDI state on the frame.

    Args:
        frame:      BGR frame to draw on (modified in place).
        features:   Feature dict from FeatureExtractor.
        midi_state: Optional dict with active MIDI notes/channels from MidiSender.
    """
    # Semi-transparent dark background for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (420, 310), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    def put(text, row, color=(0, 255, 120)):
        cv2.putText(
            frame, text,
            (10, 24 + row * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, color, 1, cv2.LINE_AA,
        )

    # --- Basic features ---
    energy    = features.get("energy", 0)
    symmetry  = features.get("symmetry", 0)
    smoothness = features.get("smoothness", 0)
    arm_angle = features.get("armAngle", 0)
    v_ext     = features.get("verticalExtension", 0)

    put(f"energy:     {energy:.2f}  smoothness: {smoothness:.2f}", 0)
    put(f"symmetry:   {symmetry:+.2f}  armAngle:   {arm_angle:.2f}", 1)
    put(f"vertExt:    {v_ext:.2f}", 2)

    # --- Chord features ---
    feet_x    = features.get("feetCenterX", 0.5)
    hip_tilt  = features.get("hipTilt", 0)
    knee_ang  = features.get("kneeAngle", 0)
    head_tilt = features.get("headTilt", 0)

    # Map feetCenterX to chord zone label
    if feet_x < 0.25:
        zone = "I"
    elif feet_x < 0.50:
        zone = "IV"
    elif feet_x < 0.75:
        zone = "V"
    else:
        zone = "VI"

    put(f"feetX:      {feet_x:.2f}  → chord {zone}", 3, color=(100, 220, 255))
    put(f"hipTilt:    {hip_tilt:+.2f}  kneeAngle:  {knee_ang:.2f}", 4)
    put(f"headTilt:   {head_tilt:+.2f}", 5)

    # --- Melody / jerk triggers ---
    r_jerk  = features.get("rightHandJerk", 0)
    l_jerk  = features.get("leftHandJerk", 0)
    r_y     = features.get("rightHandY", 0)
    l_y     = features.get("leftHandY", 0)
    r_vel   = features.get("rightArmVelocity", 0)
    l_vel   = features.get("leftArmVelocity", 0)

    # Trigger indicator adapts to the active MIDI mode
    top_vel = max(r_vel, l_vel)
    bp_triggered = top_vel > 0.08
    jerk_triggered_r = r_jerk > 0.4
    jerk_triggered_l = l_jerk > 0.4

    r_color = (0, 80, 255) if (bp_triggered or jerk_triggered_r) else (0, 255, 120)
    l_color = (0, 80, 255) if (bp_triggered or jerk_triggered_l) else (0, 255, 120)

    r_label = f"R-hand Y:{r_y:.2f} jerk:{r_jerk:.2f} vel:{r_vel:.2f}"
    l_label = f"L-hand Y:{l_y:.2f} jerk:{l_jerk:.2f} vel:{l_vel:.2f}"
    if bp_triggered:
        r_label += "  *** MELODY ***"

    put(r_label, 7, color=r_color)
    put(l_label, 8, color=l_color)

    # --- Active MIDI notes (if midi_state provided) ---
    if midi_state:
        chord_note  = midi_state.get("chord", "—")
        r_note      = midi_state.get("melody_right", "—")
        l_note      = midi_state.get("melody_left", "—")
        put(f"MIDI chord:{chord_note}  R-note:{r_note}  L-note:{l_note}", 10,
            color=(255, 200, 80))


# =============================================================================
# Argument parsing
# =============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Cuerpo Sonoro — body movement to music pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a video file for testing (e.g. tests/videos/test.mp4). "
             "If omitted, uses the live webcam from config.yaml.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show feature values overlay on the video window.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["osc", "midi"],
        default=None,
        help="Override output.mode from config.yaml.",
    )
    parser.add_argument(
        "--midi-mode",
        type=str,
        choices=["classic", "musical"],
        default=None,
        dest="midi_mode",
        help="Override output.midi_mode from config.yaml (only used when --mode midi).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cpu", "metal", "tensorrt"],
        default=None,
        help="Force a specific pose estimation backend (default: auto-detect).",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        dest="no_loop",
        help="When using --source, exit at end of file instead of looping. "
             "Useful for one-shot recordings.",
    )
    return parser.parse_args()


# =============================================================================
# Main loop
# =============================================================================

def main():
    args = _parse_args()

    # --- Config ---
    overrides = {}
    if args.mode:
        overrides["output.mode"] = args.mode
    if args.midi_mode:
        overrides["output.midi_mode"] = args.midi_mode
    if args.backend:
        overrides["pose.backend"] = args.backend

    config = Config(overrides=overrides if overrides else None)

    print(f"[main] Config: {config.describe()}")
    if args.source:
        loop_label = "single pass" if args.no_loop else "looping"
        print(f"[main] Source: video file → {args.source}  ({loop_label})")
    else:
        print(f"[main] Source: webcam device {config.camera_device_id}")
    if args.debug:
        print("[main] Debug overlay: ON")

    # --- Pipeline components ---
    try:
        camera = config.create_camera(source=args.source, loop=not args.no_loop)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    pose_estimator     = config.create_pose_estimator()
    feature_extractor  = config.create_feature_extractor()

    try:
        sender = config.create_sender()
    except Exception as e:
        print(f"[ERROR] Could not create audio sender: {e}")
        camera.release()
        sys.exit(1)

    # --- State ---
    prev_landmarks = None
    frame_count    = 0
    fps            = 0.0
    prev_time      = time.time()

    print("\nRunning. Press 'q' in the video window to quit.\n")

    # --- Loop ---
    try:
        while camera.is_open():
            frame = camera.read()
            if frame is None:
                # VideoFileCamera with loop=False would end here;
                # with loop=True this should never happen.
                break

            frame_count += 1

            # FPS calculation
            now      = time.time()
            fps      = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Pose estimation
            results   = pose_estimator.estimate(frame)
            landmarks = pose_estimator.get_landmarks(results)

            midi_state = None

            if landmarks:
                # Draw skeleton
                frame = pose_estimator.draw_skeleton(frame, results)

                # Feature extraction
                features = feature_extractor.calculate(landmarks, prev_landmarks)

                # Audio output
                config.send_features(sender, features)

                # Collect MIDI state for debug overlay
                if args.debug and config.output_mode == "midi":
                    if hasattr(sender, "current_chord"):
                        midi_state = {
                            "chord": sender.current_chord or "---",
                            "melody_right": getattr(sender, "melody_right_note", "---") or "---",
                            "melody_left":  getattr(sender, "melody_left_note", "---") or "---",
                        }

                if args.debug:
                    _draw_debug_overlay(frame, features, midi_state)

                prev_landmarks = landmarks

            else:
                prev_landmarks = None
                cv2.putText(
                    frame, "NO POSE DETECTED",
                    (50, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 2, cv2.LINE_AA,
                )

            # HUD: FPS always visible
            cv2.putText(
                frame, f"FPS: {fps:.0f}",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1, cv2.LINE_AA,
            )

            cv2.imshow("Cuerpo Sonoro", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[main] Quit requested.")
                break

    except KeyboardInterrupt:
        print("\n[main] Interrupted.")

    finally:
        print("[main] Cleaning up...")
        camera.release()
        pose_estimator.release()
        cv2.destroyAllWindows()

        # MidiSender needs explicit close to send All Notes Off
        if hasattr(sender, "close"):
            sender.close()

        print(f"[main] Done. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
