"""
End-to-end test with CSV logging.

Logs all feature values to a CSV file for later analysis.
Log file: logs/midi_e2e_debug_TIMESTAMP.csv
"""

import sys

sys.path.insert(0, '.')

import cv2
import time
import csv
import os
from datetime import datetime
from vision_processor.pose import PoseEstimator
from vision_processor.features import FeatureExtractor
from vision_processor.midi_sender import MidiSender

# Override the jerk threshold for testing
MidiSender.JERK_THRESHOLD = 0.2


def main():
    print("=" * 60)
    print("   CUERPO SONORO - E2E TEST WITH LOGGING")
    print("=" * 60)

    # Create logs directory if needed
    os.makedirs("logs", exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/midi_e2e_debug_{timestamp}.csv"

    print(f"\nLog file: {log_filename}")

    # Initialize components
    print("\n[1/4] Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[2/4] Initializing pose estimator...")
    pose_estimator = PoseEstimator(model_complexity=1)

    print("[3/4] Initializing feature extractor...")
    feature_extractor = FeatureExtractor()

    print("[4/4] Initializing MIDI sender...")
    midi_sender = MidiSender()

    # CSV columns
    csv_columns = [
        "timestamp",
        "frame",
        "fps",
        "pose_detected",
        # Chord features
        "feetCenterX",
        "chord_zone",
        "hipTilt",
        "kneeAngle",
        # Melody features
        "rightHandY",
        "leftHandY",
        "rightHandJerk",
        "leftHandJerk",
        "rightHandJerk_triggered",
        "leftHandJerk_triggered",
        "rightArmVelocity",
        "leftArmVelocity",
        # Global
        "headTilt",
        "energy",
        "smoothness",
    ]

    print("\n" + "=" * 60)
    print("   READY! Connect Surge XT to 'Cuerpo Sonoro'")
    print("=" * 60)
    print("""
HOW TO CONTROL:

  CHORDS: Move LEFT/RIGHT (feet position = chord)
  MELODY: Move hands UP/DOWN, move FAST to trigger note

  Press 'q' to quit and save log
    """)

    prev_time = time.time()
    fps = 0
    prev_landmarks = None
    frame_count = 0
    jerk_threshold = 0.2

    # Open CSV file
    with open(log_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                results = pose_estimator.estimate(frame)
                landmarks = pose_estimator.get_landmarks(results)

                frame_count += 1
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                # Prepare log row
                log_row = {
                    "timestamp": datetime.now().isoformat(),
                    "frame": frame_count,
                    "fps": round(fps, 1),
                    "pose_detected": landmarks is not None,
                }

                if landmarks:
                    frame = pose_estimator.draw_skeleton(frame, results)
                    features = feature_extractor.calculate(landmarks, prev_landmarks)
                    prev_landmarks = landmarks

                    # Send to Surge XT
                    midi_sender.update(features)

                    # Determine chord zone
                    feet_x = features.get("feetCenterX", 0.5)
                    if feet_x < 0.25:
                        chord_zone = "I"
                    elif feet_x < 0.5:
                        chord_zone = "IV"
                    elif feet_x < 0.75:
                        chord_zone = "V"
                    else:
                        chord_zone = "VI"

                    # Check triggers
                    r_triggered = features.get("rightHandJerk", 0) > jerk_threshold
                    l_triggered = features.get("leftHandJerk", 0) > jerk_threshold

                    # Fill log row
                    log_row.update({
                        "feetCenterX": round(features.get("feetCenterX", 0), 4),
                        "chord_zone": chord_zone,
                        "hipTilt": round(features.get("hipTilt", 0), 4),
                        "kneeAngle": round(features.get("kneeAngle", 0), 4),
                        "rightHandY": round(features.get("rightHandY", 0), 4),
                        "leftHandY": round(features.get("leftHandY", 0), 4),
                        "rightHandJerk": round(features.get("rightHandJerk", 0), 4),
                        "leftHandJerk": round(features.get("leftHandJerk", 0), 4),
                        "rightHandJerk_triggered": r_triggered,
                        "leftHandJerk_triggered": l_triggered,
                        "rightArmVelocity": round(features.get("rightArmVelocity", 0), 4),
                        "leftArmVelocity": round(features.get("leftArmVelocity", 0), 4),
                        "headTilt": round(features.get("headTilt", 0), 4),
                        "energy": round(features.get("energy", 0), 4),
                        "smoothness": round(features.get("smoothness", 0), 4),
                    })

                    # Console output every 30 frames
                    if frame_count % 30 == 0:
                        print(f"[{frame_count:5d}] chord={chord_zone} | "
                              f"feetX={feet_x:.2f} | "
                              f"R-jerk={features.get('rightHandJerk', 0):.3f} | "
                              f"L-jerk={features.get('leftHandJerk', 0):.3f}")

                    # Show trigger on console
                    if r_triggered:
                        print(f"  >>> RIGHT HAND TRIGGER! note at Y={features.get('rightHandY', 0):.2f}")
                    if l_triggered:
                        print(f"  >>> LEFT HAND TRIGGER! note at Y={features.get('leftHandY', 0):.2f}")

                else:
                    # No pose
                    cv2.putText(frame, "NO POSE DETECTED", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Write to CSV
                writer.writerow(log_row)

                # Simple on-screen display (visible from far)
                cv2.putText(frame, f"Frame: {frame_count} | FPS: {fps:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit and save log", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                cv2.imshow("Cuerpo Sonoro", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted")

    print(f"\n{'=' * 60}")
    print(f"   LOG SAVED: {log_filename}")
    print(f"   Total frames: {frame_count}")
    print(f"{'=' * 60}")

    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    pose_estimator.release()
    midi_sender.close()
    print("Done!")


if __name__ == "__main__":
    main()
