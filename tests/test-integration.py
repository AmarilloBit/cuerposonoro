"""
Integration test: Camera → Pose → Features → OSC → SuperCollider

Before running:
1. Open SuperCollider
2. Run: s.options.numInputBusChannels = 0; s.boot;
3. Load the OSCdef listeners (see below)

Usage:
    python tests/test_integration.py
"""

import cv2
import time
import sys

sys.path.insert(0, '.')

from vision_processor.pose import PoseEstimator
from vision_processor.features import FeatureExtractor
from vision_processor.osc_sender import OSCSender


def main():
    print("=" * 50)
    print("CUERPO SONORO - Integration Test")
    print("=" * 50)

    # Initialize components
    print("\n[1/4] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[2/4] Initializing pose estimator...")
    pose_estimator = PoseEstimator(model_complexity=0)

    print("[3/4] Initializing feature extractor...")
    feature_extractor = FeatureExtractor()

    print("[4/4] Initializing OSC sender...")
    osc_sender = OSCSender(host="127.0.0.1", port=57120)

    print("\n" + "=" * 50)
    print("Ready! Move in front of the camera.")
    print("Press 'q' to quit (click window first)")
    print("=" * 50 + "\n")

    # Tracking
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0
    prev_landmarks = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pose estimation
        results = pose_estimator.estimate(frame)
        landmarks = pose_estimator.get_landmarks(results)

        # Feature extraction & OSC
        if landmarks:
            features = feature_extractor.calculate(landmarks, prev_landmarks)
            osc_sender.send_features(features)
            prev_landmarks = landmarks

            # Draw features on screen
            y_offset = 130
            for name, value in features.items():
                # Draw bar
                bar_width = int(value * 200) if value >= 0 else int(abs(value) * 200)
                bar_x = 220 if value >= 0 else 220 - bar_width
                color = (0, 255, 0) if value >= 0 else (0, 0, 255)

                cv2.putText(frame, f"{name}:", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, (bar_x, y_offset - 12),
                              (bar_x + bar_width, y_offset), color, -1)
                cv2.putText(frame, f"{value:.2f}", (430, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25

        # Draw skeleton
        frame = pose_estimator.draw_skeleton(frame, results)

        # FPS calculation
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        # Draw status
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        status = "SENDING OSC" if landmarks else "NO POSE"
        color = (0, 255, 0) if landmarks else (0, 0, 255)
        cv2.putText(frame, status, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "127.0.0.1:57120", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Cuerpo Sonoro - Integration Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    pose_estimator.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
