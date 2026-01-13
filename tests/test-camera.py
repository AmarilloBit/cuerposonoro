#!/usr/bin/env python3
"""
Test script to verify camera capture with Logitech C922.

Usage:
    python tests/test_camera.py

Controls:
    q - Quit
    s - Save screenshot to logs/
"""

import cv2
import time
import sys
from pathlib import Path


def test_camera(device_id: int = 0, width: int = 1280, height: int = 720):
    """Test camera capture and display FPS."""

    print(f"Opening camera {device_id}...")
    cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {device_id}")
        print("Tips:")
        print("  - Check if camera is connected")
        print("  - Try different device_id (0, 1, 2...)")
        print("  - On macOS, grant camera permissions to Terminal")
        sys.exit(1)

    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Verify actual settings
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened successfully!")
    print(f"Resolution: {actual_w}x{actual_h}")
    print(f"Target FPS: {actual_fps}")
    print(f"\nPress 'q' to quit, 's' to save screenshot\n")

    # FPS calculation variables
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to read frame")
            break

        # Calculate FPS
        fps_counter += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()

        # Draw FPS on frame
        cv2.putText(
            frame,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # Draw resolution info
        cv2.putText(
            frame,
            f"{actual_w}x{actual_h}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Display frame
        cv2.imshow("Cuerpo Sonoro - Camera Test", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nExiting...")
            break
        elif key == ord('s'):
            filename = f"logs/screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final FPS: {current_fps:.1f}")


if __name__ == "__main__":
    test_camera()