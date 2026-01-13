"""
Pose estimation module using MediaPipe.
"""

import mediapipe as mp
import cv2
import numpy as np


class PoseEstimator:
    """Wrapper for MediaPipe Pose estimation."""

    def __init__(
            self,
            model_complexity: int = 1,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5
    ):
        """
        Initialize pose estimator.

        Args:
            model_complexity: 0=lite, 1=full, 2=heavy
            min_detection_confidence: Minimum detection confidence [0.0-1.0]
            min_tracking_confidence: Minimum tracking confidence [0.0-1.0]
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def estimate(self, frame: np.ndarray):
        """
        Estimate pose from BGR frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            MediaPipe pose results (None if no pose detected)
        """
        # MediaPipe needs RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results

    def draw_skeleton(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw pose skeleton on frame.

        Args:
            frame: BGR image to draw on
            results: MediaPipe pose results

        Returns:
            Frame with skeleton drawn
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame

    def get_landmarks(self, results) -> list | None:
        """
        Extract landmarks as list of (x, y, z, visibility).

        Args:
            results: MediaPipe pose results

        Returns:
            List of 33 landmarks or None if no pose detected
        """
        if not results.pose_landmarks:
            return None

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            })
        return landmarks

    def release(self):
        """Release resources."""
        self.pose.close()