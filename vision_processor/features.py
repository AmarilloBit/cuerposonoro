"""
Feature extraction from pose landmarks.
This is the simplified version for the web demo.
This file has been copy-pasted from the web demo version.
"""

import math
from typing import Optional


class FeatureExtractor:
    """Extracts musical features from MediaPipe pose landmarks."""

    def __init__(self):
        self.prev_landmarks = None
        self.smoothing_factor = 0.3  # For temporal smoothing
        self.prev_features = None

    def calculate(self, landmarks: list, prev_landmarks: Optional[list] = None) -> dict:
        """
        Calculate all features from landmarks.

        Args:
            landmarks: List of 33 landmarks, each with {x, y, z, visibility}
            prev_landmarks: Previous frame landmarks for velocity calculation

        Returns:
            dict with normalized features (0.0 - 1.0)
        """
        if not landmarks or len(landmarks) < 33:
            return self._empty_features()

        features = {
            "energy": self._calculate_energy(landmarks, prev_landmarks),
            "symmetry": self._calculate_symmetry(landmarks),
            "smoothness": self._calculate_smoothness(landmarks, prev_landmarks),
            "armAngle": self._calculate_arm_angle(landmarks),
            "verticalExtension": self._calculate_vertical_extension(landmarks),
        }

        # Apply temporal smoothing
        features = self._smooth_features(features)

        return features

    def _calculate_energy(self, landmarks: list, prev_landmarks: Optional[list]) -> float:
        """
        Overall motion energy based on velocity of key points: wrists, ankles, nose...
        More movent = more energy
        """
        if prev_landmarks is None:
            return 0.0

        # Key points: wrists, ankles, nose
        key_indices = [0, 15, 16, 27, 28]  # nose, wrists, ankles

        total_velocity = 0.0
        for idx in key_indices:
            if idx < len(landmarks) and idx < len(prev_landmarks):
                dx = landmarks[idx]["x"] - prev_landmarks[idx]["x"]
                dy = landmarks[idx]["y"] - prev_landmarks[idx]["y"]
                velocity = math.sqrt(dx ** 2 + dy ** 2)
                total_velocity += velocity

        # Normalize (empirical values, adjust as needed)
        energy = min(total_velocity * 10, 1.0)
        return energy

    def _calculate_symmetry(self, landmarks: list) -> float:
        """
        Left-right symmetry index.
        Returns: -1.0 (left heavy) to 1.0 (right heavy), 0.0 = balanced
        """
        # Compare left vs right wrist positions
        left_wrist = landmarks[15]  # left wrist
        right_wrist = landmarks[16]  # right wrist

        # Calculate horizontal center of mass for arms
        left_x = left_wrist["x"]
        right_x = right_wrist["x"]

        # Center is at 0.5, calculate deviation
        center = 0.5
        left_dev = center - left_x
        right_dev = right_x - center

        # Positive = right side more extended, negative = left side
        symmetry = right_dev - left_dev

        # Clamp to -1, 1
        return max(-1.0, min(1.0, symmetry * 2))

    def _calculate_smoothness(self, landmarks: list, prev_landmarks: Optional[list]) -> float:
        """
        Movement smoothness (inverse of jerk).
        High value = smooth, flowing movement
        Low value = abrupt, jerky movement
        """
        if prev_landmarks is None:
            return 0.5

        # Calculate acceleration changes (simplified jerk estimation)
        # Using wrists as primary indicators
        wrist_indices = [15, 16]

        total_jerk = 0.0
        for idx in wrist_indices:
            dx = landmarks[idx]["x"] - prev_landmarks[idx]["x"]
            dy = landmarks[idx]["y"] - prev_landmarks[idx]["y"]
            movement = math.sqrt(dx ** 2 + dy ** 2)
            total_jerk += movement

        # Invert and normalize (high jerk = low smoothness)
        smoothness = 1.0 - min(total_jerk * 5, 1.0)
        return max(0.0, smoothness)

    def _calculate_arm_angle(self, landmarks: list) -> float:
        """
        Average arm elevation angle (0 = down, 1 = horizontal or above).
        """
        # Shoulders and wrists
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        def arm_elevation(shoulder, wrist):
            # Vertical difference (negative y = higher in screen coords)
            dy = shoulder["y"] - wrist["y"]  # positive = arm raised
            return max(0.0, min(1.0, dy + 0.5))  # normalize

        left_angle = arm_elevation(left_shoulder, left_wrist)
        right_angle = arm_elevation(right_shoulder, right_wrist)

        return (left_angle + right_angle) / 2

    def _calculate_vertical_extension(self, landmarks: list) -> float:
        """
        How vertically extended the body is (crouching vs stretching).
        """
        nose = landmarks[0]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        ankle_y = (left_ankle["y"] + right_ankle["y"]) / 2

        # Height from ankles to nose
        height = ankle_y - nose["y"]  # positive = standing tall

        # Normalize (empirical, adjust based on testing)
        extension = max(0.0, min(1.0, height * 1.5))
        return extension

    def _smooth_features(self, features: dict) -> dict:
        """Apply exponential smoothing to reduce jitter."""
        if self.prev_features is None:
            self.prev_features = features.copy()
            return features

        smoothed = {}
        for key, value in features.items():
            prev_value = self.prev_features.get(key, value)
            smoothed[key] = (self.smoothing_factor * value +
                             (1 - self.smoothing_factor) * prev_value)

        self.prev_features = smoothed.copy()
        return smoothed

    def _empty_features(self) -> dict:
        """Return zero features when no valid pose detected."""
        return {
            "energy": 0.0,
            "symmetry": 0.0,
            "smoothness": 0.5,
            "armAngle": 0.0,
            "verticalExtension": 0.5,
        }
