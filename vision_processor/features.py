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
        Average arm elevation angle.
        0 = arms down, 1 = arms horizontal or above.
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
        0 = crouching, 1 = fully stretched.
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

    # FEATURES - MIDI - CHORDS (Lower body)

    def _calculate_feet_center_x(self, landmarks: list) -> float:
        """
        Horizontal center position of feet for chord zone selection.
        0.0 = left side of frame, 1.0 = right side of frame.
        Used to select chord degree (I, IV, V, VI) based on position in space.
        """
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        center_x = (left_ankle["x"] + right_ankle["x"]) / 2
        return max(0.0, min(1.0, center_x))

    def _calculate_hip_tilt(self, landmarks: list) -> float:
        """
        Lateral hip tilt for chord expression (pitch bend + extensions).
        -1.0 = tilted left, 0.0 = straight, 1.0 = tilted right.
        Controls pitch bend and can trigger 6th/7th chord extensions.
        """
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        # Height difference between hips
        tilt = right_hip["y"] - left_hip["y"]

        # Normalize (typical range is small, ~0.1)
        return max(-1.0, min(1.0, tilt * 5))

    def _calculate_knee_angle(self, landmarks: list) -> float:
        """
        Average knee bend angle for chord volume control.
        0.0 = knees very bent (low volume), 1.0 = legs straight (full volume).
        """
        def knee_angle_single(hip, knee, ankle):
            # Vectors from knee to hip and knee to ankle
            v1 = (hip["x"] - knee["x"], hip["y"] - knee["y"])
            v2 = (ankle["x"] - knee["x"], ankle["y"] - knee["y"])

            # Dot product and magnitudes
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if mag1 * mag2 == 0:
                return 1.0

            # Angle in radians, normalized to 0-1
            cos_angle = dot / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.acos(cos_angle)

            # 180° (pi) = straight = 1.0, 90° (pi/2) = bent = 0.0
            normalized = (angle - math.pi / 2) / (math.pi / 2)
            return max(0.0, min(1.0, normalized))

        left_angle = knee_angle_single(landmarks[23], landmarks[25], landmarks[27])
        right_angle = knee_angle_single(landmarks[24], landmarks[26], landmarks[28])

        return (left_angle + right_angle) / 2

    # FEATURES - MIDI - Melody (Hands)

    def _calculate_hand_y(self, landmarks: list, side: str) -> float:
        """
        Vertical hand position normalized to body height for note selection.
        0.0 = hand down (near hip), 1.0 = hand up (above head).
        Right hand controls lower octave (C3-B3), left hand controls higher octave (C5-B5).
        """
        wrist_idx = 16 if side == "right" else 15
        hip_idx = 24 if side == "right" else 23

        wrist = landmarks[wrist_idx]
        hip = landmarks[hip_idx]
        nose = landmarks[0]

        # Range: from hip to above head
        min_y = nose["y"] - 0.2  # Slightly above head
        max_y = hip["y"] + 0.1   # Slightly below hip

        # Invert because Y=0 is at top of image
        normalized = (max_y - wrist["y"]) / (max_y - min_y)

        return max(0.0, min(1.0, normalized))

    def _calculate_hand_jerk(self, landmarks: list, prev_landmarks: Optional[list], side: str) -> float:
        """
        Detects sudden hand movement (jerk) for note triggering.
        0.0 = no movement, 1.0 = very sudden movement.
        High jerk triggers a new note ON event.
        """
        if prev_landmarks is None:
            return 0.0

        wrist_idx = 16 if side == "right" else 15

        curr = landmarks[wrist_idx]
        prev = prev_landmarks[wrist_idx]

        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]

        velocity = math.sqrt(dx ** 2 + dy ** 2)

        # High threshold to detect only sudden movements
        jerk = velocity * 15  # Scale for sensitivity

        return max(0.0, min(1.0, jerk))

    def _calculate_arm_velocity(self, landmarks: list, prev_landmarks: Optional[list], side: str) -> float:
        """
        Arm velocity for note velocity (loudness) and duration control.
        0.0 = still, 1.0 = fast movement.
        Fast movement = loud short notes (staccato), slow = soft longer notes.
        """
        if prev_landmarks is None:
            return 0.0

        wrist_idx = 16 if side == "right" else 15

        curr = landmarks[wrist_idx]
        prev = prev_landmarks[wrist_idx]

        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]

        velocity = math.sqrt(dx ** 2 + dy ** 2)

        # Smoother scaling than jerk
        return max(0.0, min(1.0, velocity * 8))

    def _calculate_elbow_hip_angle(self, landmarks: list, side: str) -> float:
        """
        Angle between arm and torso for pitch bend / vibrato control.
        0.0 = arm close to body (stable pitch), 1.0 = arm extended (glissando/vibrato).
        Oscillating movement creates vibrato effect.
        """
        if side == "right":
            shoulder = landmarks[12]
            elbow = landmarks[14]
            hip = landmarks[24]
        else:
            shoulder = landmarks[11]
            elbow = landmarks[13]
            hip = landmarks[23]

        # Vector shoulder → elbow
        v1 = (elbow["x"] - shoulder["x"], elbow["y"] - shoulder["y"])
        # Vector shoulder → hip
        v2 = (hip["x"] - shoulder["x"], hip["y"] - shoulder["y"])

        # Cross product to determine angle magnitude
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        # Normalize
        return max(0.0, min(1.0, abs(cross) * 3))

    # FEATURES - MIDI - Global Expression (Utility methods)

    def _calculate_head_tilt(self, landmarks: list) -> float:
        """
        Lateral head tilt for global filter control.
        -1.0 = tilted left (darker sound), 0.0 = straight, 1.0 = tilted right (brighter sound).
        Controls a global cutoff filter affecting all sounds.
        """
        left_ear = landmarks[7]
        right_ear = landmarks[8]

        # Height difference between ears
        tilt = right_ear["y"] - left_ear["y"]

        return max(-1.0, min(1.0, tilt * 5))

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
        """Return default features when no valid pose detected."""
        return {
            # Existing features
            "energy": 0.0,
            "symmetry": 0.0,
            "smoothness": 0.5,
            "armAngle": 0.0,
            "verticalExtension": 0.5,
            # New MPE features
            "feetCenterX": 0.5,
            "hipTilt": 0.0,
            "kneeAngle": 1.0,
            "rightHandY": 0.5,
            "leftHandY": 0.5,
            "rightHandJerk": 0.0,
            "leftHandJerk": 0.0,
            "rightArmVelocity": 0.0,
            "leftArmVelocity": 0.0,
            "rightElbowHipAngle": 0.0,
            "leftElbowHipAngle": 0.0,
            "headTilt": 0.0,
        }
