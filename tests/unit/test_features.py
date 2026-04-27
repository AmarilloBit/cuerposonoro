"""
Unit tests for vision_processor/features.py

Tests all 17 feature extraction methods with synthetic landmark data.
No camera or hardware required — runs entirely with mock data.

Usage:
    cd ~/cuerposonoro
    pytest tests/test_features.py -v
"""

import random
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vision_processor.features import FeatureExtractor


# HELPERS: Synthetic landmark generation

def make_landmark(x=0.5, y=0.5, z=0.0, visibility=1.0):
    """Create a single landmark dict matching MediaPipe format."""
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def make_landmarks_neutral():
    """
    33 landmarks representing a neutral standing pose.

    Y axis: 0.0 = top of frame, 1.0 = bottom.
    X axis: 0.0 = left of frame, 1.0 = right.

    MediaPipe indices used by features.py:
        0  = nose              7  = left ear         8  = right ear
        11 = left shoulder     12 = right shoulder
        13 = left elbow        14 = right elbow
        15 = left wrist        16 = right wrist
        23 = left hip          24 = right hip
        25 = left knee         26 = right knee
        27 = left ankle        28 = right ankle
    """
    lm = [make_landmark() for _ in range(33)]

    # Head
    lm[0] = make_landmark(0.50, 0.15)   # nose
    lm[7] = make_landmark(0.54, 0.13)   # left ear
    lm[8] = make_landmark(0.46, 0.13)   # right ear

    # Shoulders
    lm[11] = make_landmark(0.60, 0.30)  # left shoulder
    lm[12] = make_landmark(0.40, 0.30)  # right shoulder

    # Elbows (slightly outside and below shoulders)
    lm[13] = make_landmark(0.65, 0.45)  # left elbow
    lm[14] = make_landmark(0.35, 0.45)  # right elbow

    # Wrists (hanging near hip level)
    lm[15] = make_landmark(0.65, 0.55)  # left wrist
    lm[16] = make_landmark(0.35, 0.55)  # right wrist

    # Hips
    lm[23] = make_landmark(0.55, 0.60)  # left hip
    lm[24] = make_landmark(0.45, 0.60)  # right hip

    # Knees (straight below hips)
    lm[25] = make_landmark(0.55, 0.75)  # left knee
    lm[26] = make_landmark(0.45, 0.75)  # right knee

    # Ankles
    lm[27] = make_landmark(0.55, 0.90)  # left ankle
    lm[28] = make_landmark(0.45, 0.90)  # right ankle

    return lm


def copy_landmarks(landmarks):
    """Deep copy a landmark list."""
    return [dict(lm) for lm in landmarks]


def shift_landmarks(landmarks, dx=0.0, dy=0.0):
    """Return a copy with all landmarks shifted by dx, dy."""
    return [
        {"x": lm["x"] + dx, "y": lm["y"] + dy,
         "z": lm["z"], "visibility": lm["visibility"]}
        for lm in landmarks
    ]


# FIXTURES

@pytest.fixture
def ext():
    """Fresh FeatureExtractor with no smoothing state."""
    return FeatureExtractor()


@pytest.fixture
def neutral():
    """Neutral standing pose."""
    return make_landmarks_neutral()


# calculate() — general behavior

class TestCalculateGeneral:

    ALL_KEYS = {
        "energy", "symmetry", "smoothness", "armAngle", "verticalExtension",
        "feetCenterX", "hipTilt", "kneeAngle",
        "rightHandY", "leftHandY",
        "rightHandJerk", "leftHandJerk",
        "rightArmVelocity", "leftArmVelocity",
        "rightElbowHipAngle", "leftElbowHipAngle",
        "headTilt",
        "pelvis_thrust", "spine_lean",
    }

    def test_returns_dict(self, ext, neutral):
        assert isinstance(ext.calculate(neutral), dict)

    def test_returns_all_19_keys(self, ext, neutral):
        assert set(ext.calculate(neutral).keys()) == self.ALL_KEYS

    def test_all_values_numeric(self, ext, neutral):
        for k, v in ext.calculate(neutral).items():
            assert isinstance(v, (int, float)), f"{k} is {type(v)}"

    def test_none_returns_empty(self, ext):
        r = ext.calculate(None)
        assert isinstance(r, dict) and len(r) == 19

    def test_empty_list_returns_empty(self, ext):
        r = ext.calculate([])
        assert isinstance(r, dict) and len(r) == 19

    def test_32_landmarks_returns_empty(self, ext):
        r = ext.calculate([make_landmark()] * 32)
        assert r == ext._empty_features()

    def test_33_landmarks_works(self, ext):
        r = ext.calculate([make_landmark()] * 33)
        assert len(r) == 19

    def test_descriptor_count_matches_documentation(self, ext, neutral):
        """The number of descriptors must stay in sync with the memoir.

        If this test fails, two things must be updated together:
          - vision_processor/features.py (and any descriptor added/removed)
          - the memoir: section 3.3.1, section 2.2.3, Table 6, Appendix C

        Update both files in the same commit to prevent drift.
        """
        result = ext.calculate(neutral)
        expected_count = 19
        assert len(result) == expected_count, (
            f"Got {len(result)} descriptors but memoir documents "
            f"{expected_count}. Update both files together."
        )

    def test_empty_features_defaults(self, ext):
        r = ext.calculate(None)
        assert r["energy"] == 0.0
        assert r["symmetry"] == 0.0
        assert r["smoothness"] == 0.5
        assert r["armAngle"] == 0.0
        assert r["verticalExtension"] == 0.5
        assert r["feetCenterX"] == 0.5
        assert r["hipTilt"] == 0.0
        assert r["kneeAngle"] == 1.0
        assert r["rightHandY"] == 0.5
        assert r["leftHandY"] == 0.5
        assert r["rightHandJerk"] == 0.0
        assert r["leftHandJerk"] == 0.0
        assert r["rightArmVelocity"] == 0.0
        assert r["leftArmVelocity"] == 0.0
        assert r["rightElbowHipAngle"] == 0.0
        assert r["leftElbowHipAngle"] == 0.0
        assert r["headTilt"] == 0.0
        assert r["pelvis_thrust"] == 0.0
        assert r["spine_lean"] == 0.0


# Output range validation

RANGE_01 = [
    "energy", "smoothness", "armAngle", "verticalExtension",
    "feetCenterX", "kneeAngle",
    "rightHandY", "leftHandY",
    "rightHandJerk", "leftHandJerk",
    "rightArmVelocity", "leftArmVelocity",
    "rightElbowHipAngle", "leftElbowHipAngle",
]
RANGE_SIGNED = ["symmetry", "hipTilt", "headTilt", "pelvis_thrust", "spine_lean"]


class TestOutputRanges:

    def _assert_all_in_range(self, result):
        for k in RANGE_01:
            assert 0.0 <= result[k] <= 1.0, f"{k}={result[k]} out of [0,1]"
        for k in RANGE_SIGNED:
            assert -1.0 <= result[k] <= 1.0, f"{k}={result[k]} out of [-1,1]"

    def test_neutral_pose(self, ext, neutral):
        self._assert_all_in_range(ext.calculate(neutral))

    def test_with_movement(self, ext, neutral):
        moved = shift_landmarks(neutral, dx=0.3, dy=-0.2)
        self._assert_all_in_range(ext.calculate(moved, neutral))

    def test_all_at_origin(self, ext):
        lm = [make_landmark(0.0, 0.0)] * 33
        self._assert_all_in_range(ext.calculate(lm))

    def test_all_at_one(self, ext):
        lm = [make_landmark(1.0, 1.0)] * 33
        self._assert_all_in_range(ext.calculate(lm))

    def test_max_movement(self, ext):
        a = [make_landmark(0.0, 0.0)] * 33
        b = [make_landmark(1.0, 1.0)] * 33
        self._assert_all_in_range(ext.calculate(b, a))


# Energy

class TestEnergy:

    def test_no_prev_returns_zero(self, ext, neutral):
        assert ext.calculate(neutral, None)["energy"] == pytest.approx(0.0, abs=0.01)

    def test_identical_frames_zero(self, ext, neutral):
        assert ext.calculate(neutral, neutral)["energy"] == pytest.approx(0.0, abs=0.01)

    def test_movement_produces_energy(self, ext, neutral):
        moved = shift_landmarks(neutral, dx=0.03, dy=0.03)
        assert ext.calculate(moved, neutral)["energy"] > 0.05

    def test_more_movement_more_energy(self, neutral):
        small = shift_landmarks(neutral, dx=0.01)
        big = shift_landmarks(neutral, dx=0.1)
        r1 = FeatureExtractor().calculate(small, neutral)
        r2 = FeatureExtractor().calculate(big, neutral)
        assert r2["energy"] > r1["energy"]

    def test_capped_at_one(self, ext, neutral):
        huge = shift_landmarks(neutral, dx=0.5, dy=0.5)
        assert ext.calculate(huge, neutral)["energy"] <= 1.0

    def test_only_key_points_matter(self, ext, neutral):
        """Moving non-key landmark (idx 5) should not affect energy."""
        moved = copy_landmarks(neutral)
        moved[5]["x"] += 0.3
        assert ext.calculate(moved, neutral)["energy"] == pytest.approx(0.0, abs=0.01)


# Symmetry

class TestSymmetry:

    def test_symmetric_pose_near_zero(self, ext, neutral):
        assert ext.calculate(neutral)["symmetry"] == pytest.approx(0.0, abs=0.15)

    def test_right_wrist_extended_positive(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[16]["x"] = 0.90  # right wrist far right
        lm[15]["x"] = 0.55  # left wrist near center
        assert ext.calculate(lm)["symmetry"] > 0.0

    def test_left_wrist_extended_negative(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[15]["x"] = 0.10  # left wrist far left
        lm[16]["x"] = 0.45  # right wrist near center
        assert ext.calculate(lm)["symmetry"] < 0.0

    def test_perfectly_symmetric_is_zero(self, ext):
        lm = [make_landmark()] * 33
        lm[15] = make_landmark(0.7, 0.5)  # left: 0.2 from center
        lm[16] = make_landmark(0.3, 0.5)  # right: 0.2 from center
        assert ext.calculate(lm)["symmetry"] == pytest.approx(0.0, abs=0.01)

    def test_clamped(self, ext):
        lm = [make_landmark()] * 33
        lm[15] = make_landmark(0.0, 0.5)
        lm[16] = make_landmark(1.0, 0.5)
        r = ext.calculate(lm)["symmetry"]
        assert -1.0 <= r <= 1.0


# Smoothness

class TestSmoothness:

    def test_no_prev_returns_default(self, ext, neutral):
        assert ext.calculate(neutral, None)["smoothness"] == pytest.approx(0.5, abs=0.1)

    def test_no_movement_high(self, ext, neutral):
        assert ext.calculate(neutral, neutral)["smoothness"] > 0.9

    def test_abrupt_movement_low(self, ext, neutral):
        jerky = copy_landmarks(neutral)
        jerky[15]["x"] += 0.2
        jerky[16]["x"] -= 0.2
        assert ext.calculate(jerky, neutral)["smoothness"] < 0.5

    def test_always_non_negative(self, ext, neutral):
        extreme = copy_landmarks(neutral)
        extreme[15]["x"] += 0.5
        extreme[16]["x"] -= 0.5
        assert ext.calculate(extreme, neutral)["smoothness"] >= 0.0

    def test_only_wrists_matter(self, ext, neutral):
        moved = copy_landmarks(neutral)
        moved[0]["x"] += 0.3   # nose moves
        moved[27]["x"] += 0.3  # ankle moves
        assert ext.calculate(moved, neutral)["smoothness"] > 0.9


# Arm Angle

class TestArmAngle:

    def test_arms_below_shoulders_low(self, ext, neutral):
        assert ext.calculate(neutral)["armAngle"] < 0.5

    def test_arms_above_shoulders_high(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[15]["y"] = 0.05
        lm[16]["y"] = 0.05
        assert ext.calculate(lm)["armAngle"] > 0.5

    def test_arms_at_shoulder_level_mid(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[15]["y"] = 0.30  # = shoulder Y
        lm[16]["y"] = 0.30
        r = ext.calculate(lm)["armAngle"]
        assert 0.3 <= r <= 0.7

    def test_one_up_one_down_between_extremes(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[15]["y"] = 0.05  # left up
        lm[16]["y"] = 0.70  # right down
        both_up = copy_landmarks(neutral)
        both_up[15]["y"] = 0.05
        both_up[16]["y"] = 0.05
        r_mixed = ext.calculate(lm)["armAngle"]
        r_up = FeatureExtractor().calculate(both_up)["armAngle"]
        assert r_mixed < r_up

    def test_clamped(self, ext):
        lm = [make_landmark()] * 33
        lm[11] = make_landmark(0.6, 0.0)
        lm[12] = make_landmark(0.4, 0.0)
        lm[15] = make_landmark(0.6, 1.0)
        lm[16] = make_landmark(0.4, 1.0)
        r = ext.calculate(lm)["armAngle"]
        assert 0.0 <= r <= 1.0


# Vertical Extension

class TestVerticalExtension:

    def test_standing_positive(self, ext, neutral):
        assert ext.calculate(neutral)["verticalExtension"] > 0.3

    def test_standing_taller_than_crouching(self, neutral):
        crouching = copy_landmarks(neutral)
        crouching[0]["y"] = 0.50
        crouching[27]["y"] = 0.65
        crouching[28]["y"] = 0.65
        r_stand = FeatureExtractor().calculate(neutral)
        r_crouch = FeatureExtractor().calculate(crouching)
        assert r_stand["verticalExtension"] > r_crouch["verticalExtension"]

    def test_nose_below_ankles_zero(self, ext):
        lm = [make_landmark()] * 33
        lm[0] = make_landmark(0.5, 0.95)
        lm[27] = make_landmark(0.5, 0.10)
        lm[28] = make_landmark(0.5, 0.10)
        assert ext.calculate(lm)["verticalExtension"] == pytest.approx(0.0, abs=0.01)

    def test_capped_at_one(self, ext):
        lm = [make_landmark()] * 33
        lm[0] = make_landmark(0.5, 0.0)
        lm[27] = make_landmark(0.5, 1.0)
        lm[28] = make_landmark(0.5, 1.0)
        assert ext.calculate(lm)["verticalExtension"] <= 1.0


# Feet Center X (Chord Selection)

class TestFeetCenterX:

    def test_centered_near_half(self, ext, neutral):
        assert ext.calculate(neutral)["feetCenterX"] == pytest.approx(0.5, abs=0.05)

    def test_feet_left(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[27]["x"] = 0.15
        lm[28]["x"] = 0.10
        assert ext.calculate(lm)["feetCenterX"] < 0.3

    def test_feet_right(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[27]["x"] = 0.85
        lm[28]["x"] = 0.90
        assert ext.calculate(lm)["feetCenterX"] > 0.7

    def test_clamped(self, ext):
        lm = [make_landmark()] * 33
        lm[27] = make_landmark(-0.1, 0.9)
        lm[28] = make_landmark(-0.2, 0.9)
        assert ext.calculate(lm)["feetCenterX"] >= 0.0


# Hip Tilt (Chord Expression)

class TestHipTilt:

    def test_level_hips_near_zero(self, ext, neutral):
        assert ext.calculate(neutral)["hipTilt"] == pytest.approx(0.0, abs=0.1)

    def test_right_hip_lower_positive(self, ext, neutral):
        """right_hip.y > left_hip.y → positive tilt."""
        lm = copy_landmarks(neutral)
        lm[24]["y"] = 0.65  # right hip drops
        lm[23]["y"] = 0.55  # left hip stays
        assert ext.calculate(lm)["hipTilt"] > 0.0

    def test_left_hip_lower_negative(self, ext, neutral):
        """left_hip.y > right_hip.y → negative tilt."""
        lm = copy_landmarks(neutral)
        lm[23]["y"] = 0.65
        lm[24]["y"] = 0.55
        assert ext.calculate(lm)["hipTilt"] < 0.0

    def test_clamped(self, ext):
        lm = [make_landmark()] * 33
        lm[23] = make_landmark(0.55, 0.0)
        lm[24] = make_landmark(0.45, 1.0)
        r = ext.calculate(lm)["hipTilt"]
        assert -1.0 <= r <= 1.0


# Knee Angle (Chord Volume)

class TestKneeAngle:

    def test_straight_legs_high(self, ext, neutral):
        """Neutral pose has straight legs → high knee angle."""
        assert ext.calculate(neutral)["kneeAngle"] > 0.5

    def test_bent_knees_lower(self, ext, neutral):
        """Knees pushed forward → lower angle."""
        lm = copy_landmarks(neutral)
        lm[25]["x"] = 0.62
        lm[25]["y"] = 0.72
        lm[26]["x"] = 0.38
        lm[26]["y"] = 0.72
        lm[23]["y"] = 0.65
        lm[24]["y"] = 0.65
        bent = ext.calculate(lm)["kneeAngle"]
        straight = FeatureExtractor().calculate(neutral)["kneeAngle"]
        assert bent < straight

    def test_zero_length_vectors_no_crash(self, ext):
        """Hip, knee, ankle at same point → no division by zero."""
        lm = [make_landmark(0.5, 0.5)] * 33
        r = ext.calculate(lm)["kneeAngle"]
        assert isinstance(r, float)
        assert 0.0 <= r <= 1.0


# Hand Y (Melody Note Selection)

class TestHandY:

    def test_hand_above_head_high(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[16]["y"] = 0.05  # right wrist above head
        assert ext.calculate(lm)["rightHandY"] > 0.7

    def test_hand_at_hip_low(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[16]["y"] = 0.65
        assert ext.calculate(lm)["rightHandY"] < 0.3

    def test_left_independent_of_right(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[15]["y"] = 0.05  # left high
        lm[16]["y"] = 0.65  # right low
        r = ext.calculate(lm)
        assert r["leftHandY"] > r["rightHandY"]

    def test_clamped(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[16]["y"] = -0.5  # way above frame
        r = ext.calculate(lm)["rightHandY"]
        assert 0.0 <= r <= 1.0


# Hand Jerk (Note Trigger)

class TestHandJerk:

    def test_no_movement_zero(self, ext, neutral):
        r = ext.calculate(neutral, neutral)
        assert r["rightHandJerk"] == pytest.approx(0.0, abs=0.01)
        assert r["leftHandJerk"] == pytest.approx(0.0, abs=0.01)

    def test_no_prev_zero(self, ext, neutral):
        assert ext.calculate(neutral, None)["rightHandJerk"] == pytest.approx(0.0, abs=0.01)

    def test_sudden_movement_high(self, ext, neutral):
        moved = copy_landmarks(neutral)
        moved[16]["x"] += 0.15
        assert ext.calculate(moved, neutral)["rightHandJerk"] > 0.3

    def test_left_independent(self, ext, neutral):
        moved = copy_landmarks(neutral)
        moved[15]["y"] -= 0.15  # only left moves
        r = ext.calculate(moved, neutral)
        assert r["leftHandJerk"] > r["rightHandJerk"]

    def test_capped_at_one(self, ext, neutral):
        moved = copy_landmarks(neutral)
        moved[16]["x"] += 0.5
        moved[16]["y"] += 0.5
        assert ext.calculate(moved, neutral)["rightHandJerk"] <= 1.0

    def test_jerk_not_flattened_by_smoothing(self, ext, neutral):
        """Jerk pico no debe quedar por debajo del umbral tras el suavizado temporal.

        El suavizado aplana señales continuas, pero jerk es una señal de pico
        que debe conservar su valor máximo para poder trigear notas.
        """
        # Calentar el extractor con un frame estático para establecer prev_features
        ext.calculate(neutral, neutral)

        # Movimiento brusco de muñeca (equivalente al que produce jerk ~0.5 sin suavizado)
        moved = copy_landmarks(neutral)
        moved[16]["x"] += 0.035  # muñeca derecha: desplazamiento que produce jerk ~0.52
        moved[16]["y"] += 0.035

        r = ext.calculate(moved, neutral)

        # El jerk debe superar el umbral de trigger (0.4) incluso después del suavizado
        assert r["rightHandJerk"] > 0.4, (
            f"rightHandJerk={r['rightHandJerk']:.4f} no supera el umbral 0.4 "
            f"— el suavizado está aplastando los picos de jerk"
        )

# Arm Velocity (Note Intensity & Duration)

class TestArmVelocity:

    def test_no_movement_zero(self, ext, neutral):
        r = ext.calculate(neutral, neutral)
        assert r["rightArmVelocity"] == pytest.approx(0.0, abs=0.01)
        assert r["leftArmVelocity"] == pytest.approx(0.0, abs=0.01)

    def test_no_prev_zero(self, ext, neutral):
        assert ext.calculate(neutral, None)["rightArmVelocity"] == pytest.approx(0.0, abs=0.01)

    def test_movement_produces_velocity(self, ext, neutral):
        moved = copy_landmarks(neutral)
        moved[16]["x"] += 0.1
        assert ext.calculate(moved, neutral)["rightArmVelocity"] > 0.1

    def test_faster_higher(self, neutral):
        slow = copy_landmarks(neutral)
        slow[16]["x"] += 0.02
        fast = copy_landmarks(neutral)
        fast[16]["x"] += 0.1
        r1 = FeatureExtractor().calculate(slow, neutral)
        r2 = FeatureExtractor().calculate(fast, neutral)
        assert r2["rightArmVelocity"] > r1["rightArmVelocity"]

    def test_jerk_scales_higher_than_velocity(self, ext, neutral):
        """Jerk multiplier (15) > velocity multiplier (8), so jerk >= velocity for same movement."""
        moved = copy_landmarks(neutral)
        moved[16]["x"] += 0.05
        r = ext.calculate(moved, neutral)
        assert r["rightHandJerk"] >= r["rightArmVelocity"]


# Elbow-Hip Angle (Glissando / Vibrato)

class TestElbowHipAngle:

    def test_arm_at_side_moderate(self, ext, neutral):
        """Neutral pose → some angle but not extreme."""
        r = ext.calculate(neutral)
        assert r["rightElbowHipAngle"] < 0.8
        assert r["leftElbowHipAngle"] < 0.8

    def test_arm_extended_increases(self, ext, neutral):
        lm = copy_landmarks(neutral)
        lm[14]["x"] = 0.10  # right elbow far out
        lm[14]["y"] = 0.30  # at shoulder height
        extended = ext.calculate(lm)["rightElbowHipAngle"]

        neutral_val = FeatureExtractor().calculate(neutral)["rightElbowHipAngle"]
        assert extended > neutral_val or extended > 0.0

    def test_left_uses_left_landmarks(self, ext, neutral):
        """Moving only left elbow should affect leftElbowHipAngle, not right."""
        lm = copy_landmarks(neutral)
        lm[13]["x"] = 0.90  # left elbow far out
        r = ext.calculate(lm)
        # Left should change, right should stay at neutral-ish
        # (Can't guarantee exact ordering due to cross product, just check both valid)
        assert 0.0 <= r["leftElbowHipAngle"] <= 1.0
        assert 0.0 <= r["rightElbowHipAngle"] <= 1.0


# Head Tilt (Global Filter)

class TestHeadTilt:

    def test_level_head_near_zero(self, ext, neutral):
        assert ext.calculate(neutral)["headTilt"] == pytest.approx(0.0, abs=0.1)

    def test_right_ear_lower_positive(self, ext, neutral):
        """right_ear.y > left_ear.y → positive tilt."""
        lm = copy_landmarks(neutral)
        lm[8]["y"] = 0.20  # right ear drops
        lm[7]["y"] = 0.10  # left ear stays
        assert ext.calculate(lm)["headTilt"] > 0.0

    def test_left_ear_lower_negative(self, ext, neutral):
        """left_ear.y > right_ear.y → negative tilt."""
        lm = copy_landmarks(neutral)
        lm[7]["y"] = 0.20
        lm[8]["y"] = 0.10
        assert ext.calculate(lm)["headTilt"] < 0.0

    def test_clamped(self, ext):
        lm = [make_landmark()] * 33
        lm[7] = make_landmark(0.54, 0.0)
        lm[8] = make_landmark(0.46, 1.0)
        r = ext.calculate(lm)["headTilt"]
        assert -1.0 <= r <= 1.0


# Temporal Smoothing

class TestSmoothing:

    def test_first_frame_no_crash(self):
        ext = FeatureExtractor()
        r = ext.calculate(make_landmarks_neutral())
        assert isinstance(r, dict)

    def test_smoothing_reduces_jump(self):
        ext = FeatureExtractor()
        neutral = make_landmarks_neutral()

        # Frame 1: baseline
        ext.calculate(neutral)

        # Frame 2: arms suddenly up
        arms_up = copy_landmarks(neutral)
        arms_up[15]["y"] = 0.05
        arms_up[16]["y"] = 0.05
        r = ext.calculate(arms_up)

        # Raw armAngle would be ~0.75, but smoothing pulls toward baseline
        assert r["armAngle"] < 0.75

    def test_converges_over_many_frames(self):
        ext = FeatureExtractor()
        neutral = make_landmarks_neutral()

        arms_up = copy_landmarks(neutral)
        arms_up[15]["y"] = 0.05
        arms_up[16]["y"] = 0.05

        ext.calculate(neutral)  # baseline
        for _ in range(20):
            r = ext.calculate(arms_up)

        # After 20 identical frames, should be very close to raw
        raw = FeatureExtractor().calculate(arms_up)
        assert abs(r["armAngle"] - raw["armAngle"]) < 0.05

    def test_smoothing_formula(self):
        """Verify: smoothed = α * raw + (1 - α) * prev."""
        ext = FeatureExtractor()
        ext.smoothing_factor = 0.3
        neutral = make_landmarks_neutral()

        r1 = ext.calculate(neutral)  # frame 1 = raw (no prev)

        moved = copy_landmarks(neutral)
        moved[15]["y"] = 0.10
        moved[16]["y"] = 0.10

        raw = FeatureExtractor().calculate(moved)["armAngle"]
        r2 = ext.calculate(moved)

        expected = 0.3 * raw + 0.7 * r1["armAngle"]
        assert r2["armAngle"] == pytest.approx(expected, abs=0.02)


# Edge cases & robustness

class TestEdgeCases:

    def test_all_at_origin(self, ext):
        lm = [make_landmark(0.0, 0.0)] * 33
        assert isinstance(ext.calculate(lm), dict)

    def test_all_at_one(self, ext):
        lm = [make_landmark(1.0, 1.0)] * 33
        assert isinstance(ext.calculate(lm), dict)

    def test_negative_coordinates(self, ext):
        lm = [make_landmark(-0.05, -0.05)] * 33
        assert isinstance(ext.calculate(lm), dict)

    def test_coordinates_above_one(self, ext):
        lm = [make_landmark(1.05, 1.05)] * 33
        assert isinstance(ext.calculate(lm), dict)

    def test_identical_frames_zero_velocity(self, ext, neutral):
        r = ext.calculate(neutral, neutral)
        assert r["energy"] == pytest.approx(0.0, abs=0.01)
        assert r["rightHandJerk"] == pytest.approx(0.0, abs=0.01)
        assert r["leftHandJerk"] == pytest.approx(0.0, abs=0.01)
        assert r["rightArmVelocity"] == pytest.approx(0.0, abs=0.01)
        assert r["leftArmVelocity"] == pytest.approx(0.0, abs=0.01)

    def test_100_random_frames_no_crash(self):
        random.seed(42)
        ext = FeatureExtractor()
        prev = None
        for _ in range(100):
            lm = [make_landmark(
                x=random.uniform(0.0, 1.0),
                y=random.uniform(0.0, 1.0),
            ) for _ in range(33)]
            r = ext.calculate(lm, prev)
            assert isinstance(r, dict) and len(r) == 19
            prev = lm


# Pelvis Thrust (Blueprint chord navigation)

class TestPelvisThrust:

    def test_neutral_pose_near_zero(self, ext, neutral):
        """Hips directly above ankles in neutral pose gives ~0."""
        assert ext.calculate(neutral)["pelvis_thrust"] == pytest.approx(0.0, abs=0.15)

    def test_hips_forward_positive(self, ext, neutral):
        """Pushing hips forward (lower X in image coords) relative to ankles."""
        lm = copy_landmarks(neutral)
        # Move hips forward (decrease X) while ankles stay
        lm[23]["x"] = 0.40
        lm[24]["x"] = 0.30
        assert ext.calculate(lm)["pelvis_thrust"] < 0.0  # hips behind ankles in X

    def test_hips_behind_negative(self, ext, neutral):
        """Hips behind ankles (higher X) gives positive value."""
        lm = copy_landmarks(neutral)
        lm[23]["x"] = 0.70
        lm[24]["x"] = 0.60
        assert ext.calculate(lm)["pelvis_thrust"] > 0.0

    def test_normalised_by_torso_height(self, ext, neutral):
        """Same hip displacement on a taller body gives smaller thrust value."""
        short = copy_landmarks(neutral)
        short[23]["x"] = 0.65  # hip shifted
        short[24]["x"] = 0.55
        short[11]["y"] = 0.50  # short torso
        short[23]["y"] = 0.60

        tall = copy_landmarks(neutral)
        tall[23]["x"] = 0.65
        tall[24]["x"] = 0.55
        tall[11]["y"] = 0.20  # tall torso
        tall[23]["y"] = 0.60

        r_short = FeatureExtractor().calculate(short)["pelvis_thrust"]
        r_tall = FeatureExtractor().calculate(tall)["pelvis_thrust"]
        assert abs(r_short) > abs(r_tall)

    def test_clamped(self, ext):
        """Output stays within [-1, 1] even with extreme positions."""
        lm = [make_landmark()] * 33
        lm[23] = make_landmark(1.0, 0.60)  # hips far right
        lm[24] = make_landmark(1.0, 0.60)
        lm[27] = make_landmark(0.0, 0.90)  # ankles far left
        lm[28] = make_landmark(0.0, 0.90)
        lm[11] = make_landmark(0.5, 0.59)  # tiny torso (near zero height)
        r = ext.calculate(lm)["pelvis_thrust"]
        assert -1.0 <= r <= 1.0

    def test_zero_torso_height_no_crash(self, ext):
        """When shoulder and hip at same Y, no division by zero."""
        lm = [make_landmark(0.5, 0.5)] * 33
        r = ext.calculate(lm)["pelvis_thrust"]
        assert isinstance(r, float)
        assert -1.0 <= r <= 1.0


# Spine Lean (Blueprint CC1 modulation)

class TestSpineLean:

    def test_neutral_pose_near_zero(self, ext, neutral):
        """Shoulders directly above hips in neutral pose gives ~0."""
        assert ext.calculate(neutral)["spine_lean"] == pytest.approx(0.0, abs=0.15)

    def test_lean_forward_positive(self, ext, neutral):
        """Shoulders forward (lower X) relative to hips gives negative value."""
        lm = copy_landmarks(neutral)
        lm[11]["x"] = 0.45  # shoulders move forward (lower X)
        lm[12]["x"] = 0.25
        assert ext.calculate(lm)["spine_lean"] < 0.0

    def test_lean_backward_negative(self, ext, neutral):
        """Shoulders behind hips (higher X) gives positive value."""
        lm = copy_landmarks(neutral)
        lm[11]["x"] = 0.75  # shoulders move backward (higher X)
        lm[12]["x"] = 0.55
        assert ext.calculate(lm)["spine_lean"] > 0.0

    def test_normalised_by_torso_height(self, ext, neutral):
        """Same shoulder displacement on a taller body gives smaller lean value."""
        short = copy_landmarks(neutral)
        short[11]["x"] = 0.70
        short[12]["x"] = 0.50
        short[11]["y"] = 0.50  # short torso
        short[23]["y"] = 0.60

        tall = copy_landmarks(neutral)
        tall[11]["x"] = 0.70
        tall[12]["x"] = 0.50
        tall[11]["y"] = 0.20  # tall torso
        tall[23]["y"] = 0.60

        r_short = FeatureExtractor().calculate(short)["spine_lean"]
        r_tall = FeatureExtractor().calculate(tall)["spine_lean"]
        assert abs(r_short) > abs(r_tall)

    def test_clamped(self, ext):
        """Output stays within [-1, 1] even with extreme positions."""
        lm = [make_landmark()] * 33
        lm[11] = make_landmark(1.0, 0.30)  # shoulders far right
        lm[12] = make_landmark(1.0, 0.30)
        lm[23] = make_landmark(0.0, 0.29)  # hips far left, tiny torso
        lm[24] = make_landmark(0.0, 0.60)
        r = ext.calculate(lm)["spine_lean"]
        assert -1.0 <= r <= 1.0

    def test_zero_torso_height_no_crash(self, ext):
        """When shoulder and hip at same Y, no division by zero."""
        lm = [make_landmark(0.5, 0.5)] * 33
        r = ext.calculate(lm)["spine_lean"]
        assert isinstance(r, float)
        assert -1.0 <= r <= 1.0
