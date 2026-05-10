"""
Integration tests for Cuerpo Sonoro pipeline.

Tests that modules connect and communicate correctly using mocked I/O.
No camera, SuperCollider, or Surge XT required.

Tier 2 tests cover:
- OSCSender: message formatting, bundles, address patterns
- MidiSender: chord selection, melody triggering, expression mapping
- Pipeline: FeatureExtractor → OSCSender flow
- Pipeline: FeatureExtractor → MidiSender flow

Usage:
    pytest tests/test_integration.py -v
"""

import math
import time
import pytest
import sys
import os
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vision_processor.features import FeatureExtractor
from vision_processor.osc_sender import OSCSender


# =============================================================================
# HELPERS
# =============================================================================

def make_landmark(x=0.5, y=0.5, z=0.0, visibility=1.0):
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def make_landmarks_neutral():
    """Neutral standing pose (same as test_features.py)."""
    lm = [make_landmark() for _ in range(33)]
    lm[0] = make_landmark(0.50, 0.15)
    lm[7] = make_landmark(0.54, 0.13)
    lm[8] = make_landmark(0.46, 0.13)
    lm[11] = make_landmark(0.60, 0.30)
    lm[12] = make_landmark(0.40, 0.30)
    lm[13] = make_landmark(0.65, 0.45)
    lm[14] = make_landmark(0.35, 0.45)
    lm[15] = make_landmark(0.65, 0.55)
    lm[16] = make_landmark(0.35, 0.55)
    lm[23] = make_landmark(0.55, 0.60)
    lm[24] = make_landmark(0.45, 0.60)
    lm[25] = make_landmark(0.55, 0.75)
    lm[26] = make_landmark(0.45, 0.75)
    lm[27] = make_landmark(0.55, 0.90)
    lm[28] = make_landmark(0.45, 0.90)
    return lm


def copy_landmarks(landmarks):
    return [dict(lm) for lm in landmarks]


# =============================================================================
# OSCSender — unit tests with mocked UDP
# =============================================================================

class TestOSCSender:

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_initializes_with_defaults(self, mock_udp):
        sender = OSCSender()
        mock_udp.assert_called_once_with("127.0.0.1", 57120)

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_initializes_with_custom_host_port(self, mock_udp):
        sender = OSCSender(host="192.168.1.10", port=9000)
        mock_udp.assert_called_once_with("192.168.1.10", 9000)

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_send_single_message(self, mock_udp):
        mock_client = MagicMock()
        mock_udp.return_value = mock_client

        sender = OSCSender()
        sender.send("/motion/energy", 0.75)

        mock_client.send_message.assert_called_once_with("/motion/energy", 0.75)

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_send_features_sends_all_keys(self, mock_udp):
        mock_client = MagicMock()
        mock_udp.return_value = mock_client

        sender = OSCSender()
        features = {"energy": 0.5, "symmetry": -0.2, "smoothness": 0.8}
        sender.send_features(features)

        assert mock_client.send_message.call_count == 3
        mock_client.send_message.assert_any_call("/motion/energy", 0.5)
        mock_client.send_message.assert_any_call("/motion/symmetry", -0.2)
        mock_client.send_message.assert_any_call("/motion/smoothness", 0.8)

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_send_features_uses_motion_prefix(self, mock_udp):
        mock_client = MagicMock()
        mock_udp.return_value = mock_client

        sender = OSCSender()
        sender.send_features({"armAngle": 0.6})

        address = mock_client.send_message.call_args[0][0]
        assert address.startswith("/motion/")

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_send_features_casts_to_float(self, mock_udp):
        mock_client = MagicMock()
        mock_udp.return_value = mock_client

        sender = OSCSender()
        sender.send_features({"energy": 1})  # int, not float

        sent_value = mock_client.send_message.call_args[0][1]
        assert isinstance(sent_value, float)

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_send_bundle_sends_one_bundle(self, mock_udp):
        mock_client = MagicMock()
        mock_udp.return_value = mock_client

        sender = OSCSender()
        features = {"energy": 0.5, "symmetry": 0.0}
        sender.send_bundle(features)

        # send_bundle calls client.send() once with the bundle
        mock_client.send.assert_called_once()

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_send_empty_features_no_messages(self, mock_udp):
        mock_client = MagicMock()
        mock_udp.return_value = mock_client

        sender = OSCSender()
        sender.send_features({})

        mock_client.send_message.assert_not_called()


# =============================================================================
# MidiSender — unit tests with mocked MIDI port
# =============================================================================

class TestMidiSender:

    @pytest.fixture
    def sender(self):
        """MidiSender with mocked MIDI port."""
        with patch("vision_processor.midi.classic.mido.open_output") as mock_open:
            mock_port = MagicMock()
            mock_open.return_value = mock_port
            from vision_processor.midi_sender import MidiSender
            s = MidiSender(port_name="Test Port")
            s._mock_port = mock_port  # store reference for assertions
            yield s

    def _default_features(self, **overrides):
        """Default features dict with optional overrides."""
        f = {
            "feetCenterX": 0.5, "hipTilt": 0.0, "kneeAngle": 1.0,
            "rightHandY": 0.5, "leftHandY": 0.5,
            "rightHandJerk": 0.0, "leftHandJerk": 0.0,
            "rightArmVelocity": 0.0, "leftArmVelocity": 0.0,
            "rightElbowHipAngle": 0.0, "leftElbowHipAngle": 0.0,
            "headTilt": 0.0,
        }
        f.update(overrides)
        return f

    # --- Chord selection ---

    def test_chord_zone_I(self, sender):
        assert sender._get_chord_from_position(0.10) == "I"

    def test_chord_zone_IV(self, sender):
        assert sender._get_chord_from_position(0.35) == "IV"

    def test_chord_zone_V(self, sender):
        assert sender._get_chord_from_position(0.60) == "V"

    def test_chord_zone_VI(self, sender):
        assert sender._get_chord_from_position(0.80) == "VI"

    def test_chord_zone_boundary_0(self, sender):
        assert sender._get_chord_from_position(0.0) == "I"

    def test_chord_zone_boundary_025(self, sender):
        assert sender._get_chord_from_position(0.25) == "IV"

    def test_chord_zone_out_of_range(self, sender):
        """Position >= 1.0 defaults to I."""
        assert sender._get_chord_from_position(1.0) == "I"

    # --- Chord changes send MIDI ---

    def test_chord_change_sends_note_on(self, sender):
        features = self._default_features(feetCenterX=0.10)
        sender.update(features)

        sent = sender._mock_port.send.call_args_list
        note_ons = [c for c in sent if c[0][0].type == "note_on"]
        assert len(note_ons) == 3  # triad: root, 3rd, 5th

    def test_chord_change_sends_correct_notes(self, sender):
        features = self._default_features(feetCenterX=0.10)  # Zone I
        sender.update(features)

        sent = sender._mock_port.send.call_args_list
        note_on_notes = sorted([
            c[0][0].note for c in sent if c[0][0].type == "note_on"
        ])
        assert note_on_notes == sorted([48, 52, 55])  # C Major triad

    def test_chord_change_uses_correct_channels(self, sender):
        features = self._default_features(feetCenterX=0.10)
        sender.update(features)

        sent = sender._mock_port.send.call_args_list
        channels = sorted([
            c[0][0].channel for c in sent if c[0][0].type == "note_on"
        ])
        assert channels == [1, 2, 3]  # CH_CHORD_ROOT, THIRD, FIFTH

    def test_chord_change_turns_off_old_notes(self, sender):
        # First chord: I
        sender.update(self._default_features(feetCenterX=0.10))
        sender._mock_port.send.reset_mock()

        # Second chord: IV
        sender.update(self._default_features(feetCenterX=0.35))

        sent = sender._mock_port.send.call_args_list
        note_offs = [c for c in sent if c[0][0].type == "note_off"]
        assert len(note_offs) == 3  # old triad turned off

    def test_same_chord_no_new_note_on(self, sender):
        sender.update(self._default_features(feetCenterX=0.10))
        sender._mock_port.send.reset_mock()

        # Same zone → no chord change
        sender.update(self._default_features(feetCenterX=0.12))

        sent = sender._mock_port.send.call_args_list
        note_ons = [c for c in sent if c[0][0].type == "note_on"]
        assert len(note_ons) == 0

    # --- Knee angle → chord velocity ---

    def test_straight_knees_high_velocity(self, sender):
        features = self._default_features(feetCenterX=0.10, kneeAngle=1.0)
        sender.update(features)

        sent = sender._mock_port.send.call_args_list
        velocities = [c[0][0].velocity for c in sent if c[0][0].type == "note_on"]
        assert all(v > 100 for v in velocities)

    def test_bent_knees_low_velocity(self, sender):
        features = self._default_features(feetCenterX=0.10, kneeAngle=0.0)
        sender.update(features)

        sent = sender._mock_port.send.call_args_list
        velocities = [c[0][0].velocity for c in sent if c[0][0].type == "note_on"]
        assert all(v < 60 for v in velocities)

    # --- Hip tilt → pitch bend ---

    def test_hip_tilt_center_neutral_bend(self, sender):
        sender.update(self._default_features(feetCenterX=0.10, hipTilt=0.0))

        sent = sender._mock_port.send.call_args_list
        bends = [c[0][0] for c in sent if c[0][0].type == "pitchwheel"]
        # pitch=0 means center (8192 in raw, but mido uses -8192 to 8191)
        assert any(b.pitch == 0 for b in bends)

    def test_hip_tilt_positive_bends_up(self, sender):
        sender.update(self._default_features(feetCenterX=0.10, hipTilt=0.5))

        sent = sender._mock_port.send.call_args_list
        bends = [c[0][0] for c in sent if c[0][0].type == "pitchwheel"]
        assert any(b.pitch > 0 for b in bends)

    # --- Melody triggering ---

    def test_no_jerk_no_melody(self, sender):
        sender.update(self._default_features(rightHandJerk=0.1))

        sent = sender._mock_port.send.call_args_list
        melody_notes = [
            c for c in sent
            if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_RIGHT
        ]
        assert len(melody_notes) == 0

    def test_jerk_above_threshold_triggers_note(self, sender):
        sender.update(self._default_features(rightHandJerk=0.8))

        sent = sender._mock_port.send.call_args_list
        melody_notes = [
            c for c in sent
            if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_RIGHT
        ]
        assert len(melody_notes) == 1

    def test_left_jerk_triggers_left_channel(self, sender):
        sender.update(self._default_features(leftHandJerk=0.8))

        sent = sender._mock_port.send.call_args_list
        left_notes = [
            c for c in sent
            if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_LEFT
        ]
        assert len(left_notes) == 1

    def test_jerk_at_threshold_triggers(self, sender):
        """Jerk exactly at threshold should trigger."""
        sender.update(self._default_features(rightHandJerk=0.4))
        # 0.4 is not > 0.4, so should NOT trigger
        sent = sender._mock_port.send.call_args_list
        melody = [c for c in sent if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_RIGHT]
        assert len(melody) == 0

    def test_jerk_just_above_threshold_triggers(self, sender):
        sender.update(self._default_features(rightHandJerk=0.41))
        sent = sender._mock_port.send.call_args_list
        melody = [c for c in sent if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_RIGHT]
        assert len(melody) == 1

    # --- Hand Y → note selection ---

    def test_hand_low_produces_low_note(self, sender):
        sender.update(self._default_features(rightHandY=0.0, rightHandJerk=0.8))

        sent = sender._mock_port.send.call_args_list
        notes = [c[0][0].note for c in sent if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_RIGHT]
        assert notes[0] == 48  # C3 (lowest in right hand range)

    def test_hand_high_produces_high_note(self, sender):
        sender.update(self._default_features(rightHandY=0.99, rightHandJerk=0.8))

        sent = sender._mock_port.send.call_args_list
        notes = [c[0][0].note for c in sent if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_RIGHT]
        assert notes[0] > 55  # well above G3

    def test_left_hand_uses_higher_octave(self, sender):
        sender.update(self._default_features(
            leftHandY=0.0, leftHandJerk=0.8,
            rightHandY=0.0, rightHandJerk=0.8,
        ))

        sent = sender._mock_port.send.call_args_list
        right_notes = [c[0][0].note for c in sent if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_RIGHT]
        left_notes = [c[0][0].note for c in sent if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_LEFT]
        assert left_notes[0] > right_notes[0]  # left octave (C5) > right octave (C3)

    # --- Note mapping to C Major scale ---

    def test_hand_y_to_note_in_scale(self, sender):
        """All generated notes should be in C Major scale."""
        c_major = {0, 2, 4, 5, 7, 9, 11}  # semitone offsets
        for y in [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.99]:
            note = sender._hand_y_to_note(y, 48)
            offset = (note - 48) % 12
            assert offset in c_major or note == 60, f"y={y} → note {note}, offset {offset} not in C Major"

    # --- Arm velocity → MIDI velocity ---

    def test_fast_arm_high_midi_velocity(self, sender):
        sender.update(self._default_features(rightHandJerk=0.8, rightArmVelocity=1.0))

        sent = sender._mock_port.send.call_args_list
        notes = [c[0][0] for c in sent if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_RIGHT]
        assert notes[0].velocity > 110

    def test_slow_arm_lower_midi_velocity(self, sender):
        sender.update(self._default_features(rightHandJerk=0.8, rightArmVelocity=0.0))

        sent = sender._mock_port.send.call_args_list
        notes = [c[0][0] for c in sent if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_RIGHT]
        assert notes[0].velocity < 80

    # --- Head tilt → global filter CC ---

    def test_head_center_sends_neutral_cc(self, sender):
        sender.update(self._default_features(feetCenterX=0.10, headTilt=0.0))

        sent = sender._mock_port.send.call_args_list
        ccs = [c[0][0] for c in sent if c[0][0].type == "control_change" and c[0][0].control == 74]
        assert any(cc.value == 64 for cc in ccs)  # neutral

    def test_head_tilt_right_bright(self, sender):
        sender.update(self._default_features(feetCenterX=0.10, headTilt=1.0))

        sent = sender._mock_port.send.call_args_list
        ccs = [c[0][0] for c in sent if c[0][0].type == "control_change" and c[0][0].control == 74]
        assert any(cc.value == 127 for cc in ccs)

    def test_head_tilt_left_dark(self, sender):
        sender.update(self._default_features(feetCenterX=0.10, headTilt=-1.0))

        sent = sender._mock_port.send.call_args_list
        ccs = [c[0][0] for c in sent if c[0][0].type == "control_change" and c[0][0].control == 74]
        assert any(cc.value <= 1 for cc in ccs)

    # --- Close ---

    def test_close_sends_all_notes_off(self, sender):
        sender.update(self._default_features(feetCenterX=0.10))
        sender._mock_port.send.reset_mock()

        sender.close()

        sent = sender._mock_port.send.call_args_list
        cc123 = [c for c in sent if c[0][0].type == "control_change" and c[0][0].control == 123]
        assert len(cc123) == 5  # one per used channel


# =============================================================================
# Pipeline: FeatureExtractor → OSCSender
# =============================================================================

class TestPipelineOSC:

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_features_to_osc_full_pipeline(self, mock_udp):
        """Landmarks → features → OSC messages (17 messages sent)."""
        mock_client = MagicMock()
        mock_udp.return_value = mock_client

        extractor = FeatureExtractor()
        sender = OSCSender()
        neutral = make_landmarks_neutral()

        features = extractor.calculate(neutral)
        sender.send_features(features)

        assert mock_client.send_message.call_count == 17

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_all_osc_addresses_have_motion_prefix(self, mock_udp):
        mock_client = MagicMock()
        mock_udp.return_value = mock_client

        extractor = FeatureExtractor()
        sender = OSCSender()
        features = extractor.calculate(make_landmarks_neutral())
        sender.send_features(features)

        for c in mock_client.send_message.call_args_list:
            address = c[0][0]
            assert address.startswith("/motion/"), f"Bad address: {address}"

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_all_osc_values_are_floats(self, mock_udp):
        mock_client = MagicMock()
        mock_udp.return_value = mock_client

        extractor = FeatureExtractor()
        sender = OSCSender()
        features = extractor.calculate(make_landmarks_neutral())
        sender.send_features(features)

        for c in mock_client.send_message.call_args_list:
            value = c[0][1]
            assert isinstance(value, float), f"Value {value} is {type(value)}"

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_movement_changes_osc_values(self, mock_udp):
        """Moving between frames should produce different energy values."""
        mock_client = MagicMock()
        mock_udp.return_value = mock_client

        extractor = FeatureExtractor()
        sender = OSCSender()
        neutral = make_landmarks_neutral()

        # Frame 1: still
        features1 = extractor.calculate(neutral)
        sender.send_features(features1)

        energy_calls_1 = [
            c[0][1] for c in mock_client.send_message.call_args_list
            if c[0][0] == "/motion/energy"
        ]
        mock_client.send_message.reset_mock()

        # Frame 2: moved (need new extractor to avoid smoothing interference)
        ext2 = FeatureExtractor()
        moved = copy_landmarks(neutral)
        for i in range(33):
            moved[i]["x"] += 0.1
        features2 = ext2.calculate(moved, neutral)
        sender.send_features(features2)

        energy_calls_2 = [
            c[0][1] for c in mock_client.send_message.call_args_list
            if c[0][0] == "/motion/energy"
        ]

        assert energy_calls_2[0] > energy_calls_1[0]

    @patch("vision_processor.osc_sender.udp_client.SimpleUDPClient")
    def test_bundle_sends_all_features(self, mock_udp):
        mock_client = MagicMock()
        mock_udp.return_value = mock_client

        extractor = FeatureExtractor()
        sender = OSCSender()
        features = extractor.calculate(make_landmarks_neutral())
        sender.send_bundle(features)

        mock_client.send.assert_called_once()


# =============================================================================
# Pipeline: FeatureExtractor → MidiSender
# =============================================================================

class TestPipelineMIDI:

    @pytest.fixture
    def pipeline(self):
        with patch("vision_processor.midi.classic.mido.open_output") as mock_open:
            mock_port = MagicMock()
            mock_open.return_value = mock_port
            from vision_processor.midi_sender import MidiSender
            extractor = FeatureExtractor()
            sender = MidiSender(port_name="Test")
            yield extractor, sender, mock_port

    def test_neutral_pose_produces_chord(self, pipeline):
        """Neutral standing pose → chord plays (feetCenterX ≈ 0.5 → chord V)."""
        extractor, sender, mock_port = pipeline
        neutral = make_landmarks_neutral()

        features = extractor.calculate(neutral)
        sender.update(features)

        sent = mock_port.send.call_args_list
        note_ons = [c for c in sent if c[0][0].type == "note_on"]
        assert len(note_ons) >= 3  # at least a triad

    def test_feet_position_selects_chord(self, pipeline):
        """Feet at left → chord I, feet at right → chord VI."""
        extractor, sender, mock_port = pipeline

        # Feet on the left side → chord I
        lm_left = make_landmarks_neutral()
        lm_left[27]["x"] = 0.10
        lm_left[28]["x"] = 0.10

        features = extractor.calculate(lm_left)
        sender.update(features)

        sent = mock_port.send.call_args_list
        notes = sorted([c[0][0].note for c in sent if c[0][0].type == "note_on"])
        assert notes == sorted([48, 52, 55])  # C Major triad

    def test_hand_jerk_triggers_melody(self, pipeline):
        """Sudden hand movement → melody note triggered."""
        extractor, sender, mock_port = pipeline
        neutral = make_landmarks_neutral()

        # Frame 1: still (establish baseline)
        f1 = extractor.calculate(neutral)
        sender.update(f1)
        mock_port.send.reset_mock()

        # Frame 2: right hand jumps → jerk
        moved = copy_landmarks(neutral)
        moved[16]["x"] += 0.15
        moved[16]["y"] -= 0.10

        ext2 = FeatureExtractor()
        f2 = ext2.calculate(moved, neutral)

        # Only trigger if jerk exceeds threshold
        if f2["rightHandJerk"] > sender.JERK_THRESHOLD:
            sender.update(f2)
            sent = mock_port.send.call_args_list
            melody = [c for c in sent if c[0][0].type == "note_on" and c[0][0].channel == sender.CH_MELODY_RIGHT]
            assert len(melody) >= 1

    def test_head_tilt_affects_filter(self, pipeline):
        """Tilting head → CC74 changes from neutral."""
        extractor, sender, mock_port = pipeline

        # Tilt head right
        lm = make_landmarks_neutral()
        lm[8]["y"] = 0.20  # right ear drops
        lm[7]["y"] = 0.08  # left ear rises

        features = extractor.calculate(lm)
        sender.update(features)

        sent = mock_port.send.call_args_list
        ccs = [c[0][0] for c in sent if c[0][0].type == "control_change" and c[0][0].control == 74]
        assert len(ccs) > 0
        assert ccs[0].value > 64  # brighter than neutral

    def test_full_frame_sequence(self, pipeline):
        """Simulate 5 frames of movement — no crashes, MIDI sent each frame."""
        extractor, sender, mock_port = pipeline
        neutral = make_landmarks_neutral()

        prev = None
        for i in range(5):
            lm = copy_landmarks(neutral)
            lm[27]["x"] = 0.1 + i * 0.2  # feet walk right
            lm[16]["y"] = 0.5 - i * 0.05  # right hand rises

            features = extractor.calculate(lm, prev)
            sender.update(features)
            prev = lm

        # Should have sent messages every frame
        assert mock_port.send.call_count > 10
