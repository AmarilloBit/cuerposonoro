"""
Unit tests for vision_processor/midi/musical.py

Tests the MusicalMidiSender: chord selection, melody direction,
beat quantization, global expression, and close. Uses a mocked
MIDI port — no hardware required.

Usage:
    pytest tests/unit/test_musical.py -v
"""

import time
import pytest
import sys
import os
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.fixture
def sender():
    """Create a MusicalMidiSender with mocked MIDI port."""
    with patch("vision_processor.midi.musical.mido.open_output") as mock_open:
        mock_port = MagicMock()
        mock_open.return_value = mock_port

        from vision_processor.midi.musical import MusicalMidiSender
        s = MusicalMidiSender(
            port_name="Test",
            tempo_bpm=120,
            note_subdivision=8,
            direction_threshold=0.03,
            velocity_threshold=0.4,
        )
        yield s, mock_port
        s.close()


def _msgs(mock_port, msg_type=None, channel=None):
    """Extract sent MIDI messages, optionally filtered."""
    msgs = [c[0][0] for c in mock_port.send.call_args_list]
    if msg_type:
        msgs = [m for m in msgs if m.type == msg_type]
    if channel is not None:
        msgs = [m for m in msgs if m.channel == channel]
    return msgs


# ===========================================================================
# Initialization
# ===========================================================================

class TestInit:

    def test_opens_virtual_port(self):
        with patch("vision_processor.midi.musical.mido.open_output") as mock_open:
            mock_open.return_value = MagicMock()
            from vision_processor.midi.musical import MusicalMidiSender
            s = MusicalMidiSender(port_name="TestPort")
            mock_open.assert_called_once_with("TestPort", virtual=True)
            s.close()

    def test_beat_interval_calculation(self, sender):
        s, _ = sender
        # 120 BPM, subdivision=8 → 60/120/(8/4) = 0.25s
        assert s._beat_interval == pytest.approx(0.25)

    def test_port_open_failure_does_not_crash(self):
        with patch("vision_processor.midi.musical.mido.open_output", side_effect=Exception("no port")):
            from vision_processor.midi.musical import MusicalMidiSender
            s = MusicalMidiSender(port_name="Bad")
            assert s.port is None
            s.close()


# ===========================================================================
# Chord selection
# ===========================================================================

class TestChords:

    def test_chord_zone_I(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10, "kneeAngle": 1.0})
        notes = sorted([m.note for m in _msgs(port, "note_on")])
        assert notes == sorted([48, 52, 55])  # I

    def test_chord_zone_IV(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.30, "kneeAngle": 1.0})
        notes = sorted([m.note for m in _msgs(port, "note_on")])
        assert notes == sorted([53, 57, 60])  # IV

    def test_chord_zone_V(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.60, "kneeAngle": 1.0})
        notes = sorted([m.note for m in _msgs(port, "note_on")])
        assert notes == sorted([55, 59, 62])  # V

    def test_chord_zone_VI(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.80, "kneeAngle": 1.0})
        notes = sorted([m.note for m in _msgs(port, "note_on")])
        assert notes == sorted([57, 60, 64])  # VI

    def test_chord_change_turns_off_old_notes(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10, "kneeAngle": 1.0})  # chord I
        port.send.reset_mock()
        s.update({"feetCenterX": 0.80, "kneeAngle": 1.0})  # chord VI
        offs = _msgs(port, "note_off")
        assert len(offs) == 3  # three old chord notes turned off

    def test_same_chord_no_new_messages(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10, "kneeAngle": 1.0})
        port.send.reset_mock()
        s.update({"feetCenterX": 0.10, "kneeAngle": 1.0})
        note_ons = _msgs(port, "note_on")
        assert len(note_ons) == 0

    def test_knee_angle_affects_velocity(self, sender):
        s, port = sender
        # Low kneeAngle → low velocity
        s.update({"feetCenterX": 0.10, "kneeAngle": 0.0})
        msgs_bent = _msgs(port, "note_on")
        vel_bent = msgs_bent[0].velocity

        # Change chord to trigger new note_on with high kneeAngle
        port.send.reset_mock()
        s.update({"feetCenterX": 0.80, "kneeAngle": 1.0})
        msgs_straight = _msgs(port, "note_on")
        vel_straight = msgs_straight[0].velocity

        assert vel_straight > vel_bent


# ===========================================================================
# Chord expression
# ===========================================================================

class TestChordExpression:

    def test_hip_tilt_sends_pitch_bend(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10, "kneeAngle": 1.0})
        port.send.reset_mock()
        s.update({"feetCenterX": 0.10, "hipTilt": 0.5, "kneeAngle": 1.0})
        bends = _msgs(port, "pitchwheel")
        assert len(bends) == 3  # one per chord channel

    def test_knee_angle_sends_aftertouch(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10, "kneeAngle": 1.0})
        port.send.reset_mock()
        s.update({"feetCenterX": 0.10, "kneeAngle": 0.8})
        ats = _msgs(port, "aftertouch")
        assert len(ats) == 3


# ===========================================================================
# Melody direction
# ===========================================================================

class TestMelodyDirection:

    def test_first_frame_stores_prev_hand_y(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10, "rightHandY": 0.5})
        assert s._prev_hand_y == 0.5

    def test_hand_up_enqueues_candidate(self, sender):
        s, _ = sender
        s.update({"feetCenterX": 0.10, "rightHandY": 0.5})
        s.update({"feetCenterX": 0.10, "rightHandY": 0.6})  # dy=+0.1 > threshold
        assert s._note_candidate is not None

    def test_hand_down_enqueues_candidate(self, sender):
        s, _ = sender
        s.update({"feetCenterX": 0.10, "rightHandY": 0.5})
        s.update({"feetCenterX": 0.10, "rightHandY": 0.4})  # dy=-0.1
        assert s._note_candidate is not None

    def test_no_movement_no_candidate(self, sender):
        s, _ = sender
        s.update({"feetCenterX": 0.10, "rightHandY": 0.5})
        s._note_candidate = None
        s.update({"feetCenterX": 0.10, "rightHandY": 0.51})  # dy=0.01 < 0.03
        assert s._note_candidate is None

    def test_candidate_is_within_chord_tones(self, sender):
        s, _ = sender
        s.update({"feetCenterX": 0.10, "rightHandY": 0.5})  # chord I
        s.update({"feetCenterX": 0.10, "rightHandY": 0.7})  # up
        from vision_processor.midi.musical import MusicalMidiSender
        assert s._note_candidate in MusicalMidiSender.CHORD_TONES["I"]

    def test_fast_movement_larger_jump(self, sender):
        s, _ = sender
        # Slow movement
        s.update({"feetCenterX": 0.10, "rightHandY": 0.5, "rightArmVelocity": 0.1})
        s.update({"feetCenterX": 0.10, "rightHandY": 0.6, "rightArmVelocity": 0.1})
        idx_slow = s._melody_index

        # Reset
        s._melody_index = 2
        s._prev_hand_y = 0.5
        s._note_candidate = None

        # Fast movement (same direction)
        s.update({"feetCenterX": 0.10, "rightHandY": 0.6, "rightArmVelocity": 0.9})
        idx_fast = s._melody_index

        assert idx_fast > idx_slow  # fast jumps further

    def test_melody_index_resets_on_chord_change(self, sender):
        s, _ = sender
        s.update({"feetCenterX": 0.10, "rightHandY": 0.5})
        # Move melody index away from 2
        s._melody_index = 4
        # Chord change should reset to 2
        s.update({"feetCenterX": 0.80})  # chord I → VI
        assert s._melody_index == 2

    def test_melody_index_clamped_at_bounds(self, sender):
        s, _ = sender
        s.update({"feetCenterX": 0.10, "rightHandY": 0.5})
        # Push index down repeatedly
        for i in range(10):
            s._prev_hand_y = 0.5
            s.update({"feetCenterX": 0.10, "rightHandY": 0.3, "rightArmVelocity": 0.9})
        assert s._melody_index >= 0

        # Push index up repeatedly
        for i in range(10):
            s._prev_hand_y = 0.3
            s.update({"feetCenterX": 0.10, "rightHandY": 0.7, "rightArmVelocity": 0.9})
        assert s._melody_index <= 4

    def test_elbow_angle_sends_pitch_bend_on_melody(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10, "rightHandY": 0.5})
        # Set a melody note so pitch bend is applied
        s._melody_note = 60
        port.send.reset_mock()
        s.update({
            "feetCenterX": 0.10,
            "rightHandY": 0.7,
            "rightElbowHipAngle": 0.5,
        })
        bends = _msgs(port, "pitchwheel", channel=s.CH_MELODY)
        assert len(bends) >= 1


# ===========================================================================
# Beat firing
# ===========================================================================

class TestBeatFiring:

    def test_fire_beat_sends_queued_note(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10})  # establish chord
        port.send.reset_mock()

        # Manually queue a candidate and fire
        s._note_candidate = 64
        s._candidate_velocity = 100
        s._fire_beat()

        melody_ons = _msgs(port, "note_on", channel=s.CH_MELODY)
        assert len(melody_ons) == 1
        assert melody_ons[0].note == 64
        assert melody_ons[0].velocity == 100

    def test_fire_beat_turns_off_previous(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10})
        s._melody_note = 60
        port.send.reset_mock()

        s._note_candidate = 64
        s._fire_beat()

        offs = _msgs(port, "note_off", channel=s.CH_MELODY)
        assert len(offs) == 1
        assert offs[0].note == 60

    def test_fire_beat_no_candidate_no_action(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10})
        port.send.reset_mock()
        s._note_candidate = None
        s._fire_beat()
        melody_ons = _msgs(port, "note_on", channel=s.CH_MELODY)
        assert len(melody_ons) == 0

    def test_fire_beat_clears_candidate(self, sender):
        s, _ = sender
        s._note_candidate = 64
        s._fire_beat()
        assert s._note_candidate is None

    def test_fire_beat_no_port_does_nothing(self, sender):
        s, _ = sender
        s.port = None
        s._note_candidate = 64
        s._fire_beat()  # should not raise


# ===========================================================================
# Global expression
# ===========================================================================

class TestGlobalExpression:

    def test_head_tilt_center_sends_neutral_cc(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10})
        port.send.reset_mock()
        s.update({"feetCenterX": 0.10, "headTilt": 0.0})
        ccs = _msgs(port, "control_change", channel=s.CH_MASTER)
        cc74 = [m for m in ccs if m.control == 74]
        assert len(cc74) >= 1
        assert cc74[0].value == 64

    def test_head_tilt_right_bright(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10})
        port.send.reset_mock()
        s.update({"feetCenterX": 0.10, "headTilt": 0.5})
        ccs = _msgs(port, "control_change", channel=s.CH_MASTER)
        cc74 = [m for m in ccs if m.control == 74]
        assert cc74[0].value > 64

    def test_head_tilt_left_dark(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10})
        port.send.reset_mock()
        s.update({"feetCenterX": 0.10, "headTilt": -0.5})
        ccs = _msgs(port, "control_change", channel=s.CH_MASTER)
        cc74 = [m for m in ccs if m.control == 74]
        assert cc74[0].value < 64


# ===========================================================================
# Close
# ===========================================================================

class TestClose:

    def test_close_sends_all_notes_off(self, sender):
        s, port = sender
        s.update({"feetCenterX": 0.10})
        port.send.reset_mock()
        s.close()
        ccs = _msgs(port, "control_change")
        all_notes_off = [m for m in ccs if m.control == 123]
        assert len(all_notes_off) == 4  # 4 channels

    def test_close_stops_running(self, sender):
        s, _ = sender
        s.close()
        assert s._running is False

    def test_update_after_close_does_nothing(self, sender):
        s, port = sender
        s.close()
        # After close, port is closed — update checks self.port
        # The mock port.close() doesn't actually nullify it,
        # so we set it to None to simulate a closed port
        s.port = None
        port.send.reset_mock()
        s.update({"feetCenterX": 0.10})
        assert port.send.call_count == 0


# ===========================================================================
# Update with no port
# ===========================================================================

class TestNoPort:

    def test_update_without_port_does_nothing(self):
        with patch("vision_processor.midi.musical.mido.open_output", side_effect=Exception("no port")):
            from vision_processor.midi.musical import MusicalMidiSender
            s = MusicalMidiSender(port_name="Bad")
            s.update({"feetCenterX": 0.5})  # should not raise
            s.close()
