"""
Unit tests for vision_processor/midi/musical.py — percussive pentatonic mode.

Tests focus on the per-call logic (_update_*, _fire_tick, _accent_for_position)
in isolation rather than driving the live tempo thread, so assertions don't
depend on timing. The tempo thread itself is replaced by a no-op MagicMock
during fixture setup; threading.Timer is also mocked so scheduled note_offs
can be observed without real sleeps.

Usage:
    pytest tests/unit/test_musical.py -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.fixture
def sender():
    """Create a MusicalMidiSender with mocked port, tempo thread and timer."""
    with patch("vision_processor.midi.musical.mido.open_output") as mock_open, \
         patch("vision_processor.midi.musical.threading.Thread") as mock_thread, \
         patch("vision_processor.midi.musical.threading.Timer") as mock_timer:
        mock_port = MagicMock()
        mock_open.return_value = mock_port

        thread_instance = MagicMock()
        mock_thread.return_value = thread_instance

        timer_instance = MagicMock()
        mock_timer.return_value = timer_instance

        from vision_processor.midi.musical import MusicalMidiSender
        s = MusicalMidiSender(port_name="Test", tempo_bpm=120)
        try:
            yield s, mock_port, mock_timer
        finally:
            s._running = False


def _msgs(mock_port, msg_type: str | None = None, channel: int | None = None):
    """Extract sent MIDI messages, optionally filtered by type/channel."""
    msgs = [c[0][0] for c in mock_port.send.call_args_list]
    if msg_type:
        msgs = [m for m in msgs if m.type == msg_type]
    if channel is not None:
        msgs = [m for m in msgs if m.channel == channel]
    return msgs


# ===========================================================================
# Init
# ===========================================================================

class TestInit:

    def test_opens_virtual_port(self):
        with patch("vision_processor.midi.musical.mido.open_output") as mock_open, \
             patch("vision_processor.midi.musical.threading.Thread"), \
             patch("vision_processor.midi.musical.threading.Timer"):
            mock_open.return_value = MagicMock()
            from vision_processor.midi.musical import MusicalMidiSender
            s = MusicalMidiSender(port_name="X")
            mock_open.assert_called_once_with("X", virtual=True)
            s.close()

    def test_tick_period_at_120_bpm(self, sender):
        s, _, _ = sender
        # 1/16 at 120 BPM = 60 / 120 / 4 = 0.125s
        assert s._tick_period == pytest.approx(0.125)

    def test_port_open_failure_does_not_crash(self):
        with patch("vision_processor.midi.musical.mido.open_output", side_effect=Exception("no port")), \
             patch("vision_processor.midi.musical.threading.Thread"), \
             patch("vision_processor.midi.musical.threading.Timer"):
            from vision_processor.midi.musical import MusicalMidiSender
            s = MusicalMidiSender(port_name="Bad")
            assert s.port is None
            s.close()


# ===========================================================================
# Bass density (driven by feet)
# ===========================================================================

class TestBassDensity:

    def test_left_zone_uses_low_pattern(self, sender):
        s, _, _ = sender
        s.update({"feetCenterX": 0.10, "rightArmVelocity": 0.0})
        assert s._bass_pattern is s.BASS_PATTERN_LOW

    def test_center_zone_uses_medium_pattern(self, sender):
        s, _, _ = sender
        s.update({"feetCenterX": 0.50, "rightArmVelocity": 0.0})
        assert s._bass_pattern is s.BASS_PATTERN_MEDIUM

    def test_right_zone_uses_high_pattern(self, sender):
        s, _, _ = sender
        s.update({"feetCenterX": 0.80, "rightArmVelocity": 0.0})
        assert s._bass_pattern is s.BASS_PATTERN_HIGH

    def test_high_pattern_is_3_3_2_syncopation(self):
        from vision_processor.midi.musical import MusicalMidiSender
        # 3+3+2 over 8 sixteenths repeated → ticks 0,3,6,8,11,14
        assert MusicalMidiSender.BASS_PATTERN_HIGH == frozenset({0, 3, 6, 8, 11, 14})


# ===========================================================================
# Melody candidate (driven by hand)
# ===========================================================================

class TestMelodyCandidate:

    def test_still_hand_does_not_queue_candidate(self, sender):
        s, _, _ = sender
        s.update({"feetCenterX": 0.5, "rightHandY": 0.5, "rightArmVelocity": 0.0})
        assert s._melody_candidate is None

    def test_moving_hand_queues_pentatonic_note(self, sender):
        s, _, _ = sender
        s.update({"feetCenterX": 0.5, "rightHandY": 0.5, "rightArmVelocity": 0.5})
        assert s._melody_candidate in s.HIRAJOSHI_A

    def test_hand_y_zero_picks_lowest_note(self, sender):
        s, _, _ = sender
        s.update({"feetCenterX": 0.5, "rightHandY": 0.0, "rightArmVelocity": 0.5})
        assert s._melody_candidate == s.HIRAJOSHI_A[0]

    def test_hand_y_one_picks_highest_note(self, sender):
        s, _, _ = sender
        s.update({"feetCenterX": 0.5, "rightHandY": 1.0, "rightArmVelocity": 0.5})
        assert s._melody_candidate == s.HIRAJOSHI_A[-1]

    def test_arm_velocity_modulates_candidate_velocity(self, sender):
        s, _, _ = sender
        s.update({"feetCenterX": 0.5, "rightHandY": 0.5, "rightArmVelocity": 0.2})
        slow_vel = s._melody_candidate_velocity
        s.update({"feetCenterX": 0.5, "rightHandY": 0.5, "rightArmVelocity": 0.9})
        fast_vel = s._melody_candidate_velocity
        assert fast_vel > slow_vel


# ===========================================================================
# Tick firing
# ===========================================================================

class TestFireTick:

    def test_bass_fires_on_pattern_position(self, sender):
        s, port, _ = sender
        s._bass_pattern = frozenset({0, 8})
        s._fire_tick(0)
        bass_ons = _msgs(port, "note_on", channel=s.CH_BASS)
        assert len(bass_ons) == 1
        assert bass_ons[0].note == s.BASS_NOTE

    def test_bass_silent_off_pattern(self, sender):
        s, port, _ = sender
        s._bass_pattern = frozenset({0, 8})
        s._fire_tick(1)  # not in pattern
        assert _msgs(port, "note_on", channel=s.CH_BASS) == []

    def test_melody_does_not_fire_on_sixteenth_offbeats(self, sender):
        s, port, _ = sender
        s._bass_pattern = frozenset()  # silence bass
        s._melody_candidate = 60
        s._melody_candidate_velocity = 80
        s._fire_tick(1)  # 1/16 off-beat
        assert _msgs(port, "note_on", channel=s.CH_MELODY) == []

    def test_melody_fires_on_eighth_note(self, sender):
        s, port, _ = sender
        s._bass_pattern = frozenset()
        s._melody_candidate = 64
        s._melody_candidate_velocity = 80
        s._fire_tick(2)  # 1/8 position
        melody_ons = _msgs(port, "note_on", channel=s.CH_MELODY)
        assert len(melody_ons) == 1
        assert melody_ons[0].note == 64

    def test_melody_candidate_consumed_after_firing(self, sender):
        s, _, _ = sender
        s._bass_pattern = frozenset()
        s._melody_candidate = 60
        s._fire_tick(2)
        assert s._melody_candidate is None

    def test_no_melody_fires_when_candidate_is_none(self, sender):
        s, port, _ = sender
        s._bass_pattern = frozenset()
        s._melody_candidate = None
        s._fire_tick(2)
        assert _msgs(port, "note_on", channel=s.CH_MELODY) == []

    def test_fire_short_note_schedules_note_off_via_timer(self, sender):
        s, _, mock_timer = sender
        s._bass_pattern = frozenset({0})
        s._fire_tick(0)
        # Timer should have been instantiated to schedule the note_off
        assert mock_timer.called
        # Confirm it was for NOTE_DURATION_S seconds
        args, kwargs = mock_timer.call_args
        assert args[0] == s.NOTE_DURATION_S

    def test_no_port_fires_nothing(self, sender):
        s, port, _ = sender
        s.port = None
        s._bass_pattern = frozenset({0})
        s._melody_candidate = 60
        s._fire_tick(0)
        assert port.send.call_count == 0


# ===========================================================================
# Accent table
# ===========================================================================

class TestAccent:

    @pytest.mark.parametrize("pos,expected", [
        (0,  25),   # bar downbeat
        (8,  10),   # half-bar (beat 3 of 4/4)
        (4,  0),    # other quarter beat
        (12, 0),
        (2,  -5),   # 1/8 off-beat
        (6,  -5),
        (10, -5),
        (14, -5),
        (1,  -15),  # 1/16 ghost
        (3,  -15),
        (5,  -15),
        (7,  -15),
        (9,  -15),
        (11, -15),
        (13, -15),
        (15, -15),
    ])
    def test_accent_table_exhaustive(self, pos, expected):
        from vision_processor.midi.musical import MusicalMidiSender
        assert MusicalMidiSender._accent_for_position(pos) == expected


# ===========================================================================
# Global expression (head tilt → CC74)
# ===========================================================================

class TestGlobalExpression:

    def test_head_tilt_center_sends_neutral_cc(self, sender):
        s, port = sender[:2]
        s.update({"feetCenterX": 0.5, "headTilt": 0.0, "rightArmVelocity": 0.0})
        ccs = _msgs(port, "control_change", channel=s.CH_MASTER)
        cc74 = [m for m in ccs if m.control == 74]
        assert cc74 and cc74[-1].value == 64

    def test_head_tilt_right_brightens(self, sender):
        s, port = sender[:2]
        s.update({"feetCenterX": 0.5, "headTilt": 0.5, "rightArmVelocity": 0.0})
        cc74 = [m for m in _msgs(port, "control_change", channel=s.CH_MASTER) if m.control == 74]
        assert cc74[-1].value > 64

    def test_head_tilt_left_darkens(self, sender):
        s, port = sender[:2]
        s.update({"feetCenterX": 0.5, "headTilt": -0.5, "rightArmVelocity": 0.0})
        cc74 = [m for m in _msgs(port, "control_change", channel=s.CH_MASTER) if m.control == 74]
        assert cc74[-1].value < 64


# ===========================================================================
# Close
# ===========================================================================

class TestClose:

    def test_close_sends_all_notes_off_on_used_channels(self, sender):
        s, port, _ = sender
        port.send.reset_mock()
        s.close()
        ccs = _msgs(port, "control_change")
        all_notes_off = [m for m in ccs if m.control == 123]
        # CC 123 sent on bass + melody channels
        assert {m.channel for m in all_notes_off} == {s.CH_BASS, s.CH_MELODY}

    def test_close_stops_running(self, sender):
        s, _, _ = sender
        s.close()
        assert s._running is False


# ===========================================================================
# Update with no port
# ===========================================================================

class TestNoPort:

    def test_update_without_port_does_nothing(self):
        with patch("vision_processor.midi.musical.mido.open_output", side_effect=Exception("no port")), \
             patch("vision_processor.midi.musical.threading.Thread"), \
             patch("vision_processor.midi.musical.threading.Timer"):
            from vision_processor.midi.musical import MusicalMidiSender
            s = MusicalMidiSender(port_name="Bad")
            s.update({"feetCenterX": 0.5, "rightArmVelocity": 0.5})
            s.close()
