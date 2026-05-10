"""
Unit tests for vision_processor/midi/rhythmical.py — percussive pentatonic mode
that maps all 17 kinematic features to audible output.

Tests focus on the per-call logic in isolation (energy gate, octave shifts,
density patterns, master CCs, jerk edge-detection, tick firing). The tempo
thread and threading.Timer are mocked so assertions don't depend on timing.

Usage:
    pytest tests/unit/test_rhythmical.py -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.fixture
def sender():
    """Create a RhythmicalMidiSender with mocked port, tempo thread and timer."""
    with patch("vision_processor.midi.rhythmical.mido.open_output") as mock_open, \
         patch("vision_processor.midi.rhythmical.threading.Thread") as mock_thread, \
         patch("vision_processor.midi.rhythmical.threading.Timer") as mock_timer:
        mock_port = MagicMock()
        mock_open.return_value = mock_port
        mock_thread.return_value = MagicMock()
        mock_timer.return_value = MagicMock()

        from vision_processor.midi.rhythmical import RhythmicalMidiSender
        s = RhythmicalMidiSender(port_name="Test", tempo_bpm=120)
        try:
            yield s, mock_port, mock_timer
        finally:
            s._running = False


# Default features that put the sender above the energy gate so tick firing
# isn't muted. Individual tests override specific fields as needed.
ACTIVE = {
    "energy": 0.4,
    "symmetry": 0.0,
    "smoothness": 0.5,
    "armAngle": 0.5,
    "verticalExtension": 0.5,
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


def _features(**overrides) -> dict:
    f = dict(ACTIVE)
    f.update(overrides)
    return f


def _msgs(mock_port, msg_type=None, channel=None):
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
        with patch("vision_processor.midi.rhythmical.mido.open_output") as mock_open, \
             patch("vision_processor.midi.rhythmical.threading.Thread"), \
             patch("vision_processor.midi.rhythmical.threading.Timer"):
            mock_open.return_value = MagicMock()
            from vision_processor.midi.rhythmical import RhythmicalMidiSender
            s = RhythmicalMidiSender(port_name="X")
            mock_open.assert_called_once_with("X", virtual=True)
            s.close()

    def test_tick_period_at_120_bpm(self, sender):
        s, _, _ = sender
        assert s._tick_period == pytest.approx(0.125)

    def test_port_open_failure_does_not_crash(self):
        with patch("vision_processor.midi.rhythmical.mido.open_output", side_effect=Exception("no port")), \
             patch("vision_processor.midi.rhythmical.threading.Thread"), \
             patch("vision_processor.midi.rhythmical.threading.Timer"):
            from vision_processor.midi.rhythmical import RhythmicalMidiSender
            s = RhythmicalMidiSender(port_name="Bad")
            assert s.port is None
            s.close()


# ===========================================================================
# Energy gate (the "stillness = silence" requirement)
# ===========================================================================

class TestEnergyGate:

    def test_below_gate_zero_gain(self, sender):
        s, _, _ = sender
        s.update(_features(energy=0.0))
        assert s._energy_gain == 0.0

    def test_at_gate_threshold_gain_is_floor(self, sender):
        s, _, _ = sender
        s.update(_features(energy=s.ENERGY_GATE))
        assert s._energy_gain == pytest.approx(s.GATE_VELOCITY_FACTOR)

    def test_above_full_threshold_gain_is_max(self, sender):
        s, _, _ = sender
        s.update(_features(energy=1.0))
        assert s._energy_gain == pytest.approx(s.FULL_VELOCITY_FACTOR)

    def test_silenced_state_emits_no_notes_on_tick(self, sender):
        s, port, _ = sender
        s.update(_features(energy=0.0, feetCenterX=0.5, rightArmVelocity=0.5))
        port.send.reset_mock()
        s._fire_tick(0)
        assert _msgs(port, "note_on") == []

    def test_active_state_emits_bass_on_pattern_tick(self, sender):
        s, port, _ = sender
        s.update(_features(energy=0.5, feetCenterX=0.5))
        port.send.reset_mock()
        s._fire_tick(0)
        bass_ons = _msgs(port, "note_on", channel=s.CH_BASS)
        assert len(bass_ons) == 1


# ===========================================================================
# Octave shift (verticalExtension + kneeAngle)
# ===========================================================================

class TestOctaveShift:

    def test_vertical_extension_low_shifts_down(self, sender):
        s, _, _ = sender
        s.update(_features(verticalExtension=0.1))
        assert s._octave_shift == -12

    def test_vertical_extension_neutral_no_shift(self, sender):
        s, _, _ = sender
        s.update(_features(verticalExtension=0.5))
        assert s._octave_shift == 0

    def test_vertical_extension_high_shifts_up(self, sender):
        s, _, _ = sender
        s.update(_features(verticalExtension=0.9))
        assert s._octave_shift == 12

    def test_bent_knees_shift_bass_down(self, sender):
        s, _, _ = sender
        s.update(_features(kneeAngle=0.2))
        assert s._bass_octave_off == -12

    def test_straight_knees_no_bass_octave_shift(self, sender):
        s, _, _ = sender
        s.update(_features(kneeAngle=1.0))
        assert s._bass_octave_off == 0

    def test_octave_shifts_compound_on_bass(self, sender):
        """High verticalExt + bent knees = +12 - 12 = 0 net shift on bass."""
        s, port, _ = sender
        s.update(_features(verticalExtension=0.9, kneeAngle=0.0,
                           feetCenterX=0.5, energy=0.5))
        port.send.reset_mock()
        s._fire_tick(0)
        bass_ons = _msgs(port, "note_on", channel=s.CH_BASS)
        assert bass_ons[0].note == s.BASS_NOTE  # +12 - 12 = 0

    def test_octave_shifts_apply_to_melody(self, sender):
        s, port, _ = sender
        s.update(_features(verticalExtension=0.9, rightHandY=0.0,
                           rightArmVelocity=0.5, energy=0.5, feetCenterX=-1))
        port.send.reset_mock()
        # Override bass pattern to silence so we only see melody
        s._bass_pattern = frozenset()
        s._fire_tick(0)
        mel_ons = _msgs(port, "note_on", channel=s.CH_MELODY_R)
        assert mel_ons[0].note == s.HIRAJOSHI_A[0] + 12


# ===========================================================================
# Bass density (driven by feet)
# ===========================================================================

class TestBassDensity:

    def test_left_zone_uses_low_pattern(self, sender):
        s, _, _ = sender
        s.update(_features(feetCenterX=0.10))
        assert s._bass_pattern is s.BASS_PATTERN_LOW

    def test_center_zone_uses_medium_pattern(self, sender):
        s, _, _ = sender
        s.update(_features(feetCenterX=0.50))
        assert s._bass_pattern is s.BASS_PATTERN_MEDIUM

    def test_right_zone_uses_high_pattern(self, sender):
        s, _, _ = sender
        s.update(_features(feetCenterX=0.80))
        assert s._bass_pattern is s.BASS_PATTERN_HIGH

    def test_high_pattern_is_3_3_2_syncopation(self):
        from vision_processor.midi.rhythmical import RhythmicalMidiSender
        assert RhythmicalMidiSender.BASS_PATTERN_HIGH == frozenset({0, 3, 6, 8, 11, 14})


# ===========================================================================
# Per-hand melody density (driven by armVelocity)
# ===========================================================================

class TestMelodyDensity:

    def test_below_floor_silent(self, sender):
        s, _, _ = sender
        s.update(_features(rightArmVelocity=0.01))
        assert s._right_pattern is s.MELODY_PATTERN_SILENT

    def test_low_velocity_sparse(self, sender):
        s, _, _ = sender
        s.update(_features(rightArmVelocity=0.10))
        assert s._right_pattern is s.MELODY_PATTERN_SPARSE

    def test_medium_velocity_quarter(self, sender):
        s, _, _ = sender
        s.update(_features(rightArmVelocity=0.50))
        assert s._right_pattern is s.MELODY_PATTERN_QUARTER

    def test_high_velocity_eighth(self, sender):
        s, _, _ = sender
        s.update(_features(rightArmVelocity=0.90))
        assert s._right_pattern is s.MELODY_PATTERN_EIGHTH

    def test_left_hand_independent_of_right(self, sender):
        s, _, _ = sender
        s.update(_features(rightArmVelocity=0.0, leftArmVelocity=0.9))
        assert s._right_pattern is s.MELODY_PATTERN_SILENT
        assert s._left_pattern  is s.MELODY_PATTERN_EIGHTH

    def test_arm_velocity_modulates_velocity(self, sender):
        s, _, _ = sender
        s.update(_features(rightArmVelocity=0.20))
        slow_v = s._right_velocity
        s.update(_features(rightArmVelocity=0.80))
        fast_v = s._right_velocity
        assert fast_v > slow_v


# ===========================================================================
# Per-hand melody candidate (driven by handY)
# ===========================================================================

class TestMelodyCandidate:

    def test_right_hand_y_zero_picks_lowest_note(self, sender):
        s, _, _ = sender
        s.update(_features(rightHandY=0.0, rightArmVelocity=0.5))
        assert s._right_candidate == s.HIRAJOSHI_A[0]

    def test_right_hand_y_one_picks_highest_note(self, sender):
        s, _, _ = sender
        s.update(_features(rightHandY=1.0, rightArmVelocity=0.5))
        assert s._right_candidate == s.HIRAJOSHI_A[-1]

    def test_left_hand_y_independent_of_right(self, sender):
        s, _, _ = sender
        s.update(_features(rightHandY=0.0, leftHandY=1.0,
                           rightArmVelocity=0.5, leftArmVelocity=0.5))
        assert s._right_candidate == s.HIRAJOSHI_A[0]
        assert s._left_candidate  == s.HIRAJOSHI_A[-1]

    def test_still_hand_does_not_queue_candidate(self, sender):
        s, _, _ = sender
        s.update(_features(rightArmVelocity=0.0))
        assert s._right_candidate is None


# ===========================================================================
# Master CCs (all 4 control changes sent every frame)
# ===========================================================================

class TestMasterCCs:

    @pytest.mark.parametrize("cc,key,value,expected", [
        (1,  "smoothness",  0.5, 63),    # mod wheel  ~ smoothness * 127
        (1,  "smoothness",  1.0, 127),
        (10, "symmetry",   -1.0, 1),     # pan        center + symmetry * 63
        (10, "symmetry",    0.0, 64),
        (10, "symmetry",   +1.0, 127),
        (11, "armAngle",    0.0, 0),     # expression ~ armAngle * 127
        (11, "armAngle",    1.0, 127),
        (74, "headTilt",   -1.0, 1),     # filter cutoff
        (74, "headTilt",    0.0, 64),
        (74, "headTilt",   +1.0, 127),
    ])
    def test_master_cc_value_for_input(self, sender, cc, key, value, expected):
        s, port, _ = sender
        port.send.reset_mock()
        s.update(_features(**{key: value}))
        ccs = [m for m in _msgs(port, "control_change", channel=s.CH_MASTER) if m.control == cc]
        assert ccs, f"CC{cc} not sent"
        assert ccs[-1].value == expected


# ===========================================================================
# Pitch bends (per hand + bass)
# ===========================================================================

class TestPitchBends:

    def test_right_elbow_sends_bend_on_right_channel(self, sender):
        s, port, _ = sender
        port.send.reset_mock()
        s.update(_features(rightElbowHipAngle=0.5))
        bends = _msgs(port, "pitchwheel", channel=s.CH_MELODY_R)
        assert bends and bends[-1].pitch != 0

    def test_left_elbow_sends_bend_on_left_channel(self, sender):
        s, port, _ = sender
        port.send.reset_mock()
        s.update(_features(leftElbowHipAngle=0.5))
        bends = _msgs(port, "pitchwheel", channel=s.CH_MELODY_L)
        assert bends and bends[-1].pitch != 0

    def test_hip_tilt_sends_bend_on_bass_channel(self, sender):
        s, port, _ = sender
        port.send.reset_mock()
        s.update(_features(hipTilt=0.5))
        bends = _msgs(port, "pitchwheel", channel=s.CH_BASS)
        assert bends and bends[-1].pitch != 0


# ===========================================================================
# Jerk-driven off-grid accent (rising edge only)
# ===========================================================================

class TestJerkAccent:

    def test_rising_edge_fires_accent(self, sender):
        s, port, _ = sender
        port.send.reset_mock()
        s.update(_features(rightHandJerk=0.5))  # crosses threshold
        ons = _msgs(port, "note_on", channel=s.CH_MELODY_R)
        assert len(ons) == 1

    def test_no_fire_while_held_above_threshold(self, sender):
        s, port, _ = sender
        s.update(_features(rightHandJerk=0.5))  # rising edge
        port.send.reset_mock()
        s.update(_features(rightHandJerk=0.6))  # still above; no new fire
        ons = _msgs(port, "note_on", channel=s.CH_MELODY_R)
        assert ons == []

    def test_re_fire_after_dropping_below(self, sender):
        s, port, _ = sender
        s.update(_features(rightHandJerk=0.5))   # rising edge
        s.update(_features(rightHandJerk=0.1))   # below threshold
        port.send.reset_mock()
        s.update(_features(rightHandJerk=0.5))   # rising edge again
        ons = _msgs(port, "note_on", channel=s.CH_MELODY_R)
        assert len(ons) == 1

    def test_silenced_state_does_not_fire_accent(self, sender):
        s, port, _ = sender
        port.send.reset_mock()
        s.update(_features(energy=0.0, rightHandJerk=0.9))
        ons = _msgs(port, "note_on", channel=s.CH_MELODY_R)
        assert ons == []

    def test_left_jerk_independent_of_right(self, sender):
        s, port, _ = sender
        port.send.reset_mock()
        s.update(_features(leftHandJerk=0.5))
        right_ons = _msgs(port, "note_on", channel=s.CH_MELODY_R)
        left_ons  = _msgs(port, "note_on", channel=s.CH_MELODY_L)
        assert right_ons == []
        assert len(left_ons) == 1


# ===========================================================================
# Tick firing
# ===========================================================================

class TestFireTick:

    def test_bass_fires_on_pattern_position(self, sender):
        s, port, _ = sender
        s.update(_features(feetCenterX=0.5, energy=0.5))
        port.send.reset_mock()
        s._fire_tick(0)
        assert _msgs(port, "note_on", channel=s.CH_BASS)

    def test_bass_silent_off_pattern(self, sender):
        s, port, _ = sender
        s.update(_features(feetCenterX=0.5, energy=0.5))
        port.send.reset_mock()
        # Tick 1 not in MEDIUM pattern (which is 0,2,4,6,8,10,12,14)
        s._fire_tick(1)
        assert _msgs(port, "note_on", channel=s.CH_BASS) == []

    def test_right_melody_fires_when_in_pattern(self, sender):
        s, port, _ = sender
        s.update(_features(rightHandY=0.5, rightArmVelocity=0.9, energy=0.5))
        # Override bass pattern to silence to isolate melody
        s._bass_pattern = frozenset()
        port.send.reset_mock()
        s._fire_tick(0)  # 0 is in EIGHTH pattern
        assert _msgs(port, "note_on", channel=s.CH_MELODY_R)

    def test_left_melody_fires_independently(self, sender):
        s, port, _ = sender
        s.update(_features(leftHandY=0.5, leftArmVelocity=0.9,
                           rightArmVelocity=0.0, energy=0.5))
        s._bass_pattern = frozenset()
        port.send.reset_mock()
        s._fire_tick(0)
        assert _msgs(port, "note_on", channel=s.CH_MELODY_L)
        assert _msgs(port, "note_on", channel=s.CH_MELODY_R) == []

    def test_velocity_scaled_by_energy_gain(self, sender):
        s, port, _ = sender
        # Low energy → small velocity
        s.update(_features(energy=s.ENERGY_GATE, feetCenterX=0.5))
        port.send.reset_mock()
        s._fire_tick(0)
        low_vel = _msgs(port, "note_on", channel=s.CH_BASS)[0].velocity

        # High energy → larger velocity
        s.update(_features(energy=1.0, feetCenterX=0.5))
        port.send.reset_mock()
        s._fire_tick(0)
        high_vel = _msgs(port, "note_on", channel=s.CH_BASS)[0].velocity

        assert high_vel > low_vel

    def test_candidate_consumed_after_firing(self, sender):
        s, _, _ = sender
        s.update(_features(rightHandY=0.5, rightArmVelocity=0.9, energy=0.5))
        s._bass_pattern = frozenset()
        s._fire_tick(0)
        assert s._right_candidate is None
        assert s._left_candidate  is None

    def test_no_port_fires_nothing(self, sender):
        s, port, _ = sender
        s.port = None
        port.send.reset_mock()
        s._fire_tick(0)
        assert port.send.call_count == 0


# ===========================================================================
# Accent table
# ===========================================================================

class TestAccent:

    @pytest.mark.parametrize("pos,expected", [
        (0,  25), (8,  10), (4,  0), (12, 0),
        (2,  -5), (6,  -5), (10, -5), (14, -5),
        (1, -15), (3, -15), (5, -15), (7, -15),
        (9, -15), (11, -15), (13, -15), (15, -15),
    ])
    def test_accent_table_exhaustive(self, pos, expected):
        from vision_processor.midi.rhythmical import RhythmicalMidiSender
        assert RhythmicalMidiSender._accent_for_position(pos) == expected


# ===========================================================================
# Close
# ===========================================================================

class TestClose:

    def test_close_sends_all_notes_off_on_used_channels(self, sender):
        s, port, _ = sender
        port.send.reset_mock()
        s.close()
        ccs = _msgs(port, "control_change")
        all_off = [m for m in ccs if m.control == 123]
        assert {m.channel for m in all_off} == {s.CH_BASS, s.CH_MELODY_R, s.CH_MELODY_L}

    def test_close_stops_running(self, sender):
        s, _, _ = sender
        s.close()
        assert s._running is False


# ===========================================================================
# Update with no port
# ===========================================================================

class TestNoPort:

    def test_update_without_port_does_nothing(self):
        with patch("vision_processor.midi.rhythmical.mido.open_output", side_effect=Exception("no port")), \
             patch("vision_processor.midi.rhythmical.threading.Thread"), \
             patch("vision_processor.midi.rhythmical.threading.Timer"):
            from vision_processor.midi.rhythmical import RhythmicalMidiSender
            s = RhythmicalMidiSender(port_name="Bad")
            s.update(_features(energy=0.5, rightArmVelocity=0.5))
            s.close()
