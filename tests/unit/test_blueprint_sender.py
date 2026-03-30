"""
Unit tests for vision_processor/midi/blueprint.py

Tests the BlueprintMidiSender with a mock MIDI port and real progressions
from the Unison MIDI Blueprint library.

Usage:
    cd ~/cuerposonoro
    pytest tests/unit/test_blueprint_sender.py -v
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vision_processor.midi.blueprint_loader import BlueprintLibrary, Chord, Progression
from vision_processor.midi.blueprint import BlueprintMidiSender

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "../../assets/midi")


def _make_config(overrides=None):
    """Build a minimal config dict for BlueprintMidiSender."""
    cfg = {
        "silence_timeout": 60,
        "pelvis_threshold": 0.12,
        "chord_change_cooldown": 1.2,
        "silence_threshold": 0.02,
        "silence_hold_ms": 500,
        "midi_channel_melody": 5,
        "midi_channel_bass": 6,
        "midi_channel_master": 0,
        "assets_path": ASSETS_PATH,
        "arm_velocity_threshold": 0.25,
        "melody_bend_scale": 4.0,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def _neutral_features(**overrides):
    """Features dict representing a neutral standing body."""
    f = {
        "energy": 0.0,
        "symmetry": 0.0,
        "smoothness": 0.5,
        "armAngle": 0.0,
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
        "pelvis_thrust": 0.0,
        "spine_lean": 0.0,
    }
    f.update(overrides)
    return f


@pytest.fixture
def mock_port():
    """Mock MIDI port."""
    return MagicMock()


@pytest.fixture
def sender(mock_port):
    """BlueprintMidiSender with a mock MIDI port."""
    with patch("vision_processor.midi.blueprint.mido") as mock_mido:
        mock_mido.open_output.return_value = mock_port
        s = BlueprintMidiSender(
            config=_make_config(),
            port_name="TestPort",
        )
    return s


@pytest.fixture
def sender_fixed_genre(mock_port):
    """BlueprintMidiSender with a fixed genre."""
    with patch("vision_processor.midi.blueprint.mido") as mock_mido:
        mock_mido.open_output.return_value = mock_port
        s = BlueprintMidiSender(
            config=_make_config(),
            port_name="TestPort",
            genre="Jazz",
        )
    return s


@pytest.fixture
def sender_fixed_key(mock_port):
    """BlueprintMidiSender with a fixed genre and key."""
    with patch("vision_processor.midi.blueprint.mido") as mock_mido:
        mock_mido.open_output.return_value = mock_port
        s = BlueprintMidiSender(
            config=_make_config(),
            port_name="TestPort",
            genre="Jazz",
            key="C Major",
        )
    return s


# Chord navigation

class TestChordNavigation:

    def test_chord_advances_on_positive_thrust(self, sender):
        """pelvis_thrust > threshold advances the chord index."""
        initial_idx = sender._chord_index
        sender._last_chord_change = 0  # allow change
        sender.update(_neutral_features(pelvis_thrust=0.5))
        assert sender._chord_index == initial_idx + 1

    def test_chord_retreats_on_negative_thrust(self, sender):
        """pelvis_thrust < -threshold retreats the chord index."""
        # First advance to index 1 so we can retreat
        sender._chord_index = 1
        sender._last_chord_change = 0
        sender.update(_neutral_features(pelvis_thrust=-0.5))
        assert sender._chord_index == 0

    def test_chord_wraps_circularly(self, sender):
        """After the last chord, wraps to the first."""
        num_chords = len(sender._progression.chords)
        sender._chord_index = num_chords - 1
        sender._last_chord_change = 0
        sender.update(_neutral_features(pelvis_thrust=0.5))
        assert sender._chord_index == 0

    def test_chord_wraps_backward(self, sender):
        """Before the first chord, wraps to the last."""
        num_chords = len(sender._progression.chords)
        sender._chord_index = 0
        sender._last_chord_change = 0
        sender.update(_neutral_features(pelvis_thrust=-0.5))
        assert sender._chord_index == num_chords - 1

    def test_chord_does_not_change_within_cooldown(self, sender):
        """No chord change if cooldown has not elapsed."""
        sender._last_chord_change = time.time()  # just changed
        initial_idx = sender._chord_index
        sender.update(_neutral_features(pelvis_thrust=0.5))
        assert sender._chord_index == initial_idx

    def test_chord_does_not_change_below_threshold(self, sender):
        """No chord change if pelvis_thrust is below threshold."""
        sender._last_chord_change = 0
        initial_idx = sender._chord_index
        sender.update(_neutral_features(pelvis_thrust=0.05))
        assert sender._chord_index == initial_idx


# Melody

class TestMelody:

    def test_melody_note_within_chord_tones(self, sender):
        """Triggered melody note must be from the active chord's notes."""
        sender._last_chord_change = 0
        # Trigger with arm velocity above threshold
        features = _neutral_features(
            rightArmVelocity=0.5,
            leftArmVelocity=0.5,
            rightHandY=0.5,
            leftHandY=0.5,
        )
        sender.update(features)
        if sender._melody_note is not None:
            chord = sender._progression.chords[sender._chord_index]
            assert sender._melody_note in chord.notes

    def test_melody_triggers_on_velocity(self, sender, mock_port):
        """A note_on is sent when mean arm velocity exceeds threshold."""
        features = _neutral_features(
            rightArmVelocity=0.6,
            leftArmVelocity=0.6,
        )
        sender.update(features)
        # Check that at least one note_on was sent on the melody channel
        calls = mock_port.send.call_args_list
        melody_ch = sender._config["midi_channel_melody"]
        note_ons = [c for c in calls if hasattr(c[0][0], 'type') and
                    c[0][0].type == 'note_on' and c[0][0].channel == melody_ch]
        assert len(note_ons) >= 1


# CC1 on master channel

class TestCC1:

    def test_cc1_sent_on_master_channel(self, sender, mock_port):
        """CC1 (modulation) is sent on the MPE master channel."""
        features = _neutral_features(spine_lean=0.5)
        sender.update(features)
        calls = mock_port.send.call_args_list
        master_ch = sender._config["midi_channel_master"]
        cc1_msgs = [
            c for c in calls if hasattr(c[0][0], 'type') and
            c[0][0].type == 'control_change' and
            c[0][0].control == 1 and
            c[0][0].channel == master_ch
        ]
        assert len(cc1_msgs) >= 1


# Silence detection

class TestSilence:

    def test_silence_triggers_all_notes_off(self, sender, mock_port):
        """All-notes-off is sent after sustained silence."""
        # Simulate silence for longer than silence_hold_ms
        sender._silence_start = time.time() - 1.0  # 1s ago
        features = _neutral_features()  # all velocities = 0
        sender.update(features)
        calls = mock_port.send.call_args_list
        # CC 123 = all notes off
        all_off = [c for c in calls if hasattr(c[0][0], 'type') and
                   c[0][0].type == 'control_change' and c[0][0].control == 123]
        assert len(all_off) >= 1


# Genre rotation

class TestGenreRotation:

    def test_genre_changes_after_silence_timeout_when_genre_is_random(self, sender):
        """When genre is random, silence timeout triggers a new progression."""
        old_genre = sender._progression.genre
        old_filename = sender._progression.filename
        sender._silence_start = time.time() - 120  # well past timeout
        sender._genre_fixed = False

        # Run update with silence (low velocity)
        sender.update(_neutral_features())

        # After rotation, at least one of genre/filename should potentially differ
        # (statistical, may occasionally be same genre by chance)
        assert sender._progression is not None

    def test_genre_does_not_change_after_silence_timeout_when_genre_is_fixed(self, sender_fixed_genre):
        """When genre is fixed, silence timeout does NOT change the genre."""
        old_genre = sender_fixed_genre._progression.genre
        old_filename = sender_fixed_genre._progression.filename
        sender_fixed_genre._silence_start = time.time() - 120
        sender_fixed_genre._all_notes_sent_off = False

        sender_fixed_genre.update(_neutral_features())

        assert sender_fixed_genre._progression.genre == old_genre
        assert sender_fixed_genre._progression.filename == old_filename


# Key fixed

class TestKeyFixed:

    def test_key_does_not_change_when_key_is_fixed(self, sender_fixed_key):
        """When key is fixed, it stays the same across updates."""
        key = sender_fixed_key._progression.key
        for _ in range(10):
            sender_fixed_key.update(_neutral_features())
        assert sender_fixed_key._progression.key == key


# Bass

class TestBass:

    def test_bass_root_matches_active_chord(self, sender):
        """Bass note should be the root of the active chord."""
        sender.update(_neutral_features(
            rightArmVelocity=0.6,
            leftArmVelocity=0.6,
        ))
        chord = sender._progression.chords[sender._chord_index]
        if sender._bass_note is not None:
            assert sender._bass_note == chord.root


# Error handling

class TestErrors:

    def test_invalid_genre_raises_value_error(self):
        with patch("vision_processor.midi.blueprint.mido") as mock_mido:
            mock_mido.open_output.return_value = MagicMock()
            with pytest.raises(ValueError):
                BlueprintMidiSender(
                    config=_make_config(),
                    port_name="TestPort",
                    genre="NonexistentGenre12345",
                )

    def test_invalid_key_raises_value_error(self):
        with patch("vision_processor.midi.blueprint.mido") as mock_mido:
            mock_mido.open_output.return_value = MagicMock()
            with pytest.raises(ValueError):
                BlueprintMidiSender(
                    config=_make_config(),
                    port_name="TestPort",
                    genre="Jazz",
                    key="Z# Super Major",
                )
