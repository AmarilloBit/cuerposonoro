"""
Unit tests for chord-related constants shared between ClassicMidiSender
and MusicalMidiSender.

The chord logic (zones, chord definitions, velocity range) is currently
duplicated across both senders. Until that duplication is refactored into
a shared base/mixin, these tests pin the invariant that both classes must
hold identical values — so a change in one without the other fails CI
instead of producing a silent imbalance between modes.

Usage:
    pytest tests/unit/test_chord_parity.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vision_processor.midi.classic import ClassicMidiSender
from vision_processor.midi.musical import MusicalMidiSender


@pytest.mark.parametrize(
    "attr",
    [
        "CHORD_VELOCITY_MIN",
        "CHORD_VELOCITY_MAX",
        "CHORDS",
        "CHORD_ZONES",
    ],
)
def test_chord_constant_matches_between_modes(attr: str) -> None:
    classic_value = getattr(ClassicMidiSender, attr)
    musical_value = getattr(MusicalMidiSender, attr)
    assert classic_value == musical_value, (
        f"{attr} differs between ClassicMidiSender and MusicalMidiSender. "
        f"Classic={classic_value!r} Musical={musical_value!r}. "
        f"Keep them in sync (or refactor the chord logic into a shared base)."
    )


def test_chord_velocity_range_is_lower_than_default_melody_velocity() -> None:
    """
    Sanity: chords stack 3 notes, so their max velocity must stay clearly
    below the melody floor (60 in classic) for the mix to feel balanced.
    """
    assert ClassicMidiSender.CHORD_VELOCITY_MAX < 60, (
        "Chord velocity ceiling should stay below the melody floor (60). "
        "If you raised it, double-check the mix is still balanced."
    )
