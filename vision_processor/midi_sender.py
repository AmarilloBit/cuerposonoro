"""
Backward-compatible alias for ClassicMidiSender.

All existing imports and tests use:
    from vision_processor.midi_sender import MidiSender

This file keeps that working without any changes to those files.
New code should import directly from the midi subpackage:
    from vision_processor.midi.classic import ClassicMidiSender
    from vision_processor.midi.rhythmical import RhythmicalMidiSender
"""

from vision_processor.midi.classic import ClassicMidiSender as MidiSender

__all__ = ["MidiSender"]
