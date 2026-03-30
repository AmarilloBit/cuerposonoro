"""
MIDI file parser for the Unison MIDI Blueprint library.

Scans assets/midi/, parses .mid files to extract chord progressions,
and exposes a BlueprintLibrary class for random progression selection.
"""

import logging
import os
import random
from dataclasses import dataclass

import mido

logger = logging.getLogger(__name__)


@dataclass
class Chord:
    notes: list[int]        # MIDI note numbers (absolute), sorted ascending
    root: int               # lowest note in the chord (notes[0])
    duration_ticks: int     # original duration from the MIDI file


@dataclass
class Progression:
    genre: str
    key: str                # e.g. "C Major"
    filename: str           # original .mid filename
    chords: list[Chord]     # ordered list of chords


def _parse_midi_file(filepath: str) -> list[Chord]:
    """
    Parse a single .mid file and return a list of Chord objects.

    Collects simultaneous note_on messages (within 10 ticks) as one chord.
    Returns an empty list if the file is corrupt or has no note_on events.
    """
    try:
        mid = mido.MidiFile(filepath)
    except Exception:
        logger.warning("Could not parse MIDI file: %s", filepath)
        return []

    chords = []
    current_notes = []
    current_tick = 0
    chord_start_tick = 0

    for track in mid.tracks:
        abs_tick = 0
        pending_notes = []
        pending_start = 0

        for msg in track:
            abs_tick += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                if not pending_notes or (abs_tick - pending_start) <= 10:
                    if not pending_notes:
                        pending_start = abs_tick
                    pending_notes.append(msg.note)
                else:
                    # Flush the previous chord
                    duration = abs_tick - pending_start
                    _flush_chord(chords, pending_notes, duration)
                    pending_notes = [msg.note]
                    pending_start = abs_tick

            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                # Use note_off to estimate duration of the last pending group
                if pending_notes and not chords:
                    pass  # duration will be calculated on next chord or end
                # If we see note_off after a chord was started, compute duration
                if pending_notes and (abs_tick - pending_start) > 10:
                    duration = abs_tick - pending_start
                    _flush_chord(chords, pending_notes, duration)
                    pending_notes = []

        # Flush any remaining notes
        if pending_notes:
            # Use a default duration if we can't compute one
            duration = abs_tick - pending_start if abs_tick > pending_start else 480
            _flush_chord(chords, pending_notes, duration)

    if not chords:
        logger.warning("No chords found in: %s", filepath)
        return []

    # Transpose so the root of the first chord falls in octave 3 (MIDI 48-59)
    _transpose_to_octave_3(chords)

    return chords


def _flush_chord(chords: list[Chord], notes: list[int], duration: int):
    """Sort notes ascending and append a Chord to the list."""
    sorted_notes = sorted(notes)
    chords.append(Chord(
        notes=sorted_notes,
        root=sorted_notes[0],
        duration_ticks=max(1, duration),
    ))


def _transpose_to_octave_3(chords: list[Chord]):
    """Transpose all chords so the first chord's root is in MIDI octave 3 (48-59)."""
    if not chords:
        return
    first_root = chords[0].root
    # Target: bring first_root into range 48-59
    target_octave_start = 48
    current_octave_note = first_root % 12
    target_note = target_octave_start + current_octave_note
    shift = target_note - first_root

    if shift == 0:
        return

    for chord in chords:
        chord.notes = [n + shift for n in chord.notes]
        chord.root = chord.notes[0]


def _extract_key_name(folder_name: str) -> str:
    """
    Extract key name from folder like '01 - C Major - A Minor'.
    Returns the part after the first ' - ' (e.g. 'C Major - A Minor').
    """
    parts = folder_name.split(" - ", 1)
    if len(parts) > 1:
        return parts[1]
    return folder_name


def _extract_progression_key(folder_name: str) -> str:
    """
    Extract key from progression subfolder like '01 - C Major Progressions'.
    Returns e.g. 'C Major'.
    """
    parts = folder_name.split(" - ", 1)
    if len(parts) > 1:
        return parts[1].replace(" Progressions", "")
    return folder_name


class BlueprintLibrary:
    """
    Scans the Unison MIDI Blueprint library and provides random
    progression selection by genre and key.

    All .mid files are indexed at construction time but parsed lazily
    (on first access) and cached.
    """

    def __init__(self, assets_path: str):
        self._assets_path = assets_path
        # Structure: {genre: {key: [filepath, ...]}}
        self._index: dict[str, dict[str, list[str]]] = {}
        # Cache: {filepath: list[Chord]}
        self._cache: dict[str, list[Chord]] = {}
        self._scan()

    @property
    def genres(self) -> list[str]:
        return sorted(self._index.keys())

    def _scan(self):
        """Walk the assets directory and build the file index."""
        if not os.path.isdir(self._assets_path):
            logger.warning("Assets path does not exist: %s", self._assets_path)
            return

        for genre_name in sorted(os.listdir(self._assets_path)):
            genre_path = os.path.join(self._assets_path, genre_name)
            if not os.path.isdir(genre_path):
                continue

            genre_files: dict[str, list[str]] = {}

            for key_folder in sorted(os.listdir(genre_path)):
                key_path = os.path.join(genre_path, key_folder)
                if not os.path.isdir(key_path):
                    continue

                key_name = _extract_key_name(key_folder)

                # Look for Chord Progressions subdirectory
                chord_prog_path = os.path.join(key_path, "Chord Progressions")
                if not os.path.isdir(chord_prog_path):
                    continue

                midi_files = []
                for prog_folder in os.listdir(chord_prog_path):
                    prog_path = os.path.join(chord_prog_path, prog_folder)
                    if not os.path.isdir(prog_path):
                        continue
                    for fname in os.listdir(prog_path):
                        if fname.lower().endswith(".mid"):
                            midi_files.append(os.path.join(prog_path, fname))

                if midi_files:
                    genre_files[key_name] = midi_files

            if genre_files:
                self._index[genre_name] = genre_files

    def _get_chords(self, filepath: str) -> list[Chord]:
        """Parse and cache chords from a MIDI file."""
        if filepath not in self._cache:
            self._cache[filepath] = _parse_midi_file(filepath)
        return self._cache[filepath]

    def random_progression(
        self, genre: str | None = None, key: str | None = None
    ) -> Progression:
        """
        Pick a random progression. Optionally constrain by genre and/or key.

        Raises ValueError if the specified genre or key is not found.
        """
        # Select genre
        if genre is not None:
            if genre not in self._index:
                available = ", ".join(self.genres)
                raise ValueError(
                    f"Genre '{genre}' not found. Available: {available}"
                )
            selected_genre = genre
        else:
            selected_genre = random.choice(self.genres)

        genre_keys = self._index[selected_genre]

        # Select key
        if key is not None:
            # Case-insensitive match: find a key_name that contains the given key
            matching_keys = [
                k for k in genre_keys
                if key.lower() in k.lower()
            ]
            if not matching_keys:
                available = ", ".join(sorted(genre_keys.keys()))
                raise ValueError(
                    f"Key '{key}' not found in genre '{selected_genre}'. "
                    f"Available: {available}"
                )
            selected_key = random.choice(matching_keys)
        else:
            selected_key = random.choice(list(genre_keys.keys()))

        # Select a random MIDI file
        midi_files = genre_keys[selected_key]
        filepath = random.choice(midi_files)

        chords = self._get_chords(filepath)

        # If this file had no valid chords, try others in the same key
        if not chords:
            for alt_path in midi_files:
                if alt_path != filepath:
                    chords = self._get_chords(alt_path)
                    if chords:
                        filepath = alt_path
                        break

        # If still no chords, try another key in the same genre
        if not chords:
            for alt_key, alt_files in genre_keys.items():
                if alt_key != selected_key:
                    for alt_path in alt_files:
                        chords = self._get_chords(alt_path)
                        if chords:
                            filepath = alt_path
                            selected_key = alt_key
                            break
                    if chords:
                        break

        return Progression(
            genre=selected_genre,
            key=selected_key,
            filename=os.path.basename(filepath),
            chords=chords,
        )
