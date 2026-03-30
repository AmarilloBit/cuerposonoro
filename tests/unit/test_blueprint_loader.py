"""
Unit tests for vision_processor/midi/blueprint_loader.py

Tests the MIDI file parser and BlueprintLibrary with the real Unison MIDI
Blueprint library at assets/midi/.

Usage:
    cd ~/cuerposonoro
    pytest tests/unit/test_blueprint_loader.py -v
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vision_processor.midi.blueprint_loader import BlueprintLibrary, Chord, Progression

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "../../assets/midi")


@pytest.fixture
def library():
    """BlueprintLibrary pointing at the real assets."""
    return BlueprintLibrary(ASSETS_PATH)


# Scanning

class TestScan:

    def test_scan_finds_genres(self, library):
        """assets/midi/ contains at least one genre."""
        assert len(library.genres) >= 1

    def test_genres_are_strings(self, library):
        for g in library.genres:
            assert isinstance(g, str)

    def test_known_genre_present(self, library):
        """Jazz should be in the library."""
        assert "Jazz" in library.genres


# Parsing

class TestParsing:

    def test_parse_mid_returns_chords(self, library):
        """A known .mid file yields at least one Chord."""
        prog = library.random_progression(genre="Jazz")
        assert len(prog.chords) >= 1

    def test_chord_has_root(self, library):
        """root equals notes[0] (lowest note)."""
        prog = library.random_progression(genre="Jazz")
        for chord in prog.chords:
            assert chord.root == chord.notes[0]

    def test_chord_notes_sorted_ascending(self, library):
        """Notes within each chord are sorted ascending."""
        prog = library.random_progression(genre="Jazz")
        for chord in prog.chords:
            assert chord.notes == sorted(chord.notes)

    def test_chord_has_duration(self, library):
        """Each chord has a positive duration."""
        prog = library.random_progression(genre="Jazz")
        for chord in prog.chords:
            assert chord.duration_ticks > 0


# Random progression

class TestRandomProgression:

    def test_returns_progression(self, library):
        prog = library.random_progression()
        assert isinstance(prog, Progression)

    def test_progression_has_genre(self, library):
        prog = library.random_progression()
        assert prog.genre in library.genres

    def test_progression_has_key(self, library):
        prog = library.random_progression()
        assert isinstance(prog.key, str) and len(prog.key) > 0

    def test_progression_has_filename(self, library):
        prog = library.random_progression()
        assert prog.filename.endswith(".mid")

    def test_progression_has_chords(self, library):
        prog = library.random_progression()
        assert len(prog.chords) >= 1

    def test_random_progression_different_genres_on_repeated_calls(self, library):
        """Over 50 random picks, we should see more than one genre."""
        genres_seen = set()
        for _ in range(50):
            prog = library.random_progression()
            genres_seen.add(prog.genre)
        assert len(genres_seen) > 1

    def test_fixed_genre(self, library):
        """When genre is specified, all progressions come from that genre."""
        for _ in range(10):
            prog = library.random_progression(genre="Jazz")
            assert prog.genre == "Jazz"

    def test_fixed_key(self, library):
        """When key is specified, all progressions use that key."""
        for _ in range(10):
            prog = library.random_progression(genre="Jazz", key="C Major")
            assert "C Major" in prog.key


# Transposition

class TestTransposition:

    def test_transposition_puts_root_in_octave_3(self, library):
        """Root of the first chord should be in MIDI octave 3 (48-59)."""
        for _ in range(20):
            prog = library.random_progression()
            first_root = prog.chords[0].root
            assert 48 <= first_root <= 59, (
                f"First chord root {first_root} not in octave 3 (48-59), "
                f"file: {prog.filename}"
            )


# Corrupt file handling

class TestCorruptFile:

    def test_corrupt_file_is_skipped_gracefully(self):
        """A corrupt .mid file should be skipped without raising."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal genre/key/progression directory structure
            genre_dir = os.path.join(tmpdir, "TestGenre")
            key_dir = os.path.join(genre_dir, "01 - C Major - A Minor")
            prog_dir = os.path.join(key_dir, "Chord Progressions", "01 - C Major Progressions")
            os.makedirs(prog_dir)

            # Write a corrupt .mid file
            corrupt_path = os.path.join(prog_dir, "Corrupt Prog.mid")
            with open(corrupt_path, "wb") as f:
                f.write(b"not a midi file")

            lib = BlueprintLibrary(tmpdir)
            # Should not raise, but the genre may have no valid progressions
            assert isinstance(lib.genres, list)


# Error handling

class TestErrors:

    def test_invalid_genre_raises_value_error(self, library):
        with pytest.raises(ValueError):
            library.random_progression(genre="NonexistentGenre12345")

    def test_invalid_key_raises_value_error(self, library):
        with pytest.raises(ValueError):
            library.random_progression(genre="Jazz", key="Z# Super Major")
