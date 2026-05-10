"""
Musical MIDI sender for CuerpoSonoro — percussive pentatonic mode.

Re-imagined as a rhythm-first, harmony-light sonic identity:

  - Hirajōshi pentatonic scale in A (A B C E F) for distinctly Japanese
    color. No harmonic motion: a single scale persists across the whole
    performance so the listener's attention goes to rhythm.
  - Single bass note (A2), re-triggered as a percussive pulse — not a
    sustained drone. The bass density (1/4, 1/8, or 3+3+2 syncopated)
    is controlled by feet position, providing the main rhythmic
    expressivity.
  - All notes are deliberately short (~90ms) so the listener perceives
    them as percussive hits rather than sustained tones, regardless of
    the underlying synth patch's release envelope.
  - Tempo thread fires at 1/16 for maximum subdivision granularity.
    Bass and melody fire on independent patterns over that grid,
    producing polyrhythmic groove from a tiny harmonic vocabulary.
  - Velocity accent table emphasizes the bar downbeat and ghost-notes
    the off-beats, producing groove without any harmonic change.

Pair this sender with a percussive Surge XT patch (mallet, koto,
kalimba, plucked synth). Using a sustained pad like Bloom defeats the
architecture — you'll only hear the attack envelope of every hit.

MPE Channel allocation:
  - Channel 1: Master (global expression CCs)
  - Channel 2: Bass
  - Channel 3: Melody
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import mido

from vision_processor.midi.base import BaseMidiSender


class MusicalMidiSender(BaseMidiSender):
    """Percussive pentatonic sender for the musical mode."""

    # Hirajōshi scale in A (A B C E F). MIDI notes spanning A3 to A5
    # (~2 octaves of pentatonic, 11 notes) for the melody pool. Hand Y
    # position picks an index into this list.
    HIRAJOSHI_A: list[int] = [
        57, 59, 60, 64, 65,   # A3 B3 C4 E4 F4
        69, 71, 72, 76, 77,   # A4 B4 C5 E5 F5
        81,                   # A5
    ]

    BASS_NOTE = 45  # A2 — fixed root, single percussive pulse

    # Every note (bass and melody) gets explicitly turned off after this
    # interval so the listener perceives a hit, not a sustained tone.
    NOTE_DURATION_S = 0.09

    # MPE channels (0-indexed for mido)
    CH_MASTER = 0
    CH_BASS   = 1
    CH_MELODY = 2

    # Bass density patterns: tick positions (mod 16) at which the bass
    # fires within a bar. Selected by feet position.
    BASS_PATTERN_LOW    = frozenset({0, 4, 8, 12})              # 1/4
    BASS_PATTERN_MEDIUM = frozenset({0, 2, 4, 6, 8, 10, 12, 14})  # 1/8
    BASS_PATTERN_HIGH   = frozenset({0, 3, 6, 8, 11, 14})        # 3+3+2

    BASS_BASE_VELOCITY      = 90
    MELODY_BASE_VELOCITY    = 70
    # Default for arm-velocity gate below which melody stays silent —
    # gives the performer a way to "rest" by holding the hand still.
    # Overridable per-instance via the constructor.
    MELODY_TRIGGER_THRESHOLD_DEFAULT = 0.08

    def __init__(
        self,
        port_name: str = "CuerpoSonoro",
        tempo_bpm: int = 120,
        melody_trigger_threshold: float = MELODY_TRIGGER_THRESHOLD_DEFAULT,
    ):
        self.port_name = port_name
        self.tempo_bpm = tempo_bpm
        self.melody_trigger_threshold = melody_trigger_threshold

        # Tempo thread runs at 1/16 always (max granularity needed by
        # the densest bass pattern). Beat = quarter; tick = sixteenth.
        self._tick_period = 60.0 / tempo_bpm / 4

        self.port: Optional[mido.ports.BaseOutput] = None

        # Cross-thread state — read by tempo thread, written by main.
        self._bass_pattern: frozenset[int] = self.BASS_PATTERN_MEDIUM
        self._melody_candidate: Optional[int] = None
        self._melody_candidate_velocity: int = self.MELODY_BASE_VELOCITY
        self._lock = threading.Lock()

        # Tempo thread state
        self._tick = 0
        self._running = False
        self._tempo_thread: Optional[threading.Thread] = None

        self._open_port()
        self._start_tempo_thread()

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _open_port(self) -> None:
        try:
            self.port = mido.open_output(self.port_name, virtual=True)
            print(
                f"[MusicalMidiSender] Port: {self.port_name} | "
                f"Tempo: {self.tempo_bpm} BPM | "
                f"Tick: {self._tick_period * 1000:.0f}ms (1/16)"
            )
        except Exception as e:
            print(f"[MusicalMidiSender] Error opening MIDI port: {e}")

    def _start_tempo_thread(self) -> None:
        self._running = True
        self._tempo_thread = threading.Thread(
            target=self._tempo_loop,
            name="MusicalMidiSender-tempo",
            daemon=True,
        )
        self._tempo_thread.start()

    def close(self) -> None:
        self._running = False
        if self._tempo_thread:
            self._tempo_thread.join(timeout=1.0)
        if self.port:
            self._all_notes_off()
            self.port.close()
            print("[MusicalMidiSender] Port closed.")

    # ------------------------------------------------------------------
    # Public interface (called every frame from main pipeline)
    # ------------------------------------------------------------------

    def update(self, features: dict) -> None:
        if not self.port:
            return
        self._update_bass_density(features)
        self._update_melody_candidate(features)
        self._update_global_expression(features)

    # ------------------------------------------------------------------
    # Per-frame state updates
    # ------------------------------------------------------------------

    def _update_bass_density(self, features: dict) -> None:
        feet_x = features.get("feetCenterX", 0.5)
        if feet_x < 0.33:
            pattern = self.BASS_PATTERN_LOW
        elif feet_x < 0.66:
            pattern = self.BASS_PATTERN_MEDIUM
        else:
            pattern = self.BASS_PATTERN_HIGH
        with self._lock:
            self._bass_pattern = pattern

    def _update_melody_candidate(self, features: dict) -> None:
        arm_velocity = features.get("rightArmVelocity", 0.0)
        if arm_velocity < self.melody_trigger_threshold:
            return

        hand_y = features.get("rightHandY", 0.5)
        scale_index = int(hand_y * (len(self.HIRAJOSHI_A) - 0.001))
        scale_index = max(0, min(len(self.HIRAJOSHI_A) - 1, scale_index))
        target_note = self.HIRAJOSHI_A[scale_index]

        velocity = int(self.MELODY_BASE_VELOCITY + arm_velocity * 50)
        velocity = max(1, min(127, velocity))

        with self._lock:
            self._melody_candidate = target_note
            self._melody_candidate_velocity = velocity

    def _update_global_expression(self, features: dict) -> None:
        head_tilt = features.get("headTilt", 0.0)
        cc_value = max(0, min(127, int(64 + head_tilt * 63)))
        self._control_change(74, cc_value, self.CH_MASTER)

    # ------------------------------------------------------------------
    # Tempo thread (runs at 1/16 subdivision)
    # ------------------------------------------------------------------

    def _tempo_loop(self) -> None:
        while self._running:
            beat_start = time.perf_counter()
            self._fire_tick(self._tick)
            self._tick += 1

            elapsed = time.perf_counter() - beat_start
            sleep_for = self._tick_period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _fire_tick(self, tick: int) -> None:
        if not self.port:
            return

        pos = tick % 16
        accent = self._accent_for_position(pos)

        with self._lock:
            pattern = self._bass_pattern
            candidate = self._melody_candidate
            cand_vel = self._melody_candidate_velocity
            # Consume the candidate: the performer has to keep the hand
            # in motion to keep the melody alive. Avoids a stuck note
            # phrase looping when the dancer pauses.
            self._melody_candidate = None

        if pos in pattern:
            vel = max(1, min(127, self.BASS_BASE_VELOCITY + accent))
            self._fire_short_note(self.BASS_NOTE, vel, self.CH_BASS)

        # Melody fires on the 1/8 grid (every other tick).
        if pos % 2 == 0 and candidate is not None:
            vel = max(1, min(127, cand_vel + accent))
            self._fire_short_note(candidate, vel, self.CH_MELODY)

    @staticmethod
    def _accent_for_position(pos: int) -> int:
        """
        Velocity offset based on tick position within the 16-tick bar.
        Downbeats accented, off-beats softened, 1/16 ghost-noted.
        Produces groove without changing any notes.
        """
        if pos == 0:
            return 25       # bar downbeat
        if pos == 8:
            return 10       # half-bar (beat 3 in 4/4)
        if pos % 4 == 0:
            return 0        # other quarter beats
        if pos % 2 == 0:
            return -5       # 1/8 off-beats
        return -15          # 1/16 in between (ghost notes)

    # ------------------------------------------------------------------
    # Note dispatch
    # ------------------------------------------------------------------

    def _fire_short_note(self, note: int, velocity: int, channel: int) -> None:
        """Send note_on now; schedule note_off after NOTE_DURATION_S."""
        self._note_on(note, velocity, channel)
        timer = threading.Timer(
            self.NOTE_DURATION_S,
            self._note_off,
            args=(note, channel),
        )
        timer.daemon = True
        timer.start()

    # ------------------------------------------------------------------
    # Low-level MIDI
    # ------------------------------------------------------------------

    def _note_on(self, note: int, velocity: int, channel: int) -> None:
        if self.port:
            self.port.send(mido.Message('note_on', note=note, velocity=velocity, channel=channel))

    def _note_off(self, note: int, channel: int) -> None:
        if self.port:
            self.port.send(mido.Message('note_off', note=note, velocity=0, channel=channel))

    def _control_change(self, control: int, value: int, channel: int) -> None:
        if self.port:
            self.port.send(mido.Message('control_change', control=control, value=value, channel=channel))

    def _all_notes_off(self) -> None:
        for ch in (self.CH_BASS, self.CH_MELODY):
            self._control_change(123, 0, ch)
