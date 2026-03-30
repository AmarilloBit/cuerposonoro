"""
Musical MIDI sender for CuerpoSonoro.

An alternative to ClassicMidiSender designed to produce more structured,
musical output. Key differences:

  - Melody is quantized to a fixed tempo grid (notes only fire on the beat)
  - Note selection uses chord tones, not a fixed scale — the available notes
    change when the chord changes, so the melody always sounds "inside" the harmony
  - Direction of hand movement (up/down) determines where the melody goes;
    speed of movement determines jump size (step vs. leap)
  - Single melody voice (right hand only) for cleaner output

MPE Channel allocation:
  - Channel 1:  Master (global messages)
  - Channel 2:  Chord note 1 (root)
  - Channel 3:  Chord note 2 (3rd)
  - Channel 4:  Chord note 3 (5th)
  - Channel 5:  Melody (single voice)
"""

import mido
import time
import threading
from typing import Optional

from vision_processor.midi.base import BaseMidiSender


class MusicalMidiSender(BaseMidiSender):
    """
    Tempo-quantized, chord-tone melody sender.

    Two threads run simultaneously:
      - Main thread (30fps): receives features, computes melody direction,
        enqueues a note candidate.
      - Tempo thread: fires at the configured BPM subdivision. On each
        beat, if a note candidate is queued, it gets sent to Surge XT.

    This ensures notes land on the pulse regardless of when the gesture
    occurs in the frame, producing rhythmically coherent output.
    """

    # Chord tones for each chord in C Major.
    # Each list contains 5 notes spanning roughly an octave and a half,
    # allowing the melody to move up and down with interesting intervals
    # (thirds, fourths, fifths) rather than just steps.
    CHORD_TONES = {
        "I":  [60, 64, 67, 71, 74],  # C4  E4  G4  B4  D5
        "IV": [65, 69, 72, 76, 79],  # F4  A4  C5  E5  G5
        "V":  [67, 71, 74, 77, 81],  # G4  B4  D5  F5  A5
        "VI": [69, 72, 76, 79, 83],  # A4  C5  E5  G5  B5
    }

    CHORD_ZONES = [
        (0.00, 0.25, "I"),
        (0.25, 0.50, "IV"),
        (0.50, 0.75, "V"),
        (0.75, 1.00, "VI"),
    ]

    CHORDS = {
        "I":   (48, 52, 55),
        "IV":  (53, 57, 60),
        "V":   (55, 59, 62),
        "VI":  (57, 60, 64),
    }

    CH_MASTER      = 0
    CH_CHORD_ROOT  = 1
    CH_CHORD_THIRD = 2
    CH_CHORD_FIFTH = 3
    CH_MELODY      = 4   # single voice

    def __init__(
        self,
        port_name: str = "CuerpoSonoro",
        tempo_bpm: int = 120,
        note_subdivision: int = 8,
        direction_threshold: float = 0.03,
        velocity_threshold: float = 0.4,
        jump_size_slow: int = 1,
        jump_size_fast: int = 2,
    ):
        """
        Args:
            port_name:            Virtual MIDI port name.
            tempo_bpm:            Tempo in BPM. Notes snap to this grid.
            note_subdivision:     4=quarter notes, 8=eighth notes, 16=sixteenth notes.
            direction_threshold:  Minimum hand dy to count as intentional movement.
            velocity_threshold:   Arm velocity above this = "fast" = larger jump.
            jump_size_slow:       Steps to move in chord_tones list for slow movement.
            jump_size_fast:       Steps to move in chord_tones list for fast movement.
        """
        self.port_name          = port_name
        self.tempo_bpm          = tempo_bpm
        self.note_subdivision   = note_subdivision
        self.direction_threshold = direction_threshold
        self.velocity_threshold = velocity_threshold
        self.jump_size_slow     = jump_size_slow
        self.jump_size_fast     = jump_size_fast

        # Beat interval in seconds
        self._beat_interval = 60.0 / tempo_bpm / (note_subdivision / 4)

        # MIDI port
        self.port = None

        # Chord state
        self.current_chord = None
        self.current_chord_notes = []

        # Melody state
        self._melody_index = 2          # start in the middle of the chord tones list
        self._melody_note: Optional[int] = None
        self._note_candidate: Optional[int] = None  # queued for next beat
        self._candidate_velocity: int = 80
        self._note_lock = threading.Lock()

        # Previous hand Y for direction calculation
        self._prev_hand_y: Optional[float] = None

        # Tempo thread
        self._running = False
        self._tempo_thread: Optional[threading.Thread] = None

        self._open_port()
        self._start_tempo_thread()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _open_port(self):
        try:
            self.port = mido.open_output(self.port_name, virtual=True)
            beat_ms = self._beat_interval * 1000
            print(
                f"[MusicalMidiSender] Port: {self.port_name} | "
                f"Tempo: {self.tempo_bpm} BPM | "
                f"Subdivision: 1/{self.note_subdivision} ({beat_ms:.0f}ms/beat)"
            )
        except Exception as e:
            print(f"[MusicalMidiSender] Error opening MIDI port: {e}")

    def _start_tempo_thread(self):
        self._running = True
        self._tempo_thread = threading.Thread(
            target=self._tempo_loop,
            name="MusicalMidiSender-tempo",
            daemon=True,
        )
        self._tempo_thread.start()

    def _tempo_loop(self):
        """
        Runs at the configured BPM subdivision.
        On each beat, fires the queued note candidate (if any).
        """
        while self._running:
            beat_start = time.perf_counter()

            self._fire_beat()

            # Sleep for the remainder of the beat interval
            elapsed = time.perf_counter() - beat_start
            sleep_time = self._beat_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _fire_beat(self):
        """Called on every beat. Dispatches the pending note candidate."""
        if not self.port:
            return

        with self._note_lock:
            candidate = self._note_candidate
            velocity  = self._candidate_velocity
            self._note_candidate = None

        if candidate is not None:
            # Turn off previous melody note
            if self._melody_note is not None:
                self._note_off(self._melody_note, self.CH_MELODY)

            # Fire new note
            self._note_on(candidate, velocity, self.CH_MELODY)
            self._melody_note = candidate
            print(f"[beat] Melody: note {candidate} (vel: {velocity})")

        else:
            # Silence on this beat — let the note ring or cut it
            # Current behaviour: let it ring (more musical, less choppy)
            pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, features: dict):
        """Called every frame (~30fps) by the main pipeline."""
        if not self.port:
            return

        self._update_chords(features)
        self._update_melody_direction(features)
        self._update_global_expression(features)

    def close(self):
        self._running = False
        if self._tempo_thread:
            self._tempo_thread.join(timeout=1.0)
        if self.port:
            self._all_notes_off()
            self.port.close()
            print("[MusicalMidiSender] Port closed.")

    # ------------------------------------------------------------------
    # Chords — same logic as ClassicMidiSender
    # ------------------------------------------------------------------

    def _update_chords(self, features: dict):
        feet_x     = features.get("feetCenterX", 0.5)
        hip_tilt   = features.get("hipTilt", 0.0)
        knee_angle = features.get("kneeAngle", 1.0)

        new_chord = self._get_chord_from_position(feet_x)

        if new_chord != self.current_chord:
            self._change_chord(new_chord, knee_angle)
            # Reset melody index to middle of new chord tones when chord changes
            self._melody_index = 2

        if self.current_chord:
            self._apply_chord_expression(hip_tilt, knee_angle)

    def _get_chord_from_position(self, x: float) -> str:
        for min_x, max_x, chord in self.CHORD_ZONES:
            if min_x <= x < max_x:
                return chord
        return "I"

    def _change_chord(self, new_chord: str, velocity_factor: float):
        for note, ch in zip(self.current_chord_notes,
                            [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH]):
            self._note_off(note, ch)

        self.current_chord_notes = list(self.CHORDS[new_chord])
        self.current_chord = new_chord

        velocity = int(40 + velocity_factor * 87)
        velocity = max(1, min(127, velocity))

        for note, ch in zip(self.current_chord_notes,
                            [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH]):
            self._note_on(note, velocity, ch)

        print(f"[chord] {new_chord} (vel: {velocity})")

    def _apply_chord_expression(self, hip_tilt: float, knee_angle: float):
        pitch_bend = int(8192 + hip_tilt * 4096)
        pitch_bend = max(0, min(16383, pitch_bend))
        for ch in [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH]:
            self._pitch_bend(pitch_bend, ch)

        pressure = int(knee_angle * 127)
        for ch in [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH]:
            self._channel_pressure(pressure, ch)

    # ------------------------------------------------------------------
    # Melody — direction-based, chord-tone navigation
    # ------------------------------------------------------------------

    def _update_melody_direction(self, features: dict):
        """
        Determines melody direction from hand movement and enqueues
        a note candidate for the next beat.

        Direction logic:
          dy > +threshold  → hand moving UP   → melody moves up
          dy < -threshold  → hand moving DOWN → melody moves down
          |dy| < threshold → hand still        → no new candidate (silence)

        Jump size:
          arm_velocity > velocity_threshold → jump_size_fast positions
          otherwise                         → jump_size_slow positions
        """
        hand_y        = features.get("rightHandY", 0.5)
        arm_velocity  = features.get("rightArmVelocity", 0.0)
        elbow_angle   = features.get("rightElbowHipAngle", 0.0)

        if self._prev_hand_y is None:
            self._prev_hand_y = hand_y
            return

        dy = hand_y - self._prev_hand_y
        self._prev_hand_y = hand_y

        # Determine jump size from arm velocity
        jump = self.jump_size_fast if arm_velocity > self.velocity_threshold else self.jump_size_slow

        # Determine direction
        if dy > self.direction_threshold:
            direction = +1   # hand going up in frame = lower Y value = up
        elif dy < -self.direction_threshold:
            direction = -1
        else:
            return           # no movement, no candidate

        # Navigate chord tones list
        chord_key = self.current_chord or "I"
        tones = self.CHORD_TONES[chord_key]

        new_index = self._melody_index + direction * jump
        new_index = max(0, min(len(tones) - 1, new_index))
        self._melody_index = new_index

        target_note = tones[new_index]

        # Velocity from arm speed
        velocity = int(60 + arm_velocity * 67)
        velocity = max(1, min(127, velocity))

        # Enqueue for next beat — only latest candidate counts
        with self._note_lock:
            self._note_candidate    = target_note
            self._candidate_velocity = velocity

        # Pitch bend from elbow angle (applied immediately, not quantized)
        if self._melody_note is not None:
            pitch_bend = int(8192 + elbow_angle * 2048)
            pitch_bend = max(0, min(16383, pitch_bend))
            self._pitch_bend(pitch_bend, self.CH_MELODY)

    # ------------------------------------------------------------------
    # Global expression
    # ------------------------------------------------------------------

    def _update_global_expression(self, features: dict):
        head_tilt = features.get("headTilt", 0.0)
        cc_value  = int(64 + head_tilt * 63)
        cc_value  = max(0, min(127, cc_value))
        self._control_change(74, cc_value, self.CH_MASTER)

    # ------------------------------------------------------------------
    # Low-level MIDI
    # ------------------------------------------------------------------

    def _note_on(self, note: int, velocity: int, channel: int):
        self.port.send(mido.Message('note_on', note=note, velocity=velocity, channel=channel))

    def _note_off(self, note: int, channel: int):
        self.port.send(mido.Message('note_off', note=note, velocity=0, channel=channel))

    def _pitch_bend(self, value: int, channel: int):
        self.port.send(mido.Message('pitchwheel', pitch=value - 8192, channel=channel))

    def _channel_pressure(self, value: int, channel: int):
        self.port.send(mido.Message('aftertouch', value=value, channel=channel))

    def _control_change(self, control: int, value: int, channel: int):
        self.port.send(mido.Message('control_change', control=control, value=value, channel=channel))

    def _all_notes_off(self):
        for ch in [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH, self.CH_MELODY]:
            self._control_change(123, 0, ch)
