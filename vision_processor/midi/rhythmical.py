"""
Rhythmical MIDI sender for CuerpoSonoro — percussive pentatonic mode.

Every one of the 17 kinematic features extracted by FeatureExtractor maps
to audible output. Stillness produces silence; activity produces sound
that varies with the kind of activity.

  Master gate / global modulation (CH 0):
    energy            → silence below ENERGY_GATE; scales global velocity above
    symmetry          → CC10 (pan)
    smoothness        → CC1 (mod wheel)
    armAngle          → CC11 (expression / volume)
    verticalExtension → ±12 semitone octave shift on all notes
    headTilt          → CC74 (filter cutoff)

  Bass (CH 1):
    feetCenterX       → density pattern (1/4, 1/8, 3+3+2 syncopated)
    kneeAngle         → bass-only octave shift (bent → -12)
    hipTilt           → pitch bend on bass channel

  Right melody (CH 2):
    rightHandY        → index in Hirajōshi scale
    rightArmVelocity  → fire density (1/2/4/8 hits per bar) + velocity
    rightHandJerk     → off-grid accent fire on rising threshold edge
    rightElbowHipAngle→ pitch bend on right melody channel

  Left melody (CH 3):
    leftHandY         → index in Hirajōshi scale
    leftArmVelocity   → fire density + velocity
    leftHandJerk      → off-grid accent fire on rising threshold edge
    leftElbowHipAngle → pitch bend on left melody channel

Pair with a percussive Surge XT patch (mallet, koto, kalimba, plucked
synth). Set up a layer/instance listening on channels 2-4 (CH 1-3 in
0-indexed mido terms). All notes are ~90ms duration regardless of patch
release.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import mido

from vision_processor.midi.base import BaseMidiSender


class RhythmicalMidiSender(BaseMidiSender):
    """Percussive pentatonic sender — uses all 17 kinematic features."""

    # Hirajōshi scale in A (A B C E F). MIDI notes spanning A3 to A5.
    HIRAJOSHI_A: list[int] = [
        57, 59, 60, 64, 65,   # A3 B3 C4 E4 F4
        69, 71, 72, 76, 77,   # A4 B4 C5 E5 F5
        81,                   # A5
    ]

    BASS_NOTE = 45  # A2 — fixed pentatonic root

    NOTE_DURATION_S = 0.09  # all notes percussive

    # MPE channels (0-indexed for mido)
    CH_MASTER   = 0
    CH_BASS     = 1
    CH_MELODY_R = 2
    CH_MELODY_L = 3

    # ---- Bass density patterns (tick positions mod 16) ----
    BASS_PATTERN_LOW    = frozenset({0, 4, 8, 12})              # 1/4
    BASS_PATTERN_MEDIUM = frozenset({0, 2, 4, 6, 8, 10, 12, 14})  # 1/8
    BASS_PATTERN_HIGH   = frozenset({0, 3, 6, 8, 11, 14})        # 3+3+2

    # ---- Melody density tiers (selected by arm_velocity per hand) ----
    MELODY_PATTERN_SILENT  = frozenset()
    MELODY_PATTERN_SPARSE  = frozenset({0})                          # 1/bar
    MELODY_PATTERN_HALF    = frozenset({0, 8})                       # 2/bar
    MELODY_PATTERN_QUARTER = frozenset({0, 4, 8, 12})                # 4/bar
    MELODY_PATTERN_EIGHTH  = frozenset({0, 2, 4, 6, 8, 10, 12, 14})  # 8/bar

    BASS_BASE_VELOCITY   = 90
    MELODY_BASE_VELOCITY = 70

    # ---- Energy gate / global gain ----
    # Below ENERGY_GATE the sender is fully silent (no MIDI emitted at all).
    # Between ENERGY_GATE and ENERGY_FULL, the global velocity factor scales
    # linearly from GATE_VELOCITY_FACTOR to FULL_VELOCITY_FACTOR.
    ENERGY_GATE              = 0.05
    ENERGY_FULL              = 0.60
    GATE_VELOCITY_FACTOR     = 0.40
    FULL_VELOCITY_FACTOR     = 1.20

    # ---- Per-hand thresholds ----
    # arm_velocity below this leaves the corresponding hand's melody silent.
    MELODY_VELOCITY_FLOOR    = 0.05
    # jerk above this fires an off-grid accent (rising-edge triggered).
    JERK_ACCENT_THRESHOLD    = 0.40
    JERK_ACCENT_VELOCITY     = 110

    # ---- Octave-shift zones ----
    # verticalExtension thresholds for global ±12 semitone shift.
    VERT_EXT_LOW_THRESHOLD   = 0.33
    VERT_EXT_HIGH_THRESHOLD  = 0.66
    # kneeAngle threshold for bass-only -12 shift (bent knees → low bass).
    KNEE_BEND_THRESHOLD      = 0.50

    def __init__(
        self,
        port_name: str = "CuerpoSonoro",
        tempo_bpm: int = 120,
        melody_velocity_floor: float = MELODY_VELOCITY_FLOOR,
    ):
        self.port_name = port_name
        self.tempo_bpm = tempo_bpm
        self.melody_velocity_floor = melody_velocity_floor

        # Tempo thread runs at 1/16 (max needed by densest pattern).
        self._tick_period = 60.0 / tempo_bpm / 4

        self.port: Optional[mido.ports.BaseOutput] = None

        # ---- Cross-thread state. All reads/writes go through self._lock. ----
        # Master / global modulation
        self._energy_gain      = 0.0   # 0 = silenced; scaled velocity factor otherwise
        self._octave_shift     = 0     # from verticalExtension, applies to bass + melody
        self._bass_octave_off  = 0     # from kneeAngle, applies to bass only

        # Bass
        self._bass_pattern: frozenset[int] = self.BASS_PATTERN_MEDIUM

        # Per-hand melody
        self._right_pattern: frozenset[int] = self.MELODY_PATTERN_SILENT
        self._left_pattern:  frozenset[int] = self.MELODY_PATTERN_SILENT
        self._right_candidate: Optional[int] = None
        self._left_candidate:  Optional[int] = None
        self._right_velocity = self.MELODY_BASE_VELOCITY
        self._left_velocity  = self.MELODY_BASE_VELOCITY

        # Jerk edge detection (we fire only on rising edge, not while held above)
        self._right_jerk_above = False
        self._left_jerk_above  = False

        self._lock = threading.Lock()

        # Tempo thread
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
                f"[RhythmicalMidiSender] Port: {self.port_name} | "
                f"Tempo: {self.tempo_bpm} BPM | "
                f"Tick: {self._tick_period * 1000:.0f}ms (1/16) | "
                f"Channels: master={self.CH_MASTER}, bass={self.CH_BASS}, "
                f"R-mel={self.CH_MELODY_R}, L-mel={self.CH_MELODY_L}"
            )
        except Exception as e:
            print(f"[RhythmicalMidiSender] Error opening MIDI port: {e}")

    def _start_tempo_thread(self) -> None:
        self._running = True
        self._tempo_thread = threading.Thread(
            target=self._tempo_loop,
            name="RhythmicalMidiSender-tempo",
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
            print("[RhythmicalMidiSender] Port closed.")

    # ------------------------------------------------------------------
    # Public interface (called every frame from main pipeline)
    # ------------------------------------------------------------------

    def update(self, features: dict) -> None:
        if not self.port:
            return

        # 1. Energy gate / global gain — must come first because everything
        #    else short-circuits to silence below the gate.
        energy = features.get("energy", 0.0)
        gain = self._compute_energy_gain(energy)

        # 2. Octave shifts (always computed; affect all subsequent firings).
        vert_ext = features.get("verticalExtension", 0.5)
        knee     = features.get("kneeAngle", 1.0)
        octave_shift    = self._compute_octave_shift(vert_ext)
        bass_octave_off = self._compute_bass_octave_offset(knee)

        # 3. Master CCs (always sent so a still dancer's filter still tracks).
        self._send_master_ccs(features)

        # 4. Bass + melody state for the tempo thread to consume.
        bass_pattern = self._compute_bass_pattern(features)
        right_pattern, right_vel = self._compute_melody_pattern_and_velocity(
            features.get("rightArmVelocity", 0.0)
        )
        left_pattern, left_vel = self._compute_melody_pattern_and_velocity(
            features.get("leftArmVelocity", 0.0)
        )

        right_candidate = self._compute_melody_note(
            features.get("rightHandY", 0.5), octave_shift
        )
        left_candidate = self._compute_melody_note(
            features.get("leftHandY", 0.5), octave_shift
        )

        with self._lock:
            self._energy_gain     = gain
            self._octave_shift    = octave_shift
            self._bass_octave_off = bass_octave_off
            self._bass_pattern    = bass_pattern
            self._right_pattern   = right_pattern
            self._left_pattern    = left_pattern
            self._right_velocity  = right_vel
            self._left_velocity   = left_vel
            # Only queue a candidate if the hand is actively moving above the floor.
            if features.get("rightArmVelocity", 0.0) >= self.melody_velocity_floor:
                self._right_candidate = right_candidate
            if features.get("leftArmVelocity", 0.0) >= self.melody_velocity_floor:
                self._left_candidate = left_candidate

        # 5. Per-hand pitch bend on melody channels (continuous expression).
        self._send_melody_pitch_bends(features)

        # 6. Bass pitch bend from hipTilt.
        self._send_bass_pitch_bend(features.get("hipTilt", 0.0))

        # 7. Jerk-driven off-grid accents (rising-edge triggered).
        self._maybe_fire_jerk_accent(
            features.get("rightHandJerk", 0.0),
            side="right",
            note=right_candidate,
            gain=gain,
        )
        self._maybe_fire_jerk_accent(
            features.get("leftHandJerk", 0.0),
            side="left",
            note=left_candidate,
            gain=gain,
        )

    # ------------------------------------------------------------------
    # Per-frame derivation helpers
    # ------------------------------------------------------------------

    def _compute_energy_gain(self, energy: float) -> float:
        """0.0 below gate (silence), GATE→FULL factor mapped linearly above."""
        if energy < self.ENERGY_GATE:
            return 0.0
        if energy >= self.ENERGY_FULL:
            return self.FULL_VELOCITY_FACTOR
        span = self.ENERGY_FULL - self.ENERGY_GATE
        t = (energy - self.ENERGY_GATE) / span
        return self.GATE_VELOCITY_FACTOR + t * (
            self.FULL_VELOCITY_FACTOR - self.GATE_VELOCITY_FACTOR
        )

    def _compute_octave_shift(self, vert_ext: float) -> int:
        if vert_ext < self.VERT_EXT_LOW_THRESHOLD:
            return -12
        if vert_ext > self.VERT_EXT_HIGH_THRESHOLD:
            return 12
        return 0

    def _compute_bass_octave_offset(self, knee_angle: float) -> int:
        return -12 if knee_angle < self.KNEE_BEND_THRESHOLD else 0

    def _compute_bass_pattern(self, features: dict) -> frozenset[int]:
        feet_x = features.get("feetCenterX", 0.5)
        if feet_x < 0.33:
            return self.BASS_PATTERN_LOW
        if feet_x < 0.66:
            return self.BASS_PATTERN_MEDIUM
        return self.BASS_PATTERN_HIGH

    def _compute_melody_pattern_and_velocity(
        self, arm_velocity: float
    ) -> tuple[frozenset[int], int]:
        """Hand's arm_velocity selects density tier and base velocity."""
        if arm_velocity < self.melody_velocity_floor:
            return self.MELODY_PATTERN_SILENT, self.MELODY_BASE_VELOCITY
        if arm_velocity < 0.20:
            pattern = self.MELODY_PATTERN_SPARSE
        elif arm_velocity < 0.40:
            pattern = self.MELODY_PATTERN_HALF
        elif arm_velocity < 0.65:
            pattern = self.MELODY_PATTERN_QUARTER
        else:
            pattern = self.MELODY_PATTERN_EIGHTH

        velocity = int(self.MELODY_BASE_VELOCITY + arm_velocity * 50)
        velocity = max(1, min(127, velocity))
        return pattern, velocity

    def _compute_melody_note(self, hand_y: float, octave_shift: int) -> int:
        scale_index = int(hand_y * (len(self.HIRAJOSHI_A) - 0.001))
        scale_index = max(0, min(len(self.HIRAJOSHI_A) - 1, scale_index))
        return self.HIRAJOSHI_A[scale_index] + octave_shift

    # ------------------------------------------------------------------
    # Master CCs (sent every frame regardless of gate)
    # ------------------------------------------------------------------

    def _send_master_ccs(self, features: dict) -> None:
        symmetry   = features.get("symmetry", 0.0)
        smoothness = features.get("smoothness", 0.5)
        arm_angle  = features.get("armAngle", 0.0)
        head_tilt  = features.get("headTilt", 0.0)

        # CC10 pan: -1 → 0, 0 → 64, +1 → 127
        cc10 = max(0, min(127, int(64 + symmetry * 63)))
        # CC1 mod wheel: 0–1 → 0–127
        cc1 = max(0, min(127, int(smoothness * 127)))
        # CC11 expression: 0–1 → 0–127
        cc11 = max(0, min(127, int(arm_angle * 127)))
        # CC74 filter cutoff: -1 → 1, 0 → 64, +1 → 127
        cc74 = max(0, min(127, int(64 + head_tilt * 63)))

        self._control_change(1,  cc1,  self.CH_MASTER)
        self._control_change(10, cc10, self.CH_MASTER)
        self._control_change(11, cc11, self.CH_MASTER)
        self._control_change(74, cc74, self.CH_MASTER)

    def _send_melody_pitch_bends(self, features: dict) -> None:
        right_elbow = features.get("rightElbowHipAngle", 0.0)
        left_elbow  = features.get("leftElbowHipAngle", 0.0)
        # ±2048 → ±~25% of a full semitone in default 2-semitone bend range.
        self._pitch_bend(int(8192 + right_elbow * 2048), self.CH_MELODY_R)
        self._pitch_bend(int(8192 + left_elbow  * 2048), self.CH_MELODY_L)

    def _send_bass_pitch_bend(self, hip_tilt: float) -> None:
        self._pitch_bend(int(8192 + hip_tilt * 4096), self.CH_BASS)

    # ------------------------------------------------------------------
    # Jerk-driven off-grid accents
    # ------------------------------------------------------------------

    def _maybe_fire_jerk_accent(
        self, jerk: float, side: str, note: int, gain: float
    ) -> None:
        """Fire an immediate accent on rising edge across JERK_ACCENT_THRESHOLD."""
        above = jerk > self.JERK_ACCENT_THRESHOLD
        if side == "right":
            was_above = self._right_jerk_above
            self._right_jerk_above = above
            channel = self.CH_MELODY_R
        else:
            was_above = self._left_jerk_above
            self._left_jerk_above = above
            channel = self.CH_MELODY_L

        if above and not was_above and gain > 0.0:
            vel = max(1, min(127, int(self.JERK_ACCENT_VELOCITY * gain)))
            self._fire_short_note(note, vel, channel)

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

        with self._lock:
            gain            = self._energy_gain
            octave_shift    = self._octave_shift
            bass_octave_off = self._bass_octave_off
            bass_pattern    = self._bass_pattern
            r_pattern       = self._right_pattern
            l_pattern       = self._left_pattern
            r_candidate     = self._right_candidate
            l_candidate     = self._left_candidate
            r_vel_base      = self._right_velocity
            l_vel_base      = self._left_velocity
            # Consume melody candidates: a still hand silences in 1 tick.
            self._right_candidate = None
            self._left_candidate  = None

        # Energy gate: below threshold, emit nothing.
        if gain <= 0.0:
            return

        pos = tick % 16
        accent = self._accent_for_position(pos)

        if pos in bass_pattern:
            bass_note = self.BASS_NOTE + octave_shift + bass_octave_off
            vel = self._scale_velocity(self.BASS_BASE_VELOCITY + accent, gain)
            self._fire_short_note(bass_note, vel, self.CH_BASS)

        if pos in r_pattern and r_candidate is not None:
            vel = self._scale_velocity(r_vel_base + accent, gain)
            self._fire_short_note(r_candidate, vel, self.CH_MELODY_R)

        if pos in l_pattern and l_candidate is not None:
            vel = self._scale_velocity(l_vel_base + accent, gain)
            self._fire_short_note(l_candidate, vel, self.CH_MELODY_L)

    @staticmethod
    def _accent_for_position(pos: int) -> int:
        if pos == 0:
            return 25
        if pos == 8:
            return 10
        if pos % 4 == 0:
            return 0
        if pos % 2 == 0:
            return -5
        return -15

    @staticmethod
    def _scale_velocity(base: int, gain: float) -> int:
        return max(1, min(127, int(base * gain)))

    # ------------------------------------------------------------------
    # Note dispatch
    # ------------------------------------------------------------------

    def _fire_short_note(self, note: int, velocity: int, channel: int) -> None:
        note = max(0, min(127, note))
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

    def _pitch_bend(self, value: int, channel: int) -> None:
        if self.port:
            value = max(0, min(16383, value))
            self.port.send(mido.Message('pitchwheel', pitch=value - 8192, channel=channel))

    def _control_change(self, control: int, value: int, channel: int) -> None:
        if self.port:
            self.port.send(mido.Message('control_change', control=control, value=value, channel=channel))

    def _all_notes_off(self) -> None:
        for ch in (self.CH_BASS, self.CH_MELODY_R, self.CH_MELODY_L):
            self._control_change(123, 0, ch)
