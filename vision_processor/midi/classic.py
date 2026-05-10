"""
Classic MIDI/MPE sender for CuerpoSonoro.

Original implementation: hand Y position selects note, jerk triggers it.
Two melody voices (right and left hand), full MPE expression.

This module is a direct move of the original midi_sender.py — no logic
was changed. The only difference is that MidiSender now inherits from
BaseMidiSender so it can be used interchangeably with other strategies.

MPE Channel allocation:
- Channel 1:  Master (global messages)
- Channel 2:  Chord note 1 (root)
- Channel 3:  Chord note 2 (3rd)
- Channel 4:  Chord note 3 (5th)
- Channel 5:  Melody right hand (lower octave)
- Channel 6:  Melody left hand (higher octave)
"""

import mido
import time

from vision_processor.midi.base import BaseMidiSender


class ClassicMidiSender(BaseMidiSender):
    """
    Sends motion features to Surge XT via MPE/MIDI.

    Note selection: hand Y position maps directly to a note in C Major.
    Note trigger:   jerk (sudden movement) above threshold fires the note.
    Two voices:     right hand (lower octave) and left hand (higher octave).
    """

    # Scale definition: C Major
    SCALE_NOTES = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B

    # Chord definitions for C Major (MIDI note numbers)
    CHORDS = {
        "I":   (48, 52, 55),  # C3, E3, G3  (C Major)
        "IV":  (53, 57, 60),  # F3, A3, C4  (F Major)
        "V":   (55, 59, 62),  # G3, B3, D4  (G Major)
        "VI":  (57, 60, 64),  # A3, C4, E4  (A minor)
    }

    CHORD_ZONES = [
        (0.00, 0.25, "I"),
        (0.25, 0.50, "IV"),
        (0.50, 0.75, "V"),
        (0.75, 1.00, "VI"),
    ]

    # MPE Channel assignments (0-indexed)
    CH_MASTER       = 0
    CH_CHORD_ROOT   = 1
    CH_CHORD_THIRD  = 2
    CH_CHORD_FIFTH  = 3
    CH_MELODY_RIGHT = 4
    CH_MELODY_LEFT  = 5

    MELODY_RIGHT_BASE = 48   # C3
    MELODY_LEFT_BASE  = 72   # C5

    JERK_THRESHOLD      = 0.4
    HIP_TILT_THRESHOLD  = 0.2

    # Velocity ranges (0-127). Chords stack 3 notes simultaneously, so they
    # need a much lower velocity than the melody to sit underneath it in the
    # mix instead of dominating it.
    CHORD_VELOCITY_MIN = 15
    CHORD_VELOCITY_MAX = 55

    def __init__(self, port_name: str = "CuerpoSonoro"):
        self.port_name = port_name
        self.port = None

        self.current_chord = None
        self.current_chord_notes = []
        self.melody_right_note = None
        self.melody_left_note = None
        self.melody_right_note_time = 0
        self.melody_left_note_time = 0

        self.base_note_duration = 0.3
        self.min_note_duration  = 0.15
        self.max_note_duration  = 0.6

        self._open_port()

    def _open_port(self):
        try:
            self.port = mido.open_output(self.port_name, virtual=True)
            print(f"MIDI port opened: {self.port_name}")
        except Exception as e:
            print(f"Error opening MIDI port: {e}")
            print("Available ports:", mido.get_output_names())

    def close(self):
        if self.port:
            self._all_notes_off()
            self.port.close()
            print("MIDI port closed.")

    def update(self, features: dict):
        if not self.port:
            return
        self._update_chords(features)
        self._update_melody(features)
        self._update_global_expression(features)

    # ------------------------------------------------------------------
    # Chords
    # ------------------------------------------------------------------

    def _update_chords(self, features: dict):
        feet_x    = features.get("feetCenterX", 0.5)
        hip_tilt  = features.get("hipTilt", 0.0)
        knee_angle = features.get("kneeAngle", 1.0)

        new_chord = self._get_chord_from_position(feet_x)
        if new_chord != self.current_chord:
            self._change_chord(new_chord, knee_angle)

        if self.current_chord:
            self._apply_chord_expression(hip_tilt, knee_angle)

    def _get_chord_from_position(self, x: float) -> str:
        for min_x, max_x, chord in self.CHORD_ZONES:
            if min_x <= x < max_x:
                return chord
        return "I"

    def _change_chord(self, new_chord: str, velocity_factor: float):
        for note, channel in zip(self.current_chord_notes,
                                 [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH]):
            self._note_off(note, channel)

        self.current_chord_notes = list(self.CHORDS[new_chord])
        self.current_chord = new_chord

        velocity_range = self.CHORD_VELOCITY_MAX - self.CHORD_VELOCITY_MIN
        velocity = int(self.CHORD_VELOCITY_MIN + velocity_factor * velocity_range)
        velocity = max(1, min(127, velocity))

        for note, channel in zip(self.current_chord_notes,
                                 [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH]):
            self._note_on(note, velocity, channel)

        print(f"Chord: {new_chord} (velocity: {velocity})")

    def _apply_chord_expression(self, hip_tilt: float, knee_angle: float):
        pitch_bend = int(8192 + hip_tilt * 4096)
        pitch_bend = max(0, min(16383, pitch_bend))
        for ch in [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH]:
            self._pitch_bend(pitch_bend, ch)

        pressure = int(knee_angle * 127)
        for ch in [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH]:
            self._channel_pressure(pressure, ch)

    # ------------------------------------------------------------------
    # Melody
    # ------------------------------------------------------------------

    def _update_melody(self, features: dict):
        current_time = time.time()
        self._update_melody_hand(
            features, side="right",
            hand_y=features.get("rightHandY", 0.5),
            jerk=features.get("rightHandJerk", 0.0),
            arm_velocity=features.get("rightArmVelocity", 0.0),
            elbow_angle=features.get("rightElbowHipAngle", 0.0),
            base_note=self.MELODY_RIGHT_BASE,
            channel=self.CH_MELODY_RIGHT,
            current_time=current_time,
        )
        self._update_melody_hand(
            features, side="left",
            hand_y=features.get("leftHandY", 0.5),
            jerk=features.get("leftHandJerk", 0.0),
            arm_velocity=features.get("leftArmVelocity", 0.0),
            elbow_angle=features.get("leftElbowHipAngle", 0.0),
            base_note=self.MELODY_LEFT_BASE,
            channel=self.CH_MELODY_LEFT,
            current_time=current_time,
        )

    def _update_melody_hand(self, features, side, hand_y, jerk,
                            arm_velocity, elbow_angle, base_note,
                            channel, current_time):
        if side == "right":
            current_note = self.melody_right_note
            note_start_time = self.melody_right_note_time
        else:
            current_note = self.melody_left_note
            note_start_time = self.melody_left_note_time

        target_note = self._hand_y_to_note(hand_y, base_note)

        if jerk > self.JERK_THRESHOLD:
            velocity = int(60 + arm_velocity * 67)
            velocity = max(1, min(127, velocity))
            duration = self.base_note_duration - (arm_velocity * 0.15)
            duration = max(self.min_note_duration, min(self.max_note_duration, duration))

            if current_note is not None:
                self._note_off(current_note, channel)

            self._note_on(target_note, velocity, channel)

            if side == "right":
                self.melody_right_note = target_note
                self.melody_right_note_time = current_time
            else:
                self.melody_left_note = target_note
                self.melody_left_note_time = current_time

            print(f"Melody {side}: note {target_note} (vel: {velocity}, dur: {duration:.2f}s)")

        elif current_note is not None:
            if current_time - note_start_time > self.base_note_duration:
                self._note_off(current_note, channel)
                if side == "right":
                    self.melody_right_note = None
                else:
                    self.melody_left_note = None

        if current_note is not None:
            pitch_bend = int(8192 + elbow_angle * 2048)
            pitch_bend = max(0, min(16383, pitch_bend))
            self._pitch_bend(pitch_bend, channel)

    def _hand_y_to_note(self, hand_y: float, base_note: int) -> int:
        scale_index = int(hand_y * 7.99)
        scale_index = max(0, min(7, scale_index))
        if scale_index == 7:
            return base_note + 12
        return base_note + self.SCALE_NOTES[scale_index]

    # ------------------------------------------------------------------
    # Global expression
    # ------------------------------------------------------------------

    def _update_global_expression(self, features: dict):
        head_tilt = features.get("headTilt", 0.0)
        cc_value = int(64 + head_tilt * 63)
        cc_value = max(0, min(127, cc_value))
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
        for ch in [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH,
                   self.CH_MELODY_RIGHT, self.CH_MELODY_LEFT]:
            self._control_change(123, 0, ch)
