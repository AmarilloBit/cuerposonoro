"""
Blueprint MIDI sender for CuerpoSonoro.

Loads chord progressions from the Unison MIDI Blueprint library and maps
body movement to melody, bass, and expression. All harmony comes from parsed
MIDI files (no hardcoded chord tones).

MPE Channel allocation (0-indexed):
  - Channel 0:  Master (global CC1 modulation, global pitch bend)
  - Channel 5:  Melody (per-note pitch bend)
  - Channel 6:  Bass (chord root, monophonic)
"""

import logging
import math
import time

import mido

from vision_processor.midi.base import BaseMidiSender
from vision_processor.midi.blueprint_loader import BlueprintLibrary

logger = logging.getLogger(__name__)


class BlueprintMidiSender(BaseMidiSender):
    """
    Body-driven MIDI sender using Unison MIDI Blueprint progressions.

    Chord navigation is driven by pelvis_thrust (forward/backward hip
    displacement). Melody notes come from the active chord's parsed notes.
    Bass plays the chord root. Spine lean controls CC1 modulation.
    Head tilt controls global pitch bend.
    """

    def __init__(
        self,
        config: dict,
        port_name: str = "CuerpoSonoro",
        genre: str | None = None,
        key: str | None = None,
    ):
        self._config = config
        self._port_name = port_name
        self._genre_fixed = genre is not None
        self._key_fixed = key is not None
        self._requested_genre = genre
        self._requested_key = key

        # Load the MIDI library
        assets_path = config.get("assets_path", "assets/midi")
        self._library = BlueprintLibrary(assets_path)

        # Validate genre/key before opening MIDI port
        if genre is not None and genre not in self._library.genres:
            available = ", ".join(self._library.genres)
            raise ValueError(
                f"Genre '{genre}' not found. Available: {available}"
            )
        if key is not None and genre is not None:
            # Validate by attempting to get a progression
            self._library.random_progression(genre=genre, key=key)

        # Pick initial progression
        self._progression = self._library.random_progression(
            genre=genre, key=key
        )
        self._chord_index = 0
        self._last_chord_change = 0.0

        # Melody state
        self._melody_note: int | None = None
        self._prev_right_hand_y: float | None = None
        self._last_melody_trigger: float = 0.0

        # Bass state
        self._bass_note: int | None = None

        # Silence detection
        self._silence_start: float | None = None
        self._all_notes_sent_off = False

        # MIDI port
        self._port = None
        self._open_port()

        logger.info(
            "[BlueprintMidiSender] Genre: %s, Key: %s, File: %s, Chords: %d",
            self._progression.genre,
            self._progression.key,
            self._progression.filename,
            len(self._progression.chords),
        )

    def _open_port(self):
        try:
            self._port = mido.open_output(self._port_name, virtual=True)
            print(
                f"[BlueprintMidiSender] Port: {self._port_name} | "
                f"Genre: {self._progression.genre} | "
                f"Key: {self._progression.key}"
            )
        except Exception as e:
            print(f"[BlueprintMidiSender] Error opening MIDI port: {e}")

    # ------------------------------------------------------------------
    # Public interface (BaseMidiSender)
    # ------------------------------------------------------------------

    def update(self, features: dict):
        if not self._port:
            return

        self._update_silence(features)
        self._update_chord_navigation(features)
        self._update_melody(features)
        self._update_bass(features)
        self._update_spine_modulation(features)
        self._update_head_tilt(features)

    def close(self):
        if self._port:
            self._all_notes_off()
            self._port.close()
            print("[BlueprintMidiSender] Port closed.")

    # ------------------------------------------------------------------
    # Chord navigation (pelvis_thrust)
    # ------------------------------------------------------------------

    def _update_chord_navigation(self, features: dict):
        thrust = features.get("pelvis_thrust", 0.0)
        threshold = self._config.get("pelvis_threshold", 0.12)
        cooldown = self._config.get("chord_change_cooldown", 1.2)

        now = time.time()
        if (now - self._last_chord_change) < cooldown:
            return

        if abs(thrust) < threshold:
            return

        num_chords = len(self._progression.chords)
        if num_chords == 0:
            return

        if thrust > threshold:
            self._chord_index = (self._chord_index + 1) % num_chords
        elif thrust < -threshold:
            self._chord_index = (self._chord_index - 1) % num_chords

        self._last_chord_change = now
        chord = self._progression.chords[self._chord_index]
        logger.info(
            "[chord] Index %d/%d, root=%d, notes=%s",
            self._chord_index, num_chords, chord.root, chord.notes,
        )

    # ------------------------------------------------------------------
    # Melody (MPE channel 5)
    # ------------------------------------------------------------------

    def _update_melody(self, features: dict):
        right_vel = features.get("rightArmVelocity", 0.0)
        left_vel = features.get("leftArmVelocity", 0.0)
        arm_velocity = max(right_vel, left_vel)

        arm_threshold = self._config.get("arm_velocity_threshold", 0.08)
        melody_cooldown = self._config.get("melody_cooldown", 0.15)
        melody_ch = self._config.get("midi_channel_melody", 5)

        if arm_velocity <= arm_threshold:
            return

        now = time.time()
        if (now - self._last_melody_trigger) < melody_cooldown:
            return
        self._last_melody_trigger = now

        # Note selection from active chord
        chord = self._progression.chords[self._chord_index]
        if not chord.notes:
            return

        right_y = features.get("rightHandY", 0.5)
        left_y = features.get("leftHandY", 0.5)
        mean_wrist_height = (right_y + left_y) / 2

        n = len(chord.notes)
        note_index = int(mean_wrist_height * n)
        note_index = max(0, min(n - 1, note_index))
        target_note = chord.notes[note_index]

        # Attack velocity
        velocity = int(40 + arm_velocity * 87)
        velocity = max(1, min(127, velocity))

        # Per-note pitch bend (right wrist vertical delta)
        bend_semitones = 0.0
        if self._prev_right_hand_y is not None:
            dy = right_y - self._prev_right_hand_y
            bend_scale = self._config.get("melody_bend_scale", 4.0)
            bend_semitones = -dy * bend_scale
        self._prev_right_hand_y = right_y

        bend_value = int(bend_semitones / 2 * 8192)
        bend_value = max(-8192, min(8191, bend_value))

        # Send pitch bend before note_on
        self._port.send(mido.Message(
            "pitchwheel", pitch=bend_value, channel=melody_ch
        ))

        # Note off for previous melody note
        if self._melody_note is not None:
            self._port.send(mido.Message(
                "note_off", note=self._melody_note, velocity=0, channel=melody_ch
            ))

        # Note on
        self._port.send(mido.Message(
            "note_on", note=target_note, velocity=velocity, channel=melody_ch
        ))
        self._melody_note = target_note

    # ------------------------------------------------------------------
    # Bass (MPE channel 6)
    # ------------------------------------------------------------------

    def _update_bass(self, features: dict):
        bass_ch = self._config.get("midi_channel_bass", 6)
        chord = self._progression.chords[self._chord_index]

        # Bass always plays chord root, triggered by ankle velocity
        left_ankle_vel = self._calculate_landmark_velocity(features, "leftArmVelocity")
        right_ankle_vel = self._calculate_landmark_velocity(features, "rightArmVelocity")
        ankle_vel = (left_ankle_vel + right_ankle_vel) / 2

        target_root = chord.root

        if target_root != self._bass_note:
            # Note off previous bass
            if self._bass_note is not None:
                self._port.send(mido.Message(
                    "note_off", note=self._bass_note, velocity=0, channel=bass_ch
                ))

            velocity = int(40 + ankle_vel * 87)
            velocity = max(1, min(127, velocity))

            self._port.send(mido.Message(
                "note_on", note=target_root, velocity=velocity, channel=bass_ch
            ))
            self._bass_note = target_root

    def _calculate_landmark_velocity(self, features: dict, key: str) -> float:
        return features.get(key, 0.0)

    # ------------------------------------------------------------------
    # Spine modulation (CC1 on master channel)
    # ------------------------------------------------------------------

    def _update_spine_modulation(self, features: dict):
        spine_lean = features.get("spine_lean", 0.0)
        master_ch = self._config.get("midi_channel_master", 0)

        # Forward lean (positive spine_lean) = high CC1
        cc1_value = int(abs(spine_lean) * 127)
        cc1_value = max(0, min(127, cc1_value))

        self._port.send(mido.Message(
            "control_change", control=1, value=cc1_value, channel=master_ch
        ))

    # ------------------------------------------------------------------
    # Head tilt (global pitch bend on master channel)
    # ------------------------------------------------------------------

    def _update_head_tilt(self, features: dict):
        head_tilt = features.get("headTilt", 0.0)
        master_ch = self._config.get("midi_channel_master", 0)

        # Tilt right = positive pitch bend (+50 cents)
        # Tilt left = negative pitch bend (-50 cents)
        # Neutral (< 5 degrees, roughly < 0.09 normalised) = 0
        if abs(head_tilt) < 0.09:
            bend_value = 0
        else:
            # 50 cents = 0.5 semitones, pitch bend range is +/- 2 semitones
            bend_semitones = head_tilt * 0.5
            bend_value = int(bend_semitones / 2 * 8192)
            bend_value = max(-8192, min(8191, bend_value))

        self._port.send(mido.Message(
            "pitchwheel", pitch=bend_value, channel=master_ch
        ))

    # ------------------------------------------------------------------
    # Silence detection
    # ------------------------------------------------------------------

    def _update_silence(self, features: dict):
        silence_threshold = self._config.get("silence_threshold", 0.02)
        silence_hold_ms = self._config.get("silence_hold_ms", 500)
        silence_timeout = self._config.get("silence_timeout", 60)

        # Compute mean velocity of all tracked landmarks
        velocity_keys = [
            "rightArmVelocity", "leftArmVelocity",
            "rightHandJerk", "leftHandJerk",
            "energy",
        ]
        mean_velocity = sum(features.get(k, 0.0) for k in velocity_keys) / len(velocity_keys)

        now = time.time()

        if mean_velocity < silence_threshold:
            if self._silence_start is None:
                self._silence_start = now

            elapsed_ms = (now - self._silence_start) * 1000

            # All notes off after silence_hold_ms
            if elapsed_ms >= silence_hold_ms and not self._all_notes_sent_off:
                self._all_notes_off()
                self._all_notes_sent_off = True

            # Genre rotation after silence_timeout (only if genre is random)
            elapsed_s = now - self._silence_start
            if elapsed_s >= silence_timeout and not self._genre_fixed:
                self._rotate_progression()

        else:
            if self._silence_start is not None:
                self._silence_start = None
                self._all_notes_sent_off = False

    def _rotate_progression(self):
        """Pick a new random genre, key, and progression."""
        self._progression = self._library.random_progression(
            genre=self._requested_genre,
            key=self._requested_key,
        )
        self._chord_index = 0
        self._silence_start = None
        self._all_notes_sent_off = False
        logger.info(
            "[rotate] New progression: Genre=%s, Key=%s, File=%s",
            self._progression.genre,
            self._progression.key,
            self._progression.filename,
        )

    # ------------------------------------------------------------------
    # Low-level MIDI helpers
    # ------------------------------------------------------------------

    def _all_notes_off(self):
        """Send CC 123 (all notes off) on all used channels."""
        channels = [
            self._config.get("midi_channel_master", 0),
            self._config.get("midi_channel_melody", 5),
            self._config.get("midi_channel_bass", 6),
        ]
        for ch in channels:
            self._port.send(mido.Message(
                "control_change", control=123, value=0, channel=ch
            ))
        self._melody_note = None
        self._bass_note = None

    # ------------------------------------------------------------------
    # Debug info
    # ------------------------------------------------------------------

    def debug_info(self) -> dict:
        """Return current state for debug overlay."""
        return {
            "genre": self._progression.genre,
            "key": self._progression.key,
            "filename": self._progression.filename,
            "chord_index": self._chord_index,
            "total_chords": len(self._progression.chords),
            "genre_rotation_active": not self._genre_fixed,
        }
