#!/usr/bin/env python3
"""
debug_tools/analyze_video.py — Análisis headless de vídeo para debug de features y MIDI.

Procesa un vídeo sin abrir ventana gráfica y genera:
  - features_frame_by_frame.csv   → todas las features en cada fotograma
  - triggers.csv                  → momentos en que se cruzan umbrales
  - midi_events.csv               → notas MIDI que se habrían enviado
  - summary.json                  → resumen estadístico completo

Uso:
    python debug_tools/analyze_video.py --source /ruta/al/video.mov --midi-mode classic
    python debug_tools/analyze_video.py --source /ruta/al/video.mov --midi-mode musical
    python debug_tools/analyze_video.py --source /ruta/al/video.mov --midi-mode classic --out debug_output/
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

# Permitir imports desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_processor.config import Config
from vision_processor.features import FeatureExtractor


# =============================================================================
# MIDI sender instrumentado — intercepta mensajes sin enviarlos a ningún puerto
# =============================================================================

class InstrumentedClassicSender:
    """
    Replica la lógica de ClassicMidiSender pero en lugar de enviar MIDI
    registra cada evento en una lista consultable.
    """

    SCALE_NOTES = [0, 2, 4, 5, 7, 9, 11]
    CHORDS = {
        "I":  (48, 52, 55),
        "IV": (53, 57, 60),
        "V":  (55, 59, 62),
        "VI": (57, 60, 64),
    }
    CHORD_ZONES = [
        (0.00, 0.25, "I"),
        (0.25, 0.50, "IV"),
        (0.50, 0.75, "V"),
        (0.75, 1.00, "VI"),
    ]
    CH_MASTER       = 0
    CH_CHORD_ROOT   = 1
    CH_CHORD_THIRD  = 2
    CH_CHORD_FIFTH  = 3
    CH_MELODY_RIGHT = 4
    CH_MELODY_LEFT  = 5
    MELODY_RIGHT_BASE = 48
    MELODY_LEFT_BASE  = 72
    JERK_THRESHOLD    = 0.4
    HIP_TILT_THRESHOLD = 0.6

    def __init__(self):
        self.events = []          # lista de dicts con todos los eventos MIDI
        self.current_chord = None
        self.current_chord_notes = []
        self.melody_right_note = None
        self.melody_left_note  = None
        self.melody_right_note_time = 0
        self.melody_left_note_time  = 0
        self.base_note_duration = 0.3
        self.min_note_duration  = 0.15
        self.max_note_duration  = 0.6
        self._frame = 0
        self._time  = 0.0

    def update(self, features: dict, frame: int, timestamp: float):
        self._frame = frame
        self._time  = timestamp
        self._update_chords(features)
        self._update_melody(features)

    def _log(self, event_type: str, **kwargs):
        self.events.append({
            "frame":      self._frame,
            "time_s":     round(self._time, 4),
            "event_type": event_type,
            **kwargs,
        })

    def _get_chord_from_position(self, x: float) -> str:
        for min_x, max_x, chord in self.CHORD_ZONES:
            if min_x <= x < max_x:
                return chord
        return "I"

    def _update_chords(self, features: dict):
        feet_x     = features.get("feetCenterX", 0.5)
        knee_angle = features.get("kneeAngle", 1.0)
        new_chord  = self._get_chord_from_position(feet_x)

        if new_chord != self.current_chord:
            velocity = int(40 + knee_angle * 87)
            velocity = max(1, min(127, velocity))
            self._log("chord_change",
                      chord=new_chord,
                      velocity=velocity,
                      feet_x=round(feet_x, 4))
            self.current_chord_notes = list(self.CHORDS[new_chord])
            self.current_chord = new_chord

    def _hand_y_to_note(self, hand_y: float, base_note: int) -> int:
        scale_index = int(hand_y * 7.99)
        scale_index = max(0, min(7, scale_index))
        if scale_index == 7:
            return base_note + 12
        return base_note + self.SCALE_NOTES[scale_index]

    def _update_melody(self, features: dict):
        current_time = self._time
        for side, hand_y_key, jerk_key, vel_key, base_note, ch in [
            ("right", "rightHandY", "rightHandJerk", "rightArmVelocity",
             self.MELODY_RIGHT_BASE, self.CH_MELODY_RIGHT),
            ("left",  "leftHandY",  "leftHandJerk",  "leftArmVelocity",
             self.MELODY_LEFT_BASE,  self.CH_MELODY_LEFT),
        ]:
            hand_y      = features.get(hand_y_key, 0.5)
            jerk        = features.get(jerk_key, 0.0)
            arm_velocity = features.get(vel_key, 0.0)

            current_note    = self.melody_right_note if side == "right" else self.melody_left_note
            note_start_time = self.melody_right_note_time if side == "right" else self.melody_left_note_time
            target_note     = self._hand_y_to_note(hand_y, base_note)

            if jerk > self.JERK_THRESHOLD:
                velocity = int(60 + arm_velocity * 67)
                velocity = max(1, min(127, velocity))
                duration = self.base_note_duration - (arm_velocity * 0.15)
                duration = max(self.min_note_duration, min(self.max_note_duration, duration))

                self._log("note_on",
                          side=side,
                          note=target_note,
                          velocity=velocity,
                          duration_s=round(duration, 3),
                          jerk=round(jerk, 4),
                          hand_y=round(hand_y, 4))

                if side == "right":
                    self.melody_right_note = target_note
                    self.melody_right_note_time = current_time
                else:
                    self.melody_left_note = target_note
                    self.melody_left_note_time = current_time

            elif current_note is not None:
                if current_time - note_start_time > self.base_note_duration:
                    self._log("note_off",
                              side=side,
                              note=current_note)
                    if side == "right":
                        self.melody_right_note = None
                    else:
                        self.melody_left_note = None


class InstrumentedMusicalSender:
    """
    Replica la lógica de MusicalMidiSender (sin hilo de tempo) para análisis.
    Las notas se 'disparan' en el mismo frame en que se encolarían, ya que
    no hay hilo de tempo real en el análisis offline.
    """

    CHORD_TONES = {
        "I":  [60, 64, 67, 71, 74],
        "IV": [65, 69, 72, 76, 79],
        "V":  [67, 71, 74, 77, 81],
        "VI": [69, 72, 76, 79, 83],
    }
    CHORDS = {
        "I":  (48, 52, 55),
        "IV": (53, 57, 60),
        "V":  (55, 59, 62),
        "VI": (57, 60, 64),
    }
    CHORD_ZONES = [
        (0.00, 0.25, "I"),
        (0.25, 0.50, "IV"),
        (0.50, 0.75, "V"),
        (0.75, 1.00, "VI"),
    ]

    def __init__(self,
                 direction_threshold: float = 0.03,
                 velocity_threshold: float  = 0.4,
                 jump_size_slow: int = 1,
                 jump_size_fast: int = 2):
        self.direction_threshold = direction_threshold
        self.velocity_threshold  = velocity_threshold
        self.jump_size_slow      = jump_size_slow
        self.jump_size_fast      = jump_size_fast

        self.events        = []
        self.current_chord = None
        self._melody_index = 2
        self._prev_hand_y: Optional[float] = None
        self._frame = 0
        self._time  = 0.0

    def update(self, features: dict, frame: int, timestamp: float):
        self._frame = frame
        self._time  = timestamp
        self._update_chords(features)
        self._update_melody_direction(features)

    def _log(self, event_type: str, **kwargs):
        self.events.append({
            "frame":      self._frame,
            "time_s":     round(self._time, 4),
            "event_type": event_type,
            **kwargs,
        })

    def _get_chord_from_position(self, x: float) -> str:
        for min_x, max_x, chord in self.CHORD_ZONES:
            if min_x <= x < max_x:
                return chord
        return "I"

    def _update_chords(self, features: dict):
        feet_x     = features.get("feetCenterX", 0.5)
        knee_angle = features.get("kneeAngle", 1.0)
        new_chord  = self._get_chord_from_position(feet_x)

        if new_chord != self.current_chord:
            velocity = int(40 + knee_angle * 87)
            velocity = max(1, min(127, velocity))
            self._log("chord_change",
                      chord=new_chord,
                      velocity=velocity,
                      feet_x=round(feet_x, 4))
            self.current_chord = new_chord
            self._melody_index = 2   # reset al cambiar acorde

    def _update_melody_direction(self, features: dict):
        hand_y       = features.get("rightHandY", 0.5)
        arm_velocity = features.get("rightArmVelocity", 0.0)

        if self._prev_hand_y is None:
            self._prev_hand_y = hand_y
            return

        dy = hand_y - self._prev_hand_y
        self._prev_hand_y = hand_y

        jump = self.jump_size_fast if arm_velocity > self.velocity_threshold else self.jump_size_slow

        if dy > self.direction_threshold:
            direction = +1
        elif dy < -self.direction_threshold:
            direction = -1
        else:
            return   # sin movimiento suficiente, sin nota

        chord_key = self.current_chord or "I"
        tones     = self.CHORD_TONES[chord_key]
        new_index = max(0, min(len(tones) - 1, self._melody_index + direction * jump))
        self._melody_index = new_index

        target_note = tones[new_index]
        velocity    = int(60 + arm_velocity * 67)
        velocity    = max(1, min(127, velocity))

        self._log("note_candidate",
                  note=target_note,
                  velocity=velocity,
                  dy=round(dy, 5),
                  direction=direction,
                  arm_velocity=round(arm_velocity, 4),
                  hand_y=round(hand_y, 4))


# =============================================================================
# ANÁLISIS PRINCIPAL
# =============================================================================

FEATURE_KEYS = [
    "energy", "symmetry", "smoothness", "armAngle", "verticalExtension",
    "feetCenterX", "hipTilt", "kneeAngle",
    "rightHandY", "leftHandY",
    "rightHandJerk", "leftHandJerk",
    "rightArmVelocity", "leftArmVelocity",
    "rightElbowHipAngle", "leftElbowHipAngle",
    "headTilt",
]

JERK_THRESHOLD      = 0.4
DIRECTION_THRESHOLD = 0.03
VELOCITY_THRESHOLD  = 0.4


def analyze(source: str, midi_mode: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Pipeline
    config = Config(overrides={
        "output.mode":      "midi",
        "output.midi_mode": midi_mode,
    })
    pose_estimator    = config.create_pose_estimator()
    feature_extractor = FeatureExtractor()

    if midi_mode == "musical":
        sender = InstrumentedMusicalSender(
            direction_threshold=config.musical_direction_threshold,
            velocity_threshold=config.musical_velocity_threshold,
            jump_size_slow=config.musical_jump_size_slow,
            jump_size_fast=config.musical_jump_size_fast,
        )
    else:
        sender = InstrumentedClassicSender()
        sender.JERK_THRESHOLD = config.midi_jerk_threshold

    # Abrir vídeo (una sola pasada, sin loop)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir el vídeo: {source}")
        sys.exit(1)

    fps_video  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[analyze] Vídeo: {source}")
    print(f"[analyze] FPS: {fps_video:.1f}  |  Fotogramas totales: {total_frames}")
    print(f"[analyze] Modo MIDI: {midi_mode}")
    print(f"[analyze] Procesando...")

    all_features  = []   # una fila por fotograma
    all_triggers  = []   # solo fotogramas con eventos relevantes
    prev_landmarks = None
    frame_idx      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps_video

        # Estimación de pose
        results   = pose_estimator.estimate(frame)
        landmarks = pose_estimator.get_landmarks(results)

        pose_detected = landmarks is not None

        if pose_detected:
            features = feature_extractor.calculate(landmarks, prev_landmarks)
            prev_landmarks = landmarks
        else:
            features = feature_extractor._empty_features()

        # --- Registrar features frame a frame ---
        row = {
            "frame":         frame_idx,
            "time_s":        round(timestamp, 4),
            "pose_detected": pose_detected,
        }
        for k in FEATURE_KEYS:
            row[k] = round(features.get(k, 0.0), 5)
        all_features.append(row)

        # --- Detectar triggers manualmente para triggers.csv ---
        trigger_notes = []
        if features.get("rightHandJerk", 0) > JERK_THRESHOLD:
            trigger_notes.append({"side": "right",
                                   "jerk": round(features["rightHandJerk"], 4),
                                   "threshold": JERK_THRESHOLD})
        if features.get("leftHandJerk", 0) > JERK_THRESHOLD:
            trigger_notes.append({"side": "left",
                                   "jerk": round(features["leftHandJerk"], 4),
                                   "threshold": JERK_THRESHOLD})

        for t in trigger_notes:
            all_triggers.append({
                "frame":         frame_idx,
                "time_s":        round(timestamp, 4),
                "trigger_type":  "jerk",
                **t,
                "rightHandY":    round(features.get("rightHandY", 0), 4),
                "leftHandY":     round(features.get("leftHandY", 0), 4),
                "feetCenterX":   round(features.get("feetCenterX", 0), 4),
            })

        # Triggers de dirección para musical
        if midi_mode == "musical" and sender._prev_hand_y is not None:
            dy = features.get("rightHandY", 0.5) - sender._prev_hand_y
            if abs(dy) > DIRECTION_THRESHOLD:
                all_triggers.append({
                    "frame":        frame_idx,
                    "time_s":       round(timestamp, 4),
                    "trigger_type": "direction",
                    "side":         "right",
                    "dy":           round(dy, 5),
                    "threshold":    DIRECTION_THRESHOLD,
                    "rightHandY":   round(features.get("rightHandY", 0), 4),
                    "arm_velocity": round(features.get("rightArmVelocity", 0), 4),
                })

        # --- Sender instrumentado ---
        sender.update(features, frame=frame_idx, timestamp=timestamp)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total_frames} fotogramas procesados...", end="\r")

    cap.release()
    pose_estimator.release()
    print(f"\n[analyze] {frame_idx} fotogramas procesados.")

    # ==========================================================================
    # Guardar CSVs
    # ==========================================================================

    # 1. features_frame_by_frame.csv
    features_path = os.path.join(output_dir, "features_frame_by_frame.csv")
    with open(features_path, "w", newline="") as f:
        if all_features:
            writer = csv.DictWriter(f, fieldnames=list(all_features[0].keys()))
            writer.writeheader()
            writer.writerows(all_features)
    print(f"[analyze] Guardado: {features_path}  ({len(all_features)} filas)")

    # 2. triggers.csv
    triggers_path = os.path.join(output_dir, "triggers.csv")
    trigger_fields = ["frame", "time_s", "trigger_type", "side",
                      "jerk", "dy", "threshold",
                      "rightHandY", "leftHandY", "feetCenterX", "arm_velocity"]
    with open(triggers_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trigger_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_triggers)
    print(f"[analyze] Guardado: {triggers_path}  ({len(all_triggers)} triggers)")

    # 3. midi_events.csv
    midi_path = os.path.join(output_dir, "midi_events.csv")
    midi_fields = ["frame", "time_s", "event_type",
                   "chord", "note", "velocity", "side",
                   "duration_s", "jerk", "dy", "direction",
                   "hand_y", "arm_velocity", "feet_x"]
    with open(midi_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=midi_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sender.events)
    print(f"[analyze] Guardado: {midi_path}  ({len(sender.events)} eventos MIDI)")

    # ==========================================================================
    # summary.json
    # ==========================================================================

    def safe_mean(vals):
        return round(sum(vals) / len(vals), 5) if vals else 0.0

    def safe_max(vals):
        return round(max(vals), 5) if vals else 0.0

    feature_stats = {}
    for k in FEATURE_KEYS:
        vals = [r[k] for r in all_features if r["pose_detected"]]
        feature_stats[k] = {
            "mean": safe_mean(vals),
            "max":  safe_max(vals),
            "min":  round(min(vals), 5) if vals else 0.0,
        }

    chord_counts = {}
    note_counts  = {}
    for ev in sender.events:
        if ev["event_type"] == "chord_change":
            chord_counts[ev.get("chord", "?")] = chord_counts.get(ev.get("chord", "?"), 0) + 1
        if ev["event_type"] in ("note_on", "note_candidate"):
            note_counts[str(ev.get("note", "?"))] = note_counts.get(str(ev.get("note", "?")), 0) + 1

    summary = {
        "source":           source,
        "midi_mode":        midi_mode,
        "total_frames":     frame_idx,
        "fps_video":        fps_video,
        "duration_s":       round(frame_idx / fps_video, 2),
        "poses_detected":   sum(1 for r in all_features if r["pose_detected"]),
        "poses_missed":     sum(1 for r in all_features if not r["pose_detected"]),
        "thresholds": {
            "jerk":      JERK_THRESHOLD,
            "direction": DIRECTION_THRESHOLD,
            "velocity":  VELOCITY_THRESHOLD,
        },
        "trigger_count":    len(all_triggers),
        "midi_event_count": len(sender.events),
        "chord_changes":    chord_counts,
        "notes_fired":      note_counts,
        "feature_stats":    feature_stats,
        "output_files": {
            "features": features_path,
            "triggers": triggers_path,
            "midi":     midi_path,
        },
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[analyze] Guardado: {summary_path}")

    # Resumen en consola
    print("\n" + "=" * 60)
    print(f"  RESUMEN — {midi_mode.upper()}")
    print("=" * 60)
    print(f"  Fotogramas procesados : {frame_idx}")
    print(f"  Pose detectada        : {summary['poses_detected']} / {frame_idx}")
    print(f"  Triggers detectados   : {len(all_triggers)}")
    print(f"  Eventos MIDI          : {len(sender.events)}")
    print(f"  Cambios de acorde     : {sum(chord_counts.values())}")
    print(f"  Notas disparadas      : {sum(note_counts.values())}")
    print(f"\n  Jerk máximo derecha   : {feature_stats['rightHandJerk']['max']:.4f}  (umbral: {JERK_THRESHOLD})")
    print(f"  Jerk máximo izquierda : {feature_stats['leftHandJerk']['max']:.4f}  (umbral: {JERK_THRESHOLD})")
    print(f"  Velocidad brazo (máx) : {feature_stats['rightArmVelocity']['max']:.4f}")
    print("=" * 60)
    print(f"\n  Archivos en: {output_dir}/\n")

    return summary


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Analiza un vídeo headless y vuelca features + eventos MIDI a CSV/JSON."
    )
    parser.add_argument("--source", required=True,
                        help="Ruta al vídeo (e.g. /Users/mara/CuerpoSonoro/video.mov)")
    parser.add_argument("--midi-mode", dest="midi_mode",
                        choices=["classic", "musical"], default="classic",
                        help="Modo MIDI a analizar (default: classic)")
    parser.add_argument("--out", default=None,
                        help="Carpeta de salida (default: debug_output/<nombre_video>/)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.out:
        output_dir = args.out
    else:
        video_stem = Path(args.source).stem
        output_dir = os.path.join("debug_output", video_stem)

    analyze(
        source=args.source,
        midi_mode=args.midi_mode,
        output_dir=output_dir,
    )
