"""
Centralized configuration loader for CuerpoSonoro.

Reads config.yaml and provides typed access with dot-notation overrides.
Designed so the benchmark script (and any other entry point) can easily
swap configuration values without touching multiple files.

Usage:
    # Load defaults from config.yaml
    config = Config()

    # Access values
    config.get("camera.device_id")        # → 0
    config.get("pose.model_complexity")    # → 1
    config.get("osc.send_mode")           # → "individual"

    # Create a copy with overrides (original unchanged)
    fast_config = config.with_overrides(
        camera__width=640,
        camera__height=480,
        pose__model_complexity=0,
    )

    # Build pipeline components from config
    cap = config.create_camera()
    pose = config.create_pose_estimator()
    sender = config.create_sender()
"""

import os
import copy
import yaml
from typing import Any, Optional


# Default config path (relative to project root)
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config.yaml"
)


class Config:
    """
    Pipeline configuration with override support.

    Wraps the YAML config dict and provides:
    - Dot-notation get/set: config.get("camera.width")
    - Override copies: config.with_overrides(camera__width=640)
    - Factory methods for creating pipeline components
    """

    def __init__(self, path: Optional[str] = None, overrides: Optional[dict] = None):
        """
        Args:
            path: Path to config.yaml. None uses the default project path.
            overrides: Dict of dot-notation overrides, e.g.:
                       {"camera.device_id": 1, "pose.model_complexity": 0}
        """
        config_path = path or _DEFAULT_CONFIG_PATH

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self._data = yaml.safe_load(f) or {}
        else:
            self._data = {}

        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                self.set(key, value)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """
        Get a value using dot notation.

        Args:
            dotted_key: e.g. "camera.width", "pose.model_complexity"
            default: Value to return if key not found.

        Returns:
            The config value or default.
        """
        keys = dotted_key.split(".")
        node = self._data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def set(self, dotted_key: str, value: Any):
        """
        Set a value using dot notation, creating intermediate dicts if needed.

        Args:
            dotted_key: e.g. "camera.width"
            value: Value to set.
        """
        keys = dotted_key.split(".")
        node = self._data
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

    def with_overrides(self, **kwargs) -> "Config":
        """
        Return a new Config with the given overrides applied.
        Uses double-underscore as separator (for kwarg compatibility).

        Example:
            new_config = config.with_overrides(
                camera__device_id=1,
                pose__model_complexity=0,
                camera__width=640,
            )

        Returns:
            A new Config instance with overrides applied.
        """
        new = Config.__new__(Config)
        new._data = copy.deepcopy(self._data)

        for key, value in kwargs.items():
            dotted = key.replace("__", ".")
            new.set(dotted, value)

        return new

    # =========================================================================
    # Shortcut properties for common values
    # =========================================================================

    # Camera
    @property
    def camera_device_id(self) -> int:
        return self.get("camera.device_id", 0)

    @property
    def camera_width(self) -> int:
        return self.get("camera.width", 1280)

    @property
    def camera_height(self) -> int:
        return self.get("camera.height", 720)

    @property
    def camera_fps(self) -> int:
        return self.get("camera.fps", 30)

    @property
    def camera_buffer_size(self) -> int:
        return self.get("camera.buffer_size", 1)

    # Pose
    @property
    def pose_model_complexity(self) -> int:
        return self.get("pose.model_complexity", 1)

    @property
    def pose_min_detection_confidence(self) -> float:
        return self.get("pose.min_detection_confidence", 0.5)

    @property
    def pose_min_tracking_confidence(self) -> float:
        return self.get("pose.min_tracking_confidence", 0.5)

    # Features
    @property
    def features_smoothing_factor(self) -> float:
        return self.get("features.smoothing_factor", 0.3)

    # Output
    @property
    def output_mode(self) -> str:
        return self.get("output.mode", "osc")

    # OSC
    @property
    def osc_host(self) -> str:
        return self.get("osc.host", "127.0.0.1")

    @property
    def osc_port(self) -> int:
        return self.get("osc.port", 57120)

    @property
    def osc_send_mode(self) -> str:
        return self.get("osc.send_mode", "individual")

    # MIDI
    @property
    def midi_port_name(self) -> str:
        return self.get("midi.port_name", "CuerpoSonoro")

    @property
    def midi_jerk_threshold(self) -> float:
        return self.get("midi.jerk_threshold", 0.4)

    @property
    def midi_mode(self) -> str:
        return self.get("output.midi_mode", "classic")

    # Musical sender settings
    @property
    def musical_tempo_bpm(self) -> int:
        return self.get("musical.tempo_bpm", 120)

    @property
    def musical_note_subdivision(self) -> int:
        return self.get("musical.note_subdivision", 8)

    @property
    def musical_direction_threshold(self) -> float:
        return self.get("musical.direction_threshold", 0.03)

    @property
    def musical_velocity_threshold(self) -> float:
        return self.get("musical.velocity_threshold", 0.4)

    @property
    def musical_jump_size_slow(self) -> int:
        return self.get("musical.jump_size_slow", 1)

    @property
    def musical_jump_size_fast(self) -> int:
        return self.get("musical.jump_size_fast", 2)

    # Camera profiles
    @property
    def camera_profiles(self) -> dict:
        return self.get("camera_profiles", {})

    # Benchmark settings
    @property
    def benchmark_frames(self) -> int:
        return self.get("benchmark.frames", 300)

    @property
    def benchmark_warmup(self) -> int:
        return self.get("benchmark.warmup", 30)

    @property
    def benchmark_resolutions(self) -> list:
        return self.get("benchmark.resolutions", [
            {"width": 640, "height": 480, "name": "480p"},
            {"width": 1280, "height": 720, "name": "720p"},
        ])

    @property
    def benchmark_pose_models(self) -> list:
        return self.get("benchmark.pose_models", [
            {"model_complexity": 0, "name": "lite"},
            {"model_complexity": 1, "name": "full"},
        ])

    @property
    def benchmark_output_modes(self) -> list:
        return self.get("benchmark.output_modes", [
            {"mode": "osc", "send_mode": "individual", "name": "osc"},
            {"mode": "osc", "send_mode": "bundle", "name": "osc_bundle"},
            {"mode": "midi", "send_mode": None, "name": "midi"},
        ])

    @property
    def benchmark_backends(self) -> list:
        return self.get("benchmark.backends", [{"name": "cpu"}])

    # =========================================================================
    # Factory methods — create pipeline components from config
    # =========================================================================

    def create_camera(self, source: str | None = None, loop: bool = True):
        """
        Create a camera from config.

        Args:
            source: Optional video file path (e.g. "tests/videos/test.mp4").
                    If None, opens the live webcam defined in config.yaml.
            loop: When `source` is a video file, restart from the beginning
                  on EOF if True; otherwise stop after one pass. Ignored for
                  webcam sources.

        Returns:
            WebcamCamera or VideoFileCamera instance.

        Raises:
            RuntimeError: If the camera or file cannot be opened.
        """
        from vision_processor.capture import WebcamCamera, VideoFileCamera

        if source is not None:
            return VideoFileCamera(path=source, loop=loop)

        return WebcamCamera(
            device_id=self.camera_device_id,
            width=self.camera_width,
            height=self.camera_height,
            fps=self.camera_fps,
            buffer_size=self.camera_buffer_size,
        )

    @staticmethod
    def _detect_backend(requested: str | None = None) -> str:
        """
        Detect the best available pose estimation backend.

        Priority (automatic):
            1. TensorRT  — NVIDIA Jetson (/etc/nv_tegra_release present + tensorrt importable)
            2. Metal     — Mac Apple Silicon (Darwin + arm64)
            3. CPU       — universal fallback

        Args:
            requested: If given ("tensorrt", "metal", "cpu"), skip detection
                       and return that value directly. Raises ValueError if
                       the requested backend is not available.

        Returns:
            One of: "tensorrt", "metal", "cpu"
        """
        import os
        import platform

        if requested:
            requested = requested.lower()
            if requested not in ("tensorrt", "metal", "cpu"):
                raise ValueError(
                    f"Unknown backend: '{requested}'. "
                    "Valid options: tensorrt, metal, cpu"
                )
            if requested == "tensorrt":
                try:
                    import tensorrt  # noqa: F401
                except ImportError:
                    raise RuntimeError(
                        "Backend 'tensorrt' requested but TensorRT is not available."
                    )
            if requested == "metal":
                if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
                    raise RuntimeError(
                        "Backend 'metal' requested but not running on Mac Apple Silicon."
                    )
            return requested

        # Automatic detection

        # 1. Jetson: nv_tegra_release exists and TensorRT is importable
        if os.path.exists("/etc/nv_tegra_release"):
            try:
                import tensorrt  # noqa: F401
                return "tensorrt"
            except ImportError:
                pass  # Jetson without TensorRT in environment, fall through to CPU

        # 2. Mac Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                from mediapipe.tasks.python.core.base_options import BaseOptions  # noqa: F401
                return "metal"
            except Exception:
                pass  # MediaPipe without Tasks API, fall through to CPU

        # 3. Universal CPU fallback
        return "cpu"

    def create_pose_estimator(self):
        """
        Create a PoseEstimator using the best available backend.

        Reads 'pose.backend' from config.yaml if present, otherwise
        auto-detects. Can also be overridden at runtime via:
            config = Config(overrides={"pose.backend": "cpu"})

        Returns:
            CPUPoseEstimator, MetalPoseEstimator, or TensorRTPoseEstimator.
            All expose the same interface (BasePoseEstimator).
        """
        requested = self.get("pose.backend", None)
        backend = self._detect_backend(requested)

        kwargs = dict(
            model_complexity=self.pose_model_complexity,
            min_detection_confidence=self.pose_min_detection_confidence,
            min_tracking_confidence=self.pose_min_tracking_confidence,
        )

        if backend == "tensorrt":
            from vision_processor.backends.tensorrt import TensorRTPoseEstimator
            return TensorRTPoseEstimator(**kwargs)

        elif backend == "metal":
            from vision_processor.backends.metal import MetalPoseEstimator
            return MetalPoseEstimator(**kwargs)

        else:
            from vision_processor.backends.cpu import CPUPoseEstimator
            return CPUPoseEstimator(**kwargs)

    def create_feature_extractor(self):
        """Create a FeatureExtractor from config."""
        from vision_processor.features import FeatureExtractor

        extractor = FeatureExtractor()
        extractor.smoothing_factor = self.features_smoothing_factor
        return extractor

    def create_sender(self):
        """
        Create the appropriate sender based on output.mode and output.midi_mode.

        Returns:
            OSCSender, ClassicMidiSender, or MusicalMidiSender instance.
        """
        mode = self.output_mode

        if mode == "osc":
            from vision_processor.osc_sender import OSCSender
            return OSCSender(host=self.osc_host, port=self.osc_port)

        elif mode == "midi":
            midi_mode = self.midi_mode

            if midi_mode == "musical":
                from vision_processor.midi.musical import MusicalMidiSender
                return MusicalMidiSender(
                    port_name=self.midi_port_name,
                    tempo_bpm=self.musical_tempo_bpm,
                    note_subdivision=self.musical_note_subdivision,
                    direction_threshold=self.musical_direction_threshold,
                    velocity_threshold=self.musical_velocity_threshold,
                    jump_size_slow=self.musical_jump_size_slow,
                    jump_size_fast=self.musical_jump_size_fast,
                )
            else:
                from vision_processor.midi.classic import ClassicMidiSender
                sender = ClassicMidiSender(port_name=self.midi_port_name)
                sender.JERK_THRESHOLD = self.midi_jerk_threshold
                return sender

        else:
            raise ValueError(f"Unknown output mode: {mode}")

    def send_features(self, sender, features: dict):
        """
        Send features using the appropriate method for the current config.

        Handles OSC individual vs bundle, and MIDI update.

        Args:
            sender: OSCSender or MidiSender instance.
            features: Feature dict from FeatureExtractor.
        """
        mode = self.output_mode

        if mode == "osc":
            if self.osc_send_mode == "bundle":
                sender.send_bundle(features)
            else:
                sender.send_features(features)
        elif mode == "midi":
            sender.update(features)

    # =========================================================================
    # Description helpers (for logging and reports)
    # =========================================================================

    def describe(self) -> str:
        """Human-readable summary of current config."""
        parts = [
            f"camera={self.camera_device_id} ({self.camera_width}x{self.camera_height})",
            f"pose={self.pose_model_complexity}",
            f"output={self.output_mode}",
        ]
        if self.output_mode == "osc":
            parts.append(f"send={self.osc_send_mode}")
        return " | ".join(parts)

    def to_metadata(self) -> dict:
        """
        Flat dict of config values for LatencyLogger metadata.

        Returns:
            Dict suitable for LatencyLogger(config=...).
        """
        backend = self.get("pose.backend") or self._detect_backend()

        meta = {
            "backend": backend,
            "camera_device_id": self.camera_device_id,
            "resolution": f"{self.camera_width}x{self.camera_height}",
            "pose_model_complexity": self.pose_model_complexity,
            "output_mode": self.output_mode,
        }
        if self.output_mode == "osc":
            meta["osc_send_mode"] = self.osc_send_mode
            meta["osc_target"] = f"{self.osc_host}:{self.osc_port}"
        elif self.output_mode == "midi":
            meta["midi_port"] = self.midi_port_name
            meta["jerk_threshold"] = self.midi_jerk_threshold

        for profile_name, profile in self.camera_profiles.items():
            if profile.get("device_id") == self.camera_device_id:
                meta["camera_name"] = profile.get("name", profile_name)
                break

        return meta

    def __repr__(self) -> str:
        return f"Config({self.describe()})"
