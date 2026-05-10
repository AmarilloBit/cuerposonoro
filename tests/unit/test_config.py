"""
Unit tests for vision_processor/config.py

Tests Config loading, dot-notation access, override copies, backend
detection, and factory methods. No camera or hardware required.

Usage:
    pytest tests/unit/test_config.py -v
"""

import os
import sys
import tempfile
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vision_processor.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_dir: str, content: str) -> str:
    """Write a YAML string to a temp file, return the path."""
    path = os.path.join(tmp_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(textwrap.dedent(content))
    return path


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def minimal_yaml(tmp_dir):
    """A minimal config.yaml with a few keys set."""
    return _write_yaml(tmp_dir, """\
        camera:
          device_id: 2
          width: 640
          height: 480
          fps: 60
          buffer_size: 2
        pose:
          model_complexity: 0
          min_detection_confidence: 0.7
          min_tracking_confidence: 0.8
        features:
          smoothing_factor: 0.5
        output:
          mode: midi
          midi_mode: rhythmical
        osc:
          host: "10.0.0.1"
          port: 9000
          send_mode: bundle
        midi:
          port_name: "TestPort"
          jerk_threshold: 0.35
    """)


@pytest.fixture
def empty_yaml(tmp_dir):
    """An empty YAML file (loads as None)."""
    return _write_yaml(tmp_dir, "")


# ===========================================================================
# Loading & defaults
# ===========================================================================

class TestLoading:

    def test_loads_from_file(self, minimal_yaml):
        config = Config(path=minimal_yaml)
        assert config.camera_device_id == 2

    def test_missing_file_uses_defaults(self, tmp_dir):
        config = Config(path=os.path.join(tmp_dir, "nonexistent.yaml"))
        assert config.camera_device_id == 0
        assert config.camera_width == 1280

    def test_empty_file_uses_defaults(self, empty_yaml):
        config = Config(path=empty_yaml)
        assert config.camera_device_id == 0

    def test_overrides_in_constructor(self, minimal_yaml):
        config = Config(path=minimal_yaml, overrides={"camera.device_id": 7})
        assert config.camera_device_id == 7


# ===========================================================================
# get / set
# ===========================================================================

class TestGetSet:

    def test_get_existing_key(self, minimal_yaml):
        config = Config(path=minimal_yaml)
        assert config.get("camera.width") == 640

    def test_get_missing_key_returns_default(self, minimal_yaml):
        config = Config(path=minimal_yaml)
        assert config.get("nonexistent.key", 42) == 42

    def test_get_nested_missing_returns_default(self, minimal_yaml):
        config = Config(path=minimal_yaml)
        assert config.get("camera.nonexistent", -1) == -1

    def test_set_creates_intermediate_dicts(self, empty_yaml):
        config = Config(path=empty_yaml)
        config.set("a.b.c", 99)
        assert config.get("a.b.c") == 99

    def test_set_overwrites_existing(self, minimal_yaml):
        config = Config(path=minimal_yaml)
        config.set("camera.device_id", 5)
        assert config.get("camera.device_id") == 5


# ===========================================================================
# with_overrides
# ===========================================================================

class TestWithOverrides:

    def test_returns_new_config(self, minimal_yaml):
        original = Config(path=minimal_yaml)
        copy = original.with_overrides(camera__width=320)
        assert copy.camera_width == 320
        assert original.camera_width == 640  # unchanged

    def test_double_underscore_becomes_dot(self, minimal_yaml):
        config = Config(path=minimal_yaml)
        copy = config.with_overrides(pose__model_complexity=2)
        assert copy.pose_model_complexity == 2

    def test_multiple_overrides(self, minimal_yaml):
        config = Config(path=minimal_yaml)
        copy = config.with_overrides(
            camera__device_id=0,
            camera__width=1920,
            pose__model_complexity=2,
        )
        assert copy.camera_device_id == 0
        assert copy.camera_width == 1920
        assert copy.pose_model_complexity == 2


# ===========================================================================
# Property accessors
# ===========================================================================

class TestProperties:

    def test_camera_properties(self, minimal_yaml):
        c = Config(path=minimal_yaml)
        assert c.camera_device_id == 2
        assert c.camera_width == 640
        assert c.camera_height == 480
        assert c.camera_fps == 60
        assert c.camera_buffer_size == 2

    def test_pose_properties(self, minimal_yaml):
        c = Config(path=minimal_yaml)
        assert c.pose_model_complexity == 0
        assert c.pose_min_detection_confidence == 0.7
        assert c.pose_min_tracking_confidence == 0.8

    def test_features_properties(self, minimal_yaml):
        c = Config(path=minimal_yaml)
        assert c.features_smoothing_factor == 0.5

    def test_output_properties(self, minimal_yaml):
        c = Config(path=minimal_yaml)
        assert c.output_mode == "midi"
        assert c.midi_mode == "rhythmical"

    def test_osc_properties(self, minimal_yaml):
        c = Config(path=minimal_yaml)
        assert c.osc_host == "10.0.0.1"
        assert c.osc_port == 9000
        assert c.osc_send_mode == "bundle"

    def test_midi_properties(self, minimal_yaml):
        c = Config(path=minimal_yaml)
        assert c.midi_port_name == "TestPort"
        assert c.midi_jerk_threshold == 0.35

    def test_defaults_when_missing(self, empty_yaml):
        c = Config(path=empty_yaml)
        assert c.camera_device_id == 0
        assert c.camera_width == 1280
        assert c.camera_height == 720
        assert c.camera_fps == 30
        assert c.camera_buffer_size == 1
        assert c.pose_model_complexity == 1
        assert c.features_smoothing_factor == 0.3
        assert c.output_mode == "osc"
        assert c.midi_mode == "classic"
        assert c.osc_host == "127.0.0.1"
        assert c.osc_port == 57120
        assert c.osc_send_mode == "individual"
        assert c.midi_port_name == "CuerpoSonoro"
        assert c.midi_jerk_threshold == 0.4

    def test_camera_profiles(self, tmp_dir):
        path = _write_yaml(tmp_dir, """\
            camera_profiles:
              macbook:
                device_id: 1
                name: "MacBook"
        """)
        c = Config(path=path)
        assert "macbook" in c.camera_profiles
        assert c.camera_profiles["macbook"]["name"] == "MacBook"

    def test_benchmark_defaults(self, empty_yaml):
        c = Config(path=empty_yaml)
        assert c.benchmark_frames == 300
        assert c.benchmark_warmup == 30
        assert len(c.benchmark_resolutions) == 2
        assert len(c.benchmark_pose_models) == 2
        assert len(c.benchmark_output_modes) == 3
        assert len(c.benchmark_backends) == 1

    def test_rhythmical_properties_defaults(self, empty_yaml):
        c = Config(path=empty_yaml)
        assert c.rhythmical_tempo_bpm == 120
        assert c.rhythmical_melody_velocity_floor == 0.05


# ===========================================================================
# _detect_backend
# ===========================================================================

class TestDetectBackend:

    def test_cpu_always_available(self):
        assert Config._detect_backend("cpu") == "cpu"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            Config._detect_backend("vulkan")

    def test_tensorrt_raises_when_unavailable(self):
        with pytest.raises(RuntimeError, match="TensorRT is not available"):
            Config._detect_backend("tensorrt")

    def test_auto_detect_returns_string(self):
        result = Config._detect_backend()
        assert result in ("tensorrt", "metal", "cpu")


# ===========================================================================
# Factory: create_feature_extractor
# ===========================================================================

class TestCreateFeatureExtractor:

    def test_returns_extractor(self, minimal_yaml):
        config = Config(path=minimal_yaml)
        ext = config.create_feature_extractor()
        from vision_processor.features import FeatureExtractor
        assert isinstance(ext, FeatureExtractor)

    def test_applies_smoothing_factor(self, minimal_yaml):
        config = Config(path=minimal_yaml)
        ext = config.create_feature_extractor()
        assert ext.smoothing_factor == 0.5


# ===========================================================================
# Factory: create_sender
# ===========================================================================

class TestCreateSender:

    def test_osc_sender(self, tmp_dir):
        from unittest.mock import patch, MagicMock
        path = _write_yaml(tmp_dir, "output:\n  mode: osc")
        config = Config(path=path)
        with patch("vision_processor.osc_sender.udp_client.SimpleUDPClient"):
            sender = config.create_sender()
        from vision_processor.osc_sender import OSCSender
        assert isinstance(sender, OSCSender)

    def test_classic_midi_sender(self, tmp_dir):
        from unittest.mock import patch, MagicMock
        path = _write_yaml(tmp_dir, """\
            output:
              mode: midi
              midi_mode: classic
            midi:
              port_name: "Test"
              jerk_threshold: 0.5
        """)
        config = Config(path=path)
        with patch("vision_processor.midi.classic.mido.open_output") as mock_open:
            mock_open.return_value = MagicMock()
            sender = config.create_sender()
        from vision_processor.midi.classic import ClassicMidiSender
        assert isinstance(sender, ClassicMidiSender)
        assert sender.JERK_THRESHOLD == 0.5

    def test_rhythmical_midi_sender(self, tmp_dir):
        from unittest.mock import patch, MagicMock
        path = _write_yaml(tmp_dir, """\
            output:
              mode: midi
              midi_mode: rhythmical
        """)
        config = Config(path=path)
        with patch("vision_processor.midi.rhythmical.mido.open_output") as mock_open:
            mock_open.return_value = MagicMock()
            sender = config.create_sender()
        from vision_processor.midi.rhythmical import RhythmicalMidiSender
        assert isinstance(sender, RhythmicalMidiSender)
        sender.close()

    def test_unknown_mode_raises(self, tmp_dir):
        path = _write_yaml(tmp_dir, "output:\n  mode: foobar")
        config = Config(path=path)
        with pytest.raises(ValueError, match="Unknown output mode"):
            config.create_sender()


# ===========================================================================
# send_features
# ===========================================================================

class TestSendFeatures:

    def test_osc_individual(self, tmp_dir):
        from unittest.mock import patch, MagicMock
        path = _write_yaml(tmp_dir, """\
            output:
              mode: osc
            osc:
              send_mode: individual
        """)
        config = Config(path=path)
        sender = MagicMock()
        config.send_features(sender, {"energy": 0.5})
        sender.send_features.assert_called_once_with({"energy": 0.5})

    def test_osc_bundle(self, tmp_dir):
        from unittest.mock import MagicMock
        path = _write_yaml(tmp_dir, """\
            output:
              mode: osc
            osc:
              send_mode: bundle
        """)
        config = Config(path=path)
        sender = MagicMock()
        config.send_features(sender, {"energy": 0.5})
        sender.send_bundle.assert_called_once_with({"energy": 0.5})

    def test_midi_update(self, tmp_dir):
        from unittest.mock import MagicMock
        path = _write_yaml(tmp_dir, "output:\n  mode: midi")
        config = Config(path=path)
        sender = MagicMock()
        config.send_features(sender, {"energy": 0.5})
        sender.update.assert_called_once_with({"energy": 0.5})


# ===========================================================================
# describe / to_metadata
# ===========================================================================

class TestDescription:

    def test_describe_contains_key_info(self, minimal_yaml):
        config = Config(path=minimal_yaml)
        desc = config.describe()
        assert "640x480" in desc
        assert "midi" in desc

    def test_repr(self, minimal_yaml):
        config = Config(path=minimal_yaml)
        assert "Config(" in repr(config)

    def test_to_metadata_osc(self, tmp_dir):
        path = _write_yaml(tmp_dir, """\
            output:
              mode: osc
            osc:
              host: "10.0.0.1"
              port: 9000
              send_mode: bundle
        """)
        config = Config(path=path)
        meta = config.to_metadata()
        assert meta["output_mode"] == "osc"
        assert meta["osc_send_mode"] == "bundle"
        assert "osc_target" in meta

    def test_to_metadata_midi(self, tmp_dir):
        path = _write_yaml(tmp_dir, """\
            output:
              mode: midi
            midi:
              port_name: "TestPort"
              jerk_threshold: 0.35
        """)
        config = Config(path=path)
        meta = config.to_metadata()
        assert meta["output_mode"] == "midi"
        assert meta["midi_port"] == "TestPort"
        assert meta["jerk_threshold"] == 0.35

    def test_to_metadata_camera_name(self, tmp_dir):
        path = _write_yaml(tmp_dir, """\
            camera:
              device_id: 1
            camera_profiles:
              macbook:
                device_id: 1
                name: "MacBook Pro"
        """)
        config = Config(path=path)
        meta = config.to_metadata()
        assert meta.get("camera_name") == "MacBook Pro"
