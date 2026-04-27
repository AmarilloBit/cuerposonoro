"""
TensorRT backend for pose estimation on NVIDIA Jetson.

NOTE: This backend is a stub. Four separate GPU-acceleration paths were
attempted (MediaPipe pip wheel with GPU delegate, ONNX Runtime CUDA,
TFLite-to-ONNX conversion, TensorRT native TFLite parser) and all
failed for the reasons documented in the memoir's appendix F. The
production path on Jetson is MediaPipe on CPU with `jetson_clocks`.

This file is kept as a reference for future work, not as a working
backend.
"""

import logging
import os

import cv2
import numpy as np

from vision_processor.pose import BasePoseEstimator

logger = logging.getLogger(__name__)

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
_ONNX_PATH = os.path.join(_MODEL_DIR, "pose_model.onnx")
_ENGINE_PATH = os.path.join(_MODEL_DIR, "pose_engine.trt")

# Input dimensions expected by the BlazePose model
_INPUT_HEIGHT = 256
_INPUT_WIDTH = 256

# Number of landmarks in BlazePose output
_NUM_LANDMARKS = 33


class TensorRTPoseEstimator(BasePoseEstimator):
    """
    Pose estimator using TensorRT on NVIDIA Jetson GPU.

    Loads a TensorRT engine built from the BlazePose ONNX model.
    Inference runs entirely on the Jetson Ampere GPU.

    The result object returned by estimate() is a numpy array of shape
    (33, 4) with columns [x, y, z, visibility], normalized to [0, 1].
    All conversion to the standard landmark dict format happens here.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        # All GPU-specific imports are deferred to avoid import errors
        # on machines where pycuda/tensorrt are not installed.
        import tensorrt as trt

        self._trt = trt
        self._min_confidence = min_detection_confidence

        if not os.path.exists(_ENGINE_PATH):
            if not os.path.exists(_ONNX_PATH):
                raise FileNotFoundError(
                    f"ONNX model not found at {_ONNX_PATH}.\n"
                    "Run first: python debug_tools/convert_model.py"
                )
            logger.info("Building TensorRT engine (first run, ~2-3 min)...")
            self._build_engine()
            logger.info("Engine built and cached.")

        logger.info("Loading TensorRT engine...")
        self._engine, self._context = self._load_engine()
        self._allocate_buffers()

        import mediapipe as mp
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        logger.info("Backend: TensorRT (Jetson GPU Ampere)")

    def _build_engine(self):
        """Build TensorRT engine from ONNX model and save to disk."""
        trt = self._trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(_ONNX_PATH, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"Failed to parse ONNX model: {errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

        # FP16 enabled on Ampere and later
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 enabled.")

        serialized = builder.build_serialized_network(network, config)
        with open(_ENGINE_PATH, "wb") as f:
            f.write(serialized)

    def _load_engine(self):
        """Load TensorRT engine from disk."""
        trt = self._trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(_ENGINE_PATH, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        return engine, context

    def _allocate_buffers(self):
        """Allocate pinned host memory and device memory for inference I/O."""
        try:
            import pycuda.autoinit  # noqa: F401 — initializes CUDA context
            import pycuda.driver as cuda
        except ImportError as e:
            raise RuntimeError(
                "pycuda is required for the TensorRT backend."
                "Install it on the Jetson with: pip install pycuda"
            ) from e

        self._cuda = cuda
        self._inputs = []
        self._outputs = []
        self._bindings = []
        self._stream = cuda.Stream()

        for binding in self._engine:
            size = self._trt.volume(self._engine.get_binding_shape(binding))
            dtype = self._trt.nptype(self._engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(device_mem))

            if self._engine.binding_is_input(binding):
                self._inputs.append({"host": host_mem, "device": device_mem})
            else:
                self._outputs.append({"host": host_mem, "device": device_mem})

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare frame for TensorRT inference.

        Resizes to 256x256, converts BGR to RGB, normalizes to [0, 1],
        and reshapes to NCHW format (1, 3, 256, 256).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (_INPUT_WIDTH, _INPUT_HEIGHT))
        normalized = resized.astype(np.float32) / 255.0
        nchw = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(nchw)

    def estimate(self, frame: np.ndarray):
        """
        Run pose estimation on GPU via TensorRT.

        Returns a numpy array of shape (33, 4): [x, y, z, visibility].
        Returns None if no pose is detected (mean visibility below threshold).
        """
        cuda = self._cuda
        input_data = self._preprocess(frame)

        np.copyto(self._inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(
            self._inputs[0]["device"],
            self._inputs[0]["host"],
            self._stream,
        )

        self._context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle,
        )

        cuda.memcpy_dtoh_async(
            self._outputs[0]["host"],
            self._outputs[0]["device"],
            self._stream,
        )
        self._stream.synchronize()

        raw = self._outputs[0]["host"].reshape(_NUM_LANDMARKS, -1)

        if raw.shape[1] >= 4:
            if raw[:, 3].mean() < self._min_confidence:
                return None

        return raw

    def get_landmarks(self, results) -> list | None:
        """Convert TensorRT output array to standard landmark dict list."""
        if results is None:
            return None

        return [
            {
                "x": float(results[i, 0]),
                "y": float(results[i, 1]),
                "z": float(results[i, 2]),
                "visibility": float(results[i, 3]) if results.shape[1] > 3 else 1.0,
            }
            for i in range(_NUM_LANDMARKS)
        ]

    def draw_skeleton(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw pose skeleton from TensorRT landmark array."""
        if results is None:
            return frame

        from mediapipe.framework.formats import landmark_pb2

        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for i in range(_NUM_LANDMARKS):
            lm = landmark_list.landmark.add()
            lm.x = float(results[i, 0])
            lm.y = float(results[i, 1])
            lm.z = float(results[i, 2])

        self._mp_drawing.draw_landmarks(
            frame,
            landmark_list,
            self._mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        return frame

    def release(self):
        """Release TensorRT and CUDA resources."""
        del self._context
        del self._engine
        logger.info("TensorRT backend released.")
