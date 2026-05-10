"""
Camera abstraction layer for CuerpoSonoro.

Provides a common interface for different video sources so the rest of
the pipeline doesn't need to know whether it's reading from a webcam or
a pre-recorded video file.

Usage:
    # Webcam (default)
    camera = WebcamCamera(device_id=0, width=1280, height=720, fps=30)

    # Video file (loops automatically)
    camera = VideoFileCamera("tests/videos/test_session.mp4")

    # Common interface for both
    while camera.is_open():
        frame = camera.read()
        if frame is None:
            break
        # process frame...

    camera.release()
"""

import time

import cv2
import numpy as np
from abc import ABC, abstractmethod


class BaseCamera(ABC):
    """Abstract base class defining the camera interface."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """
        Read next frame.

        Returns:
            BGR frame as numpy array, or None if no frame is available.
        """

    @abstractmethod
    def is_open(self) -> bool:
        """Return True if the camera/video source is open and ready."""

    @abstractmethod
    def release(self):
        """Release the camera/video source and free resources."""


class WebcamCamera(BaseCamera):
    """
    Live webcam capture using OpenCV.

    Wraps cv2.VideoCapture with an integer device ID.
    Configured for low-latency capture (buffer_size=1).
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        buffer_size: int = 1,
    ):
        """
        Args:
            device_id:   Camera index (0 = default/first camera).
            width:       Requested frame width in pixels.
            height:      Requested frame height in pixels.
            fps:         Requested frame rate.
            buffer_size: Internal capture buffer size. 1 minimises latency.
        """
        self._cap = cv2.VideoCapture(device_id)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera with device_id={device_id}. "
                "Check that the camera is connected and permissions are granted."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[WebcamCamera] device={device_id} | resolution={actual_w}x{actual_h}")

    def read(self) -> np.ndarray | None:
        """Read the next frame from the webcam, flipped horizontally."""
        ret, frame = self._cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)

    def is_open(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        self._cap.release()
        print("[WebcamCamera] Released.")

    def get(self, prop):
        return self._cap.get(prop)


class VideoFileCamera(BaseCamera):
    """
    Video file playback using OpenCV.

    Loops the video automatically when it reaches the end.
    Useful for repeatable testing and feature validation without
    needing a live camera.
    """

    def __init__(self, path: str, loop: bool = True, realtime: bool = True):
        """
        Args:
            path: Path to the video file (mp4, avi, mov, etc.)
            loop: If True, restart the video when it ends. Default True.
            realtime: If True, throttle reads to match the source FPS so
                downstream listeners (e.g., MIDI synth) hear the same timing
                a live performer would produce. When the pipeline is slower
                than source FPS, frames are skipped (via grab()) so wall-clock
                pacing stays tied to the original timeline. Default True.
        """
        self._path = path
        self._loop = loop
        self._realtime = realtime
        self._cap = self._open()
        self._frame_period = 1.0 / self._fps if self._realtime and self._fps > 0 else 0.0
        self._next_frame_at = time.perf_counter()
        self._skipped_frames = 0

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video file: '{self._path}'. "
                "Check that the file exists and the format is supported."
            )
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"[VideoFileCamera] file={self._path} | "
            f"resolution={w}x{h} | fps={self._fps:.1f} | frames={total} | loop={self._loop}"
        )
        return cap

    def read(self) -> np.ndarray | None:
        """
        Read the next frame. If the video ends and loop=True, restart from
        the beginning. If loop=False, return None to signal end of input.

        When realtime=True:
          - If the pipeline is faster than source FPS, sleep to match.
          - If the pipeline is slower than source FPS, skip ahead in the
            video (via cap.grab(), which decodes without copying to Python)
            so wall-clock pacing stays tied to the original timeline.
        """
        if self._frame_period > 0.0:
            now = time.perf_counter()
            sleep_for = self._next_frame_at - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                self._next_frame_at += self._frame_period
            else:
                frames_behind = int(-sleep_for / self._frame_period)
                for _ in range(frames_behind):
                    if not self._cap.grab():
                        break
                    self._skipped_frames += 1
                self._next_frame_at = now + self._frame_period

        ret, frame = self._cap.read()

        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._next_frame_at = time.perf_counter() + self._frame_period
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                return None

        return frame

    def is_open(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        self._cap.release()
        if self._skipped_frames > 0:
            print(
                f"[VideoFileCamera] Released. "
                f"Frames skipped to maintain realtime: {self._skipped_frames}."
            )
        else:
            print("[VideoFileCamera] Released.")
