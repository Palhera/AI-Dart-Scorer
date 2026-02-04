import platform
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

from backend.vision.calibration_store import (
    CalibrationRuntime,
    rotate_warp_matrix,
    update_warp_matrix,
)
from backend.vision.keypoint_detection import compute_homography_matrix


@dataclass
class CameraConfig:
    index: int
    width: int = 720
    height: int = 720
    fps: int = 30
    jpeg_fps: int = 15
    jpeg_quality: int = 70
    buffer_size: int = 1


class SyncClock:
    """
    Shared metronome to align JPEG publishing across multiple cameras.

    Rationale: the MJPEG streaming endpoint can poll per-camera "sequence numbers".
    Using a shared tick makes cameras publish at the same cadence, which helps
    multi-camera UIs stay visually synchronized and avoids per-camera drift.
    """

    def __init__(self, fps: float):
        self._interval = 1.0 / max(float(fps), 1.0)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._tick = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="SyncClock", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    def get_tick(self) -> int:
        with self._lock:
            return self._tick

    def wait_for_tick(self, last_tick: int, timeout: float) -> int:
        # Blocks until tick changes or timeout; used similarly to "wait for new frame".
        deadline = time.monotonic() + max(float(timeout), 0.0)
        with self._lock:
            while True:
                if self._tick != last_tick:
                    return self._tick
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._tick
                self._cond.wait(timeout=remaining)

    def _run(self) -> None:
        next_ts = time.monotonic()
        while not self._stop.is_set():
            next_ts += self._interval
            delay = next_ts - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            with self._lock:
                self._tick += 1
                self._cond.notify_all()


class CameraRunner:
    # Internal "prepared" output resolution. Downstream calibration/warping assumes this size.
    _output_size = (720, 720)

    def __init__(self, cam_id: str, cfg: CameraConfig, sync_clock: Optional[SyncClock] = None):
        self.cam_id = cam_id
        self.cfg = cfg
        self._sync_clock = sync_clock

        # CalibrationRuntime encapsulates per-camera intrinsics + current warp (homography).
        # It is applied consistently in _prepare_frame() and during JPEG publishing.
        self._calibration = CalibrationRuntime(cam_id, cfg.width, cfg.height)
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # Single lock protects both the latest raw frame and the latest encoded JPEG.
        # Condition variables are used to let API handlers wait for "newer than seq".
        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)
        self._jpeg_cond = threading.Condition(self._lock)
        self._latest_jpeg: Optional[bytes] = None
        self._latest_jpeg_seq = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_seq = 0
        self._is_open = False
        self._last_jpeg_ts = 0.0
        self._last_jpeg_tick = -1

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"Cam-{self.cam_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self._is_open = False

    def is_open(self) -> bool:
        return self._is_open

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_latest_jpeg_with_seq(self) -> Tuple[Optional[bytes], int]:
        with self._lock:
            return self._latest_jpeg, self._latest_jpeg_seq

    def wait_for_jpeg(self, last_seq: int, timeout: float = 1.0) -> Tuple[Optional[bytes], int]:
        # Used by the MJPEG streaming endpoint: blocks until a new JPEG is available
        # (seq differs from last_seq) or until timeout elapses.
        deadline = time.monotonic() + max(float(timeout), 0.0)
        with self._lock:
            while True:
                if self._latest_jpeg is not None and self._latest_jpeg_seq != last_seq:
                    return self._latest_jpeg, self._latest_jpeg_seq
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._latest_jpeg, self._latest_jpeg_seq
                self._jpeg_cond.wait(timeout=remaining)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        # Always return a copy to prevent callers from mutating shared state.
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def capture_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        # Waits for a newer raw frame than the one observed at call time.
        deadline = time.monotonic() + max(float(timeout), 0.0)
        with self._lock:
            start_seq = self._latest_frame_seq
            while True:
                if self._latest_frame is not None and self._latest_frame_seq != start_seq:
                    return self._latest_frame.copy()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if self._latest_frame is None:
                        return None
                    return self._latest_frame.copy()
                self._frame_cond.wait(timeout=remaining)

    def capture_prepared_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        # Convenience for calibration endpoints: returns undistorted/cropped/resized frame.
        frame = self.capture_frame(timeout=timeout)
        if frame is None:
            return None
        return self._prepare_frame(frame)

    def calibrate_homography(self) -> Optional[np.ndarray]:
        # Auto-calibration from the current prepared frame; persists the resulting warp.
        frame = self.get_latest_frame()
        if frame is None:
            return None

        frame = self._prepare_frame(frame)
        matrix = compute_homography_matrix(frame)
        if matrix is None:
            return None

        update_warp_matrix(self.cam_id, frame.shape[1], frame.shape[0], matrix)
        return matrix

    def rotate_homography(self, angle_deg: float) -> Optional[np.ndarray]:
        # Rotation is applied to the persisted warp for the prepared output size.
        width, height = self._output_size
        try:
            return rotate_warp_matrix(self.cam_id, width, height, angle_deg)
        except Exception:
            return None

    def reset_homography(self) -> bool:
        # Clears persisted warp (back to identity) for the prepared output size.
        width, height = self._output_size
        try:
            update_warp_matrix(self.cam_id, width, height, None)
            return True
        except Exception:
            return False

    def _center_crop_square(self, frame: np.ndarray) -> np.ndarray:
        # Cropping before resizing keeps geometry symmetric around the optical center,
        # which matters for homography/board-centered calibrations.
        h, w = frame.shape[:2]
        if w == h:
            return frame
        if w > h:
            x0 = (w - h) // 2
            return frame[:, x0 : x0 + h]
        y0 = (h - w) // 2
        return frame[y0 : y0 + w, :]

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        # Defines the canonical coordinate system for all downstream vision steps.
        frame = self._calibration.undistort(frame)
        frame = self._center_crop_square(frame)
        if frame.shape[1] != self._output_size[0] or frame.shape[0] != self._output_size[1]:
            frame = cv2.resize(frame, self._output_size, interpolation=cv2.INTER_LINEAR)
        return frame

    def _open(self) -> None:
        # Try platform-specific backends in order; camera indices/backends can vary widely
        # across OSes and drivers, so we probe until we can reliably read frames.
        backends = self._candidate_backends()

        for backend in backends:
            cap = cv2.VideoCapture(self.cfg.index, backend)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.cfg.buffer_size))
            except Exception:
                pass

            if not cap.isOpened():
                cap.release()
                continue

            # Some backends report "opened" but fail to deliver frames initially.
            ok = False
            for _ in range(5):
                ok, _ = cap.read()
                if ok:
                    break
                time.sleep(0.1)

            if not ok:
                cap.release()
                continue

            self._cap = cap
            self._is_open = True
            return

        raise RuntimeError(
            f"Camera {self.cam_id} (index={self.cfg.index}) cannot read frames"
        )

    @staticmethod
    def _candidate_backends() -> List[int]:
        system = platform.system().lower()
        if system == "windows":
            raw = [cv2.CAP_DSHOW, getattr(cv2, "CAP_MSMF", None), cv2.CAP_ANY]
        elif system == "linux":
            raw = [getattr(cv2, "CAP_V4L2", None), getattr(cv2, "CAP_GSTREAMER", None), cv2.CAP_ANY]
        elif system == "darwin":
            raw = [getattr(cv2, "CAP_AVFOUNDATION", None), cv2.CAP_ANY]
        else:
            raw = [cv2.CAP_ANY]

        seen = set()
        backends: List[int] = []
        for item in raw:
            if item is None:
                continue
            if item in seen:
                continue
            seen.add(item)
            backends.append(int(item))
        return backends or [cv2.CAP_ANY]

    def _run(self) -> None:
        # Dedicated thread: reads frames as fast as the driver provides them, and
        # publishes (1) latest raw frame and (2) rate-limited JPEGs for streaming.
        while not self._stop.is_set():
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._open()

                ok, frame = self._cap.read()
                if not ok:
                    # Treat read failures as transient and reopen after a short backoff.
                    self._cap.release()
                    self._cap = None
                    self._is_open = False
                    time.sleep(0.2)
                    continue

                with self._lock:
                    self._latest_frame = frame
                    self._latest_frame_seq += 1
                    self._frame_cond.notify_all()

                # JPEG publish pacing:
                # - if SyncClock is provided, publish at shared ticks
                # - otherwise, use per-camera wall-clock throttling
                if self._sync_clock is not None:
                    tick = self._sync_clock.get_tick()
                    if tick == self._last_jpeg_tick:
                        time.sleep(0.001)
                        continue
                    self._last_jpeg_tick = tick
                else:
                    now = time.monotonic()
                    jpeg_interval = 1.0 / max(float(self.cfg.jpeg_fps), 1.0)
                    if now - self._last_jpeg_ts < jpeg_interval:
                        time.sleep(0.001)
                        continue

                frame = self._prepare_frame(frame)
                frame = self._calibration.warp(frame)

                ok, buf = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)],
                )
                if ok:
                    data = buf.tobytes()
                    with self._lock:
                        self._latest_jpeg = data
                        # Sequence numbers are used by clients to wait for "new" frames.
                        self._latest_jpeg_seq = self._last_jpeg_tick if self._sync_clock is not None else (self._latest_jpeg_seq + 1)
                        self._jpeg_cond.notify_all()
                    if self._sync_clock is None:
                        self._last_jpeg_ts = now

                time.sleep(0.001)

            except Exception:
                # Fail closed: release the capture and retry after a backoff.
                self._is_open = False
                try:
                    if self._cap is not None:
                        self._cap.release()
                except Exception:
                    pass
                self._cap = None
                time.sleep(0.5)


class CameraManager:
    def __init__(self, configs: Dict[str, CameraConfig]):
        # Use the minimum jpeg_fps so all cameras can comfortably follow the shared clock.
        sync_fps = min((cfg.jpeg_fps for cfg in configs.values()), default=15)
        self._sync_clock = SyncClock(sync_fps)
        self.cams = {
            cam_id: CameraRunner(cam_id, cfg, sync_clock=self._sync_clock)
            for cam_id, cfg in configs.items()
        }

    def start_all(self) -> None:
        self._sync_clock.start()
        for cam in self.cams.values():
            cam.start()

    def stop_all(self) -> None:
        for cam in self.cams.values():
            cam.stop()
        self._sync_clock.stop()

    def all_ready(self) -> bool:
        return all(cam.is_open() for cam in self.cams.values())

    def get_jpeg(self, cam_id: str) -> Optional[bytes]:
        cam = self.cams.get(cam_id)
        if not cam:
            return None
        return cam.get_latest_jpeg()

    def get_jpeg_with_seq(self, cam_id: str) -> Tuple[Optional[bytes], int]:
        cam = self.cams.get(cam_id)
        if not cam:
            return None, 0
        return cam.get_latest_jpeg_with_seq()

    def wait_for_jpeg(self, cam_id: str, last_seq: int, timeout: float = 1.0) -> Tuple[Optional[bytes], int]:
        cam = self.cams.get(cam_id)
        if not cam:
            return None, last_seq
        return cam.wait_for_jpeg(last_seq, timeout=timeout)

    def capture_frame(self, cam_id: str, timeout: float = 1.0) -> Optional[np.ndarray]:
        cam = self.cams.get(cam_id)
        if not cam:
            return None
        return cam.capture_frame(timeout=timeout)

    def capture_prepared_frame(self, cam_id: str, timeout: float = 1.0) -> Optional[np.ndarray]:
        cam = self.cams.get(cam_id)
        if not cam:
            return None
        return cam.capture_prepared_frame(timeout=timeout)

    def calibrate_homography(self, cam_id: str) -> Optional[np.ndarray]:
        cam = self.cams.get(cam_id)
        if not cam:
            return None
        return cam.calibrate_homography()

    def rotate_homography(self, cam_id: str, angle_deg: float) -> Optional[np.ndarray]:
        cam = self.cams.get(cam_id)
        if not cam:
            return None
        return cam.rotate_homography(angle_deg)

    def reset_homography(self, cam_id: str) -> bool:
        cam = self.cams.get(cam_id)
        if not cam:
            return False
        return cam.reset_homography()
