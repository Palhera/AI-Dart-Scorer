import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict

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


class CameraRunner:
    _output_size = (720, 720)

    def __init__(self, cam_id: str, cfg: CameraConfig):
        self.cam_id = cam_id
        self.cfg = cfg

        self._calibration = CalibrationRuntime(cam_id, cfg.width, cfg.height)
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_seq = 0
        self._is_open = False
        self._last_jpeg_ts = 0.0

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

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def capture_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
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
        frame = self.capture_frame(timeout=timeout)
        if frame is None:
            return None
        return self._prepare_frame(frame)

    def calibrate_homography(self) -> Optional[np.ndarray]:
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
        width, height = self._output_size
        try:
            return rotate_warp_matrix(self.cam_id, width, height, angle_deg)
        except Exception:
            return None

    def reset_homography(self) -> bool:
        width, height = self._output_size
        try:
            update_warp_matrix(self.cam_id, width, height, None)
            return True
        except Exception:
            return False

    def _center_crop_square(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w == h:
            return frame
        if w > h:
            x0 = (w - h) // 2
            return frame[:, x0 : x0 + h]
        y0 = (h - w) // 2
        return frame[y0 : y0 + w, :]

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = self._calibration.undistort(frame)
        frame = self._center_crop_square(frame)
        if frame.shape[1] != self._output_size[0] or frame.shape[0] != self._output_size[1]:
            frame = cv2.resize(frame, self._output_size, interpolation=cv2.INTER_LINEAR)
        return frame

    def _open(self) -> None:
        cap = cv2.VideoCapture(self.cfg.index, cv2.CAP_DSHOW)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Camera {self.cam_id} (index={self.cfg.index}) cannot read frames")

        self._cap = cap
        self._is_open = True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._open()

                ok, frame = self._cap.read()
                if not ok:
                    self._cap.release()
                    self._cap = None
                    self._is_open = False
                    time.sleep(0.2)
                    continue

                with self._lock:
                    self._latest_frame = frame
                    self._latest_frame_seq += 1
                    self._frame_cond.notify_all()

                now = time.monotonic()
                jpeg_interval = 1.0 / max(float(self.cfg.jpeg_fps), 1.0)
                if now - self._last_jpeg_ts < jpeg_interval:
                    time.sleep(0.001)
                    continue

                frame = self._prepare_frame(frame)
                frame = self._calibration.warp(frame)

                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    data = buf.tobytes()
                    with self._lock:
                        self._latest_jpeg = data
                    self._last_jpeg_ts = now

                time.sleep(0.001)

            except Exception:
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
        self.cams = {cam_id: CameraRunner(cam_id, cfg) for cam_id, cfg in configs.items()}

    def start_all(self) -> None:
        for cam in self.cams.values():
            cam.start()

    def stop_all(self) -> None:
        for cam in self.cams.values():
            cam.stop()

    def all_ready(self) -> bool:
        return all(cam.is_open() for cam in self.cams.values())

    def get_jpeg(self, cam_id: str) -> Optional[bytes]:
        cam = self.cams.get(cam_id)
        if not cam:
            return None
        return cam.get_latest_jpeg()

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
