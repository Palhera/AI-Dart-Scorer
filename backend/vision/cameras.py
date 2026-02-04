import platform
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, List

import cv2
import numpy as np

from backend.vision.calibration_store import (
    CalibrationRuntime,
    rotate_warp_matrix,
    update_warp_matrix,
)


@dataclass
class CameraConfig:
    index: int
    width: int = 720
    height: int = 720
    fps: int = 30
    jpeg_quality: int = 70
    buffer_size: int = 1


class CameraRunner:
    # Internal "prepared" output resolution. Downstream calibration/warping assumes this size.
    _output_size = (720, 720)
    _warmup_frames = 2

    def __init__(self, cam_id: str, cfg: CameraConfig):
        self.cam_id = cam_id
        self.cfg = cfg

        # CalibrationRuntime encapsulates per-camera intrinsics + current warp (homography).
        # It is applied consistently in _prepare_frame() and snapshot encoding.
        self._calibration = CalibrationRuntime(cam_id, cfg.width, cfg.height)
        self._lock = threading.Lock()

    def capture_prepared_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        # Returns undistorted/cropped/resized frame for calibration routines.
        frame = self._capture_raw_frame(timeout=timeout)
        if frame is None:
            return None
        return self._prepare_frame(frame)

    def capture_warped_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        # Returns the prepared frame with the persisted warp applied.
        frame = self.capture_prepared_frame(timeout=timeout)
        if frame is None:
            return None
        return self._calibration.warp(frame)

    def capture_snapshot(self, timeout: float = 1.0, fmt: str = "jpg") -> Optional[bytes]:
        # Captures a single frame and encodes it as JPEG/PNG for UI previews.
        frame = self.capture_warped_frame(timeout=timeout)
        if frame is None:
            return None

        fmt = (fmt or "jpg").lower()
        if fmt in ("png", "image/png"):
            ok, buf = cv2.imencode(".png", frame)
        else:
            ok, buf = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)],
            )

        if not ok:
            return None
        return buf.tobytes()

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

    def _capture_raw_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        # Open the camera, grab a single frame, then release immediately.
        with self._lock:
            try:
                cap = self._open_capture()
            except Exception:
                return None

            try:
                return self._read_frame(cap, timeout=timeout)
            finally:
                try:
                    cap.release()
                except Exception:
                    pass

    def _read_frame(self, cap: cv2.VideoCapture, timeout: float = 1.0) -> Optional[np.ndarray]:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        warmup_left = self._warmup_frames
        last_frame = None

        while time.monotonic() <= deadline:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            last_frame = frame
            if warmup_left > 0:
                warmup_left -= 1
                continue
            return frame

        return last_frame

    def _open_capture(self) -> cv2.VideoCapture:
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

            return cap

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


class CameraManager:
    def __init__(self, configs: Dict[str, CameraConfig]):
        self.cams = {
            cam_id: CameraRunner(cam_id, cfg)
            for cam_id, cfg in configs.items()
        }

    def capture_snapshot(self, cam_id: str, fmt: str = "jpg", timeout: float = 1.0) -> Optional[bytes]:
        cam = self.cams.get(cam_id)
        if not cam:
            return None
        return cam.capture_snapshot(timeout=timeout, fmt=fmt)

    def capture_prepared_frame(self, cam_id: str, timeout: float = 1.0) -> Optional[np.ndarray]:
        cam = self.cams.get(cam_id)
        if not cam:
            return None
        return cam.capture_prepared_frame(timeout=timeout)

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
