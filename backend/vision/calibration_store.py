import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration_data"


def calibration_path(cam_id: str) -> Path:
    return CALIBRATION_DIR / f"{cam_id}.json"


def _default_calibration(cam_id: str, width: int, height: int) -> Dict[str, Any]:
    # Single on-disk schema for all cameras; kept human-readable for manual inspection/editing.
    return {
        "camera_id": cam_id,
        "image_size": {"width": width, "height": height},
        "undistort": {
            "camera_matrix": None,
            "dist_coeffs": None,
        },
        "warp": {
            "matrix": None,
        },
    }


def ensure_calibration_file(cam_id: str, width: int, height: int) -> Path:
    # Create a per-camera calibration file on first run so runtime code can assume it exists.
    path = calibration_path(cam_id)
    if path.exists():
        return path
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    data = _default_calibration(cam_id, width, height)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path


def load_calibration(cam_id: str, width: int, height: int) -> Dict[str, Any]:
    # Fail-safe load: if file is corrupted/invalid, rewrite defaults to recover automatically.
    path = ensure_calibration_file(cam_id, width, height)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    data = _default_calibration(cam_id, width, height)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return data


def update_warp_matrix(
    cam_id: str,
    width: int,
    height: int,
    matrix: Optional[np.ndarray],
) -> Dict[str, Any]:
    # Persists the homography to disk so calibration survives restarts.
    path = ensure_calibration_file(cam_id, width, height)
    data = load_calibration(cam_id, width, height)

    if not isinstance(data, dict):
        data = _default_calibration(cam_id, width, height)

    data["camera_id"] = cam_id
    data["image_size"] = {"width": int(width), "height": int(height)}

    warp = data.get("warp", {}) if isinstance(data.get("warp"), dict) else {}
    if matrix is None:
        warp["matrix"] = None
    else:
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError(f"Expected 3x3 homography matrix, got {mat.shape}.")
        warp["matrix"] = mat.tolist()
    data["warp"] = warp

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return data


def _rotation_homography(angle_deg: float, width: int, height: int) -> np.ndarray:
    # Builds a pixel-space rotation about the image center (not the origin).
    if width <= 0 or height <= 0:
        raise ValueError("Image size must be positive for rotation.")

    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)

    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    t1 = np.array(
        [
            [1.0, 0.0, -cx],
            [0.0, 1.0, -cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    t2 = np.array(
        [
            [1.0, 0.0, cx],
            [0.0, 1.0, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return t2 @ rot @ t1


def rotate_warp_matrix(
    cam_id: str,
    width: int,
    height: int,
    angle_deg: float,
) -> np.ndarray:
    # Applies an additional rotation on top of the currently persisted warp, then saves it back.
    data = load_calibration(cam_id, width, height)
    warp = data.get("warp", {}) if isinstance(data, dict) else {}
    h_raw = warp.get("matrix")

    if h_raw is None:
        base = np.eye(3, dtype=np.float64)
    else:
        base = np.asarray(h_raw, dtype=np.float64)
        if base.shape != (3, 3):
            raise ValueError(f"Expected 3x3 homography matrix, got {base.shape}.")

    rot = _rotation_homography(angle_deg, width, height)
    rotated = rot @ base
    update_warp_matrix(cam_id, width, height, rotated)
    return rotated


class CalibrationRuntime:
    """
    Lightweight runtime wrapper around the on-disk calibration file.

    Key property: it hot-reloads when the JSON file changes, allowing calibration
    updates via API/UI without restarting camera threads.
    """

    def __init__(self, cam_id: str, width: int, height: int):
        self.cam_id = cam_id
        self._width = width
        self._height = height
        self._path = ensure_calibration_file(cam_id, width, height)

        self._data: Dict[str, Any] = {}
        self._mtime: Optional[float] = None
        self._undistort_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._undistort_key: Optional[Tuple] = None

        self._load()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        # Convenience for pipelines that want both undistort + warp in one step.
        try:
            self._maybe_reload()
            out = self.undistort(frame)
            out = self._apply_warp(out)
            return out
        except Exception:
            return frame

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        try:
            self._maybe_reload()
            return self._apply_undistort(frame)
        except Exception:
            return frame

    def warp(self, frame: np.ndarray) -> np.ndarray:
        try:
            self._maybe_reload()
            return self._apply_warp(frame)
        except Exception:
            return frame

    def _load(self) -> None:
        self._data = load_calibration(self.cam_id, self._width, self._height)
        try:
            self._mtime = self._path.stat().st_mtime
        except Exception:
            self._mtime = None
        # Any calibration change invalidates cached undistort maps.
        self._undistort_maps = None
        self._undistort_key = None

    def _maybe_reload(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except Exception:
            return
        if self._mtime is None or mtime > self._mtime:
            self._load()

    def _apply_undistort(self, frame: np.ndarray) -> np.ndarray:
        cfg = self._data.get("undistort", {}) if isinstance(self._data, dict) else {}
        k_raw = cfg.get("camera_matrix")
        d_raw = cfg.get("dist_coeffs")
        if k_raw is None or d_raw is None:
            return frame

        k = np.asarray(k_raw, dtype=np.float32)
        d = np.asarray(d_raw, dtype=np.float32).reshape(-1)
        if k.shape != (3, 3) or d.size < 4:
            return frame
        # This runtime only supports fisheye 4-coefficient distortion (k1..k4).
        d = d[:4]

        size = (int(frame.shape[1]), int(frame.shape[0]))
        key = (size, k.tobytes(), d.tobytes())
        if self._undistort_key != key:
            # Computing maps is expensive; cache and recompute only when parameters change.
            self._undistort_key = key
            r = np.eye(3, dtype=np.float32)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                k, d, r, k, size, cv2.CV_16SC2
            )
            self._undistort_maps = (map1, map2)

        if not self._undistort_maps:
            return frame
        map1, map2 = self._undistort_maps
        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    def _apply_warp(self, frame: np.ndarray) -> np.ndarray:
        cfg = self._data.get("warp", {}) if isinstance(self._data, dict) else {}
        h_raw = cfg.get("matrix")
        if h_raw is None:
            return frame
        h = np.asarray(h_raw, dtype=np.float32)
        if h.shape != (3, 3):
            return frame

        # Warp is applied in the current frame's pixel space; callers should ensure
        # a consistent prepared size if they want stable calibration results.
        size = (int(frame.shape[1]), int(frame.shape[0]))
        return cv2.warpPerspective(frame, h, size, flags=cv2.INTER_LINEAR)
