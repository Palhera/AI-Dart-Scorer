import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration_data"


def calibration_path(cam_id: str) -> Path:
    return CALIBRATION_DIR / f"{cam_id}.json"


def _default_calibration(cam_id: str, width: int, height: int) -> Dict[str, Any]:
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
    path = calibration_path(cam_id)
    if path.exists():
        return path
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    data = _default_calibration(cam_id, width, height)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path


def load_calibration(cam_id: str, width: int, height: int) -> Dict[str, Any]:
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


class CalibrationRuntime:
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
        try:
            self._maybe_reload()
            out = self._apply_undistort(frame)
            out = self._apply_warp(out)
            return out
        except Exception:
            return frame

    def _load(self) -> None:
        self._data = load_calibration(self.cam_id, self._width, self._height)
        try:
            self._mtime = self._path.stat().st_mtime
        except Exception:
            self._mtime = None
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
        d = d[:4]

        size = (int(frame.shape[1]), int(frame.shape[0]))
        key = (size, k.tobytes(), d.tobytes())
        if self._undistort_key != key:
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

        size = (int(frame.shape[1]), int(frame.shape[0]))
        return cv2.warpPerspective(frame, h, size, flags=cv2.INTER_LINEAR)
