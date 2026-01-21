from __future__ import annotations

import base64
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.vision.vision_types import U8
PointI = Tuple[int, int]


def ensure_bgr_u8(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H, W, 3), got {img_bgr.shape}.")
    if img_bgr.dtype != U8:
        # OpenCV works best with uint8 for these operations.
        img_bgr = img_bgr.astype(U8, copy=False)
    return img_bgr


def to_3ch(mask_u8: np.ndarray) -> np.ndarray:
    """Convert single-channel uint8 mask to 3-channel BGR mask."""
    if mask_u8.ndim != 2:
        raise ValueError(f"Expected single-channel mask, got {mask_u8.shape}.")
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def decode_base64_image(image_b64: str) -> Optional[np.ndarray]:
    """
    Decode base64 (optionally data URL) into OpenCV BGR uint8 image.
    Returns None if decoding fails.
    """
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[-1]

    try:
        raw = base64.b64decode(image_b64, validate=True)
    except Exception:
        try:
            raw = base64.b64decode(image_b64)
        except Exception:
            return None

    data = np.frombuffer(raw, dtype=U8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def line_border_points(rho: float, theta: float, width: int, height: int) -> List[PointI]:
    """Return up to 2 intersection points between the line and the image border."""
    pts: List[PointI] = []
    ct, st = math.cos(theta), math.sin(theta)

    # Intersections with x = 0 and x = width-1
    if abs(st) > 1e-6:
        y0 = (rho - 0.0 * ct) / st
        y1 = (rho - (width - 1.0) * ct) / st
        if 0.0 <= y0 <= height - 1.0:
            pts.append((0, int(round(y0))))
        if 0.0 <= y1 <= height - 1.0:
            pts.append((width - 1, int(round(y1))))

    # Intersections with y = 0 and y = height-1
    if abs(ct) > 1e-6:
        x0 = (rho - 0.0 * st) / ct
        x1 = (rho - (height - 1.0) * st) / ct
        if 0.0 <= x0 <= width - 1.0:
            pts.append((int(round(x0)), 0))
        if 0.0 <= x1 <= width - 1.0:
            pts.append((int(round(x1)), height - 1))

    # Remove duplicates, keep first two
    uniq: List[PointI] = []
    for p in pts:
        if p not in uniq:
            uniq.append(p)
    return uniq[:2]


__all__ = ["decode_base64_image", "ensure_bgr_u8", "line_border_points", "to_3ch"]
