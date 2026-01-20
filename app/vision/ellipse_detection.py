from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import cv2
import numpy as np


Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]  # ((cx,cy),(major,minor), angle_deg)


@dataclass(frozen=True, slots=True)
class EllipseDetectConfig:
    """Tunable parameters for outer-ellipse detection."""
    min_area_ratio: float = 0.02          # min contour area / image area
    max_center_offset: float = 0.18       # max distance from image center / min(H,W)
    morph_kernel: int = 5                 # close kernel size (odd recommended)
    close_iters: int = 2                  # close iterations


def detect_outer_ellipse(
    mask_bgr: np.ndarray,
    *,
    cfg: Optional[EllipseDetectConfig] = None,
) -> Optional[Ellipse]:
    """
    Detect the most plausible "outer" ellipse from a (binary) mask image.

    Input:
      - mask_bgr: BGR image (typically 0/255 mask) of shape (H, W, 3).

    Output:
      - OpenCV ellipse tuple: ((cx, cy), (major_axis, minor_axis), angle_deg)
      - None if no valid ellipse is found.

    Selection criteria:
      - Largest contour area among candidates
      - Candidate must exceed cfg.min_area_ratio of image area
      - Ellipse center must be near the image center (cfg.max_center_offset)
      - Contour must have >= 5 points (fitEllipse requirement)
    """
    cfg = cfg or EllipseDetectConfig()

    if mask_bgr.ndim != 3 or mask_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H, W, 3); got {mask_bgr.shape}.")

    gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)

    k = max(1, int(cfg.morph_kernel))
    if k % 2 == 0:
        k += 1  # keep kernel odd for symmetry
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if cfg.close_iters > 0:
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=int(cfg.close_iters))

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    min_area = float(h * w) * float(cfg.min_area_ratio)
    max_offset = float(min(w, h)) * float(cfg.max_center_offset)
    cx0, cy0 = (w * 0.5), (h * 0.5)

    best_area = -1.0
    best_ellipse: Optional[Ellipse] = None

    for contour in contours:
        if contour.shape[0] < 5:
            continue

        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (a1, a2), angle = ellipse

        # Basic sanity checks
        if a1 <= 0.0 or a2 <= 0.0:
            continue
        if math.hypot(cx - cx0, cy - cy0) > max_offset:
            continue

        if area > best_area:
            best_area = area
            best_ellipse = ( (float(cx), float(cy)), (float(a1), float(a2)), float(angle) )

    return best_ellipse
