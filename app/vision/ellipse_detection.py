from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import cv2
import numpy as np

from app.vision.vision_types import ImageInput, TransformParams, U8
from app.vision.vision_utils import decode_base64_image, ensure_bgr_u8, to_3ch


Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]  # ((cx,cy),(major,minor), angle_deg)


@dataclass(frozen=True, slots=True)
class EllipseDetectConfig:
    """Tunable parameters for outer-ellipse detection."""
    min_area_ratio: float = 0.02          # min contour area / image area
    max_center_offset: float = 0.18       # max distance from image center / min(H,W)
    morph_kernel: int = 5                 # close kernel size (odd recommended)
    close_iters: int = 2                  # close iterations


def build_red_green_mask(img_bgr: np.ndarray, params: Optional[TransformParams] = None) -> np.ndarray:
    """
    Build a 3-channel (BGR) binary mask for red and green regions in HSV space.
    """
    params = params or TransformParams()
    img_bgr = ensure_bgr_u8(img_bgr)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    low_green = np.array(params.low_green, dtype=U8)
    high_green = np.array(params.high_green, dtype=U8)
    low_red_1 = np.array(params.low_red_1, dtype=U8)
    high_red_1 = np.array(params.high_red_1, dtype=U8)
    low_red_2 = np.array(params.low_red_2, dtype=U8)
    high_red_2 = np.array(params.high_red_2, dtype=U8)

    green = cv2.inRange(hsv, low_green, high_green)
    red1 = cv2.inRange(hsv, low_red_1, high_red_1)
    red2 = cv2.inRange(hsv, low_red_2, high_red_2)

    mask_u8 = cv2.bitwise_or(green, cv2.bitwise_or(red1, red2))
    return to_3ch(mask_u8)


def detect_outer_ellipse_from_image(
    image_input: ImageInput,
    *,
    params: Optional[TransformParams] = None,
    cfg: Optional[EllipseDetectConfig] = None,
) -> Optional[Ellipse]:
    """
    Detect the outer ellipse from a BGR image or base64 input.
    """
    img_bgr = decode_base64_image(image_input) if isinstance(image_input, str) else image_input
    if img_bgr is None:
        return None

    params = params or TransformParams()
    mask_bgr = build_red_green_mask(img_bgr, params=params)
    return detect_outer_ellipse(mask_bgr, cfg=cfg)


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
