import math
from typing import Optional, Tuple

import cv2
import numpy as np

from backend.vision.vision_utils import ensure_bgr_u8

Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]  # ((cx,cy),(major,minor), angle_deg)

LOW_GREEN = (40, 60, 60)
HIGH_GREEN = (80, 255, 255)
LOW_RED_1 = (0, 60, 60)
HIGH_RED_1 = (10, 255, 255)
LOW_RED_2 = (170, 60, 60)
HIGH_RED_2 = (180, 255, 255)

MIN_AREA_RATIO = 0.02
MAX_CENTER_OFFSET = 0.18
MORPH_KERNEL = 5
CLOSE_ITERS = 2


def _odd_kernel_size(k: int) -> int:
    k = max(1, int(k))
    return k + 1 if (k % 2 == 0) else k


_LOW_GREEN = np.array(LOW_GREEN, dtype=np.uint8)
_HIGH_GREEN = np.array(HIGH_GREEN, dtype=np.uint8)
_LOW_RED_1 = np.array(LOW_RED_1, dtype=np.uint8)
_HIGH_RED_1 = np.array(HIGH_RED_1, dtype=np.uint8)
_LOW_RED_2 = np.array(LOW_RED_2, dtype=np.uint8)
_HIGH_RED_2 = np.array(HIGH_RED_2, dtype=np.uint8)

_MORPH_KERNEL_ODD = _odd_kernel_size(MORPH_KERNEL)
_MORPH_CLOSE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (_MORPH_KERNEL_ODD, _MORPH_KERNEL_ODD)
)


def detect_outer_ellipse(img_bgr: np.ndarray) -> Optional[Ellipse]:
    """
    Detect the best outer ellipse on a dartboard by:
      1) building a red+green HSV mask internally,
      2) closing it morphologically,
      3) fitting ellipses on external contours and selecting the best candidate.

    Input:
      - img_bgr: BGR uint8 image (will be normalized via ensure_bgr_u8)

    Output:
      - Ellipse ((cx, cy), (major, minor), angle_deg) or None
    """
    img_bgr = ensure_bgr_u8(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, _LOW_GREEN, _HIGH_GREEN)
    red1 = cv2.inRange(hsv, _LOW_RED_1, _HIGH_RED_1)
    red2 = cv2.inRange(hsv, _LOW_RED_2, _HIGH_RED_2)

    mask_u8 = cv2.bitwise_or(green, red1)
    mask_u8 = cv2.bitwise_or(mask_u8, red2)

    if CLOSE_ITERS > 0:
        mask_u8 = cv2.morphologyEx(
            mask_u8, cv2.MORPH_CLOSE, _MORPH_CLOSE_KERNEL, iterations=CLOSE_ITERS
        )

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = mask_u8.shape[:2]
    min_area = float(h * w) * MIN_AREA_RATIO
    max_offset = float(min(w, h)) * MAX_CENTER_OFFSET
    cx0, cy0 = (w * 0.5), (h * 0.5)

    best_area = -1.0
    best_ellipse: Optional[Ellipse] = None

    for contour in contours:
        if contour.shape[0] < 5:
            continue

        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        (cx, cy), (a1, a2), angle = cv2.fitEllipse(contour)
        if a1 <= 0.0 or a2 <= 0.0:
            continue
        if math.hypot(cx - cx0, cy - cy0) > max_offset:
            continue

        if area > best_area:
            best_area = area
            best_ellipse = ((float(cx), float(cy)), (float(a1), float(a2)), float(angle))

    return best_ellipse
