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


def build_red_green_mask(img_bgr: np.ndarray) -> np.ndarray:
    # HSV thresholding to isolate typical dartboard red/green sectors.
    # The outer ellipse is fitted on these regions because they tend to form a strong ring.
    img_bgr = ensure_bgr_u8(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, np.array(LOW_GREEN, dtype=np.uint8), np.array(HIGH_GREEN, dtype=np.uint8))
    red1 = cv2.inRange(hsv, np.array(LOW_RED_1, dtype=np.uint8), np.array(HIGH_RED_1, dtype=np.uint8))
    red2 = cv2.inRange(hsv, np.array(LOW_RED_2, dtype=np.uint8), np.array(HIGH_RED_2, dtype=np.uint8))

    return cv2.bitwise_or(green, cv2.bitwise_or(red1, red2))


def detect_outer_ellipse(mask_u8: np.ndarray) -> Optional[Ellipse]:
    # Finds the "best" ellipse candidate from red/green regions.
    # Heuristics prefer: large area + centered near the image center (typical board framing).
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)

    k = max(1, int(MORPH_KERNEL))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Closing helps connect fragmented sector blobs into a more continuous ring.
    if CLOSE_ITERS > 0:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS)

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
        # cv2.fitEllipse requires at least 5 points.
        if contour.shape[0] < 5:
            continue

        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        (cx, cy), (a1, a2), angle = cv2.fitEllipse(contour)
        if a1 <= 0.0 or a2 <= 0.0:
            continue

        # Reject off-center candidates (common failure mode: background objects with red/green).
        if math.hypot(cx - cx0, cy - cy0) > max_offset:
            continue

        if area > best_area:
            best_area = area
            best_ellipse = ((float(cx), float(cy)), (float(a1), float(a2)), float(angle))

    return best_ellipse
