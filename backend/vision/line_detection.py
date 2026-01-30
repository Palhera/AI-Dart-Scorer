from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.vision.vision_utils import ensure_bgr_u8, line_border_points

LineRT = Tuple[float, float]

MAX_LINES = 10
CANNY_LOW = 40
CANNY_HIGH = 120
HOUGH_RHO = 1.0
HOUGH_THETA = float(np.pi / 180.0)
HOUGH_MIN_VOTES = 80
HOUGH_VOTES_FRAC = 0.2
UNIQUE_RHO_TOL = 20.0
UNIQUE_THETA_TOL = float(np.deg2rad(6.0))

MASK_KERNEL = 3
OPEN_ITERS = 1
CLOSE_ITERS = 2
SUPPORT_SAMPLES = 160


def build_white_mask(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = ensure_bgr_u8(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask_u8 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    k = max(1, int(MASK_KERNEL))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if OPEN_ITERS > 0:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITERS)
    if CLOSE_ITERS > 0:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS)

    return mask_u8


def _select_unique_lines(raw_lines: Optional[np.ndarray]) -> List[LineRT]:
    if raw_lines is None:
        return []

    selected: List[LineRT] = []
    for rho, theta in raw_lines[:, 0]:
        if len(selected) >= MAX_LINES:
            break
        r, t = float(rho), float(theta)
        if any(abs(r - sr) <= UNIQUE_RHO_TOL and abs(t - st) <= UNIQUE_THETA_TOL for sr, st in selected):
            continue
        selected.append((r, t))
    return selected


def _line_support(edges: np.ndarray, rho: float, theta: float) -> float:
    h, w = edges.shape[:2]
    pts = line_border_points(rho, theta, w, h)
    if len(pts) < 2:
        return 0.0

    (x0, y0), (x1, y1) = pts
    length = max(abs(x1 - x0), abs(y1 - y0))
    samples = max(SUPPORT_SAMPLES, int(length))

    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    xs = np.rint(x0 + (x1 - x0) * t).astype(np.int32)
    ys = np.rint(y0 + (y1 - y0) * t).astype(np.int32)

    valid = (0 <= xs) & (xs < w) & (0 <= ys) & (ys < h)
    if not np.any(valid):
        return 0.0

    hits = edges[ys[valid], xs[valid]] > 0
    return float(hits.sum()) / float(valid.sum())


def detect_lines_from_mask(mask_u8: np.ndarray) -> dict:
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(mask_u8, CANNY_LOW, CANNY_HIGH)
    h, w = mask_u8.shape[:2]
    votes = max(HOUGH_MIN_VOTES, int(min(h, w) * HOUGH_VOTES_FRAC))

    raw_lines = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, votes)
    lines = _select_unique_lines(raw_lines)

    lines_out = []
    for rho, theta in lines:
        accuracy = _line_support(edges, rho, theta)
        lines_out.append(
            {
                "rho": float(rho),
                "theta_rad": float(theta),
                "accuracy": float(accuracy),
            }
        )

    return {"lines": lines_out}
