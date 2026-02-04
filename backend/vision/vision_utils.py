import math
from typing import List, Tuple

import numpy as np

PointI = Tuple[int, int]


def ensure_bgr_u8(img_bgr: np.ndarray) -> np.ndarray:
    # Defensive normalization for OpenCV pipelines:
    # ensures a 3-channel BGR image with uint8 storage, which most OpenCV ops assume.
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H, W, 3), got {img_bgr.shape}.")
    if img_bgr.dtype != np.uint8:
        # Conversion is cheap and avoids subtle OpenCV misbehavior with other dtypes.
        img_bgr = img_bgr.astype(np.uint8, copy=False)
    return img_bgr


def line_border_points(rho: float, theta: float, width: int, height: int) -> List[PointI]:
    """
    Compute up to two intersection points between a Hough line (rho, theta)
    and the rectangular image border.

    This is used to convert an infinite Hough line into a finite segment
    spanning the image, suitable for sampling or visualization.
    """
    pts: List[PointI] = []
    ct, st = math.cos(theta), math.sin(theta)

    # Intersections with vertical borders: x = 0 and x = width - 1
    if abs(st) > 1e-6:
        y0 = (rho - 0.0 * ct) / st
        y1 = (rho - (width - 1.0) * ct) / st
        if 0.0 <= y0 <= height - 1.0:
            pts.append((0, int(round(y0))))
        if 0.0 <= y1 <= height - 1.0:
            pts.append((width - 1, int(round(y1))))

    # Intersections with horizontal borders: y = 0 and y = height - 1
    if abs(ct) > 1e-6:
        x0 = (rho - 0.0 * st) / ct
        x1 = (rho - (height - 1.0) * st) / ct
        if 0.0 <= x0 <= width - 1.0:
            pts.append((int(round(x0)), 0))
        if 0.0 <= x1 <= width - 1.0:
            pts.append((int(round(x1)), height - 1))

    # Deduplicate and return at most two points (a line segment).
    uniq: List[PointI] = []
    for p in pts:
        if p not in uniq:
            uniq.append(p)
    return uniq[:2]
