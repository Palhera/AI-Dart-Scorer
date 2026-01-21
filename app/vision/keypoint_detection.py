import math
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.vision.ellipse_detection import build_red_green_mask, detect_outer_ellipse
from app.vision.line_detection import build_white_mask, detect_lines_from_mask
from app.vision.vision_utils import line_border_points

PointF = Tuple[float, float]
Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]

DEDUP_TOL_PX = 3.0


def _line_ellipse_intersections(rho: float, theta: float, ellipse: Ellipse) -> List[Tuple[float, float, float]]:
    (cx, cy), (major, minor), angle_deg = ellipse
    a = float(major) * 0.5
    b = float(minor) * 0.5
    if a <= 0.0 or b <= 0.0:
        return []

    ct = math.cos(theta)
    st = math.sin(theta)
    p0 = np.array([ct * rho, st * rho], dtype=np.float64)
    d = np.array([-st, ct], dtype=np.float64)

    phi = math.radians(angle_deg)
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    rot = np.array([[cphi, sphi], [-sphi, cphi]], dtype=np.float64)

    p0r = rot @ (p0 - np.array([cx, cy], dtype=np.float64))
    dr = rot @ d

    ax = dr[0] / a
    by = dr[1] / b
    cx0 = p0r[0] / a
    cy0 = p0r[1] / b

    qa = ax * ax + by * by
    qb = 2.0 * (cx0 * ax + cy0 * by)
    qc = cx0 * cx0 + cy0 * cy0 - 1.0

    if abs(qa) < 1e-12:
        return []

    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return []

    sqrt_disc = math.sqrt(max(0.0, disc))
    t1 = (-qb - sqrt_disc) / (2.0 * qa)
    t2 = (-qb + sqrt_disc) / (2.0 * qa)

    inv_rot = np.array([[cphi, -sphi], [sphi, cphi]], dtype=np.float64)
    pts: List[Tuple[float, float, float]] = []
    for t in (t1, t2):
        p = p0r + t * dr
        pw = inv_rot @ p + np.array([cx, cy], dtype=np.float64)
        pts.append((float(pw[0]), float(pw[1]), float(t)))
    return pts


def _dedupe_points(points: Iterable[PointF], tol_px: float) -> List[PointF]:
    out: List[PointF] = []
    tol2 = tol_px * tol_px
    for x, y in points:
        keep = True
        for ox, oy in out:
            if (x - ox) * (x - ox) + (y - oy) * (y - oy) <= tol2:
                keep = False
                break
        if keep:
            out.append((x, y))
    return out


def _filter_in_bounds(points: Sequence[PointF], width: int, height: int) -> List[PointF]:
    return [(x, y) for x, y in points if 0.0 <= x < float(width) and 0.0 <= y < float(height)]


def _draw_overlay(
    img_bgr: np.ndarray,
    points: Sequence[PointF],
    lines: Sequence[dict],
    ellipse: Optional[Ellipse],
    *,
    draw_points: bool,
    draw_lines: bool,
    draw_ellipse: bool,
) -> np.ndarray:
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]

    if draw_lines:
        for line in lines:
            rho = float(line["rho"])
            theta = float(line["theta_rad"])
            pts = line_border_points(rho, theta, w, h)
            if len(pts) == 2:
                cv2.line(overlay, pts[0], pts[1], (0, 255, 0), 1)

    if draw_ellipse and ellipse is not None:
        (cx, cy), (major, minor), angle = ellipse
        axes = (int(round(major * 0.5)), int(round(minor * 0.5)))
        center = (int(round(cx)), int(round(cy)))
        if axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(overlay, center, axes, float(angle), 0, 360, (0, 0, 255), 1)

    if draw_points:
        for x, y in points:
            cv2.circle(overlay, (int(round(x)), int(round(y))), 4, (0, 255, 255), -1)

    return overlay


def compute_keypoints(
    img_bgr: np.ndarray,
    *,
    overlay_points: bool = True,
    overlay_lines: bool = False,
    overlay_circles: bool = False,
) -> Optional[np.ndarray]:
    if img_bgr is None:
        return None

    white_mask = build_white_mask(img_bgr)
    lines = detect_lines_from_mask(white_mask).get("lines", [])

    rg_mask = build_red_green_mask(img_bgr)
    ellipse = detect_outer_ellipse(rg_mask)

    points: List[PointF] = []
    if ellipse is not None:
        for line in lines:
            rho = float(line["rho"])
            theta = float(line["theta_rad"])
            intersections = _line_ellipse_intersections(rho, theta, ellipse)
            points.extend([(x, y) for x, y, _t in intersections])

    h, w = img_bgr.shape[:2]
    points = _filter_in_bounds(points, w, h)
    points = _dedupe_points(points, DEDUP_TOL_PX)

    return _draw_overlay(
        img_bgr,
        points,
        lines,
        ellipse,
        draw_points=overlay_points,
        draw_lines=overlay_lines,
        draw_ellipse=overlay_circles,
    )
