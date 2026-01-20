from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.vision.ellipse_detection import EllipseDetectConfig, detect_outer_ellipse
from app.vision.image_transform import TransformParams, build_red_green_mask, build_white_mask
from app.vision.line_detection import DetectConfig, detect_lines_from_mask

PointF = Tuple[float, float]
Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]


@dataclass(frozen=True, slots=True)
class KeypointConfig:
    """Configuration for keypoint selection/deduplication."""
    dedupe_tol_px: float = 3.0
    one_per_line: bool = False


def _line_ellipse_intersections(rho: float, theta: float, ellipse: Ellipse) -> List[Tuple[float, float, float]]:
    """
    Return intersection points between a polar-form line and a rotated ellipse.

    Output points are (x, y, t) where t is the line parameter.
    """
    (cx, cy), (major, minor), angle_deg = ellipse
    a = float(major) * 0.5
    b = float(minor) * 0.5
    if a <= 0.0 or b <= 0.0:
        return []

    # Line in point + direction form.
    ct = math.cos(theta)
    st = math.sin(theta)
    p0 = np.array([ct * rho, st * rho], dtype=np.float64)
    d = np.array([-st, ct], dtype=np.float64)

    # Rotate into ellipse coordinates (x', y').
    phi = math.radians(angle_deg)
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    rot = np.array([[cphi, sphi], [-sphi, cphi]], dtype=np.float64)

    p0r = rot @ (p0 - np.array([cx, cy], dtype=np.float64))
    dr = rot @ d

    # Quadratic solve for (x'/a)^2 + (y'/b)^2 = 1.
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

    pts: List[Tuple[float, float, float]] = []
    for t in (t1, t2):
        p = p0r + t * dr
        # Rotate back to image coordinates.
        inv_rot = np.array([[cphi, -sphi], [sphi, cphi]], dtype=np.float64)
        pw = inv_rot @ p + np.array([cx, cy], dtype=np.float64)
        pts.append((float(pw[0]), float(pw[1]), float(t)))
    return pts


def _dedupe_points(points: Iterable[PointF], tol_px: float) -> List[PointF]:
    out: List[PointF] = []
    tol2 = tol_px * tol_px
    for x, y in points:
        if not out:
            out.append((x, y))
            continue
        keep = True
        for ox, oy in out:
            if (x - ox) * (x - ox) + (y - oy) * (y - oy) <= tol2:
                keep = False
                break
        if keep:
            out.append((x, y))
    return out


def _filter_in_bounds(points: Sequence[PointF], width: int, height: int) -> List[PointF]:
    kept: List[PointF] = []
    for x, y in points:
        if 0.0 <= x < float(width) and 0.0 <= y < float(height):
            kept.append((x, y))
    return kept


def _line_border_points(rho: float, theta: float, width: int, height: int) -> List[Tuple[int, int]]:
    """Return up to 2 intersection points between the line and the image border."""
    pts: List[Tuple[int, int]] = []
    ct, st = math.cos(theta), math.sin(theta)

    if abs(st) > 1e-6:
        y0 = (rho - 0.0 * ct) / st
        y1 = (rho - (width - 1.0) * ct) / st
        if 0.0 <= y0 <= height - 1.0:
            pts.append((0, int(round(y0))))
        if 0.0 <= y1 <= height - 1.0:
            pts.append((width - 1, int(round(y1))))

    if abs(ct) > 1e-6:
        x0 = (rho - 0.0 * st) / ct
        x1 = (rho - (height - 1.0) * st) / ct
        if 0.0 <= x0 <= width - 1.0:
            pts.append((int(round(x0)), 0))
        if 0.0 <= x1 <= width - 1.0:
            pts.append((int(round(x1)), height - 1))

    uniq: List[Tuple[int, int]] = []
    for p in pts:
        if p not in uniq:
            uniq.append(p)
    return uniq[:2]


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
            pts = _line_border_points(rho, theta, w, h)
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
    params: Optional[TransformParams] = None,
    line_cfg: Optional[DetectConfig] = None,
    ellipse_cfg: Optional[EllipseDetectConfig] = None,
    kp_cfg: Optional[KeypointConfig] = None,
    overlay_points: bool = True,
    overlay_lines: bool = False,
    overlay_circles: bool = False,
) -> dict:
    """
    Pipeline:
      1) Apply white mask to the original image
      2) Detect lines on the white mask
      3) Apply red/green mask to the original image
      4) Detect the outer ellipse on the red/green mask
      5) Intersect lines with ellipse to get keypoints
      6) Optional overlay for points, lines, and ellipse
    """
    if img_bgr is None:
        return {"keypoints": [], "overlay": None}

    params = params or TransformParams()
    line_cfg = line_cfg or DetectConfig()
    ellipse_cfg = ellipse_cfg or EllipseDetectConfig()
    kp_cfg = kp_cfg or KeypointConfig()

    white_mask = build_white_mask(img_bgr, params=params)
    line_result = detect_lines_from_mask(white_mask, img_bgr, cfg=line_cfg)
    lines = line_result.get("lines", [])

    rg_mask = build_red_green_mask(img_bgr, params=params)
    ellipse = detect_outer_ellipse(rg_mask, cfg=ellipse_cfg)

    points: List[PointF] = []
    if ellipse is not None:
        for line in lines:
            rho = float(line["rho"])
            theta = float(line["theta_rad"])
            intersections = _line_ellipse_intersections(rho, theta, ellipse)
            if not intersections:
                continue
            if kp_cfg.one_per_line:
                # Choose the intersection in the positive line direction (largest t).
                x, y, _t = max(intersections, key=lambda item: item[2])
                points.append((x, y))
            else:
                points.extend([(x, y) for x, y, _t in intersections])

    h, w = img_bgr.shape[:2]
    points = _filter_in_bounds(points, w, h)
    points = _dedupe_points(points, kp_cfg.dedupe_tol_px)

    overlay = _draw_overlay(
        img_bgr,
        points,
        lines,
        ellipse,
        draw_points=overlay_points,
        draw_lines=overlay_lines,
        draw_ellipse=overlay_circles,
    )

    return {
        "keypoints": [{"x": float(x), "y": float(y)} for x, y in points],
        "overlay": overlay,
    }
