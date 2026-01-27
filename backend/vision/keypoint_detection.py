import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from backend.vision.ellipse_detection import build_red_green_mask, detect_outer_ellipse
from backend.vision.homography import warp_to_reference_with_matrix
from backend.vision.line_detection import build_white_mask, detect_lines_from_mask
from backend.vision.reference import draw_reference_overlay

PointF = Tuple[float, float]
Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]


def _line_ellipse_intersections(rho: float, theta: float, ellipse: Ellipse) -> List[PointF]:
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
    pts: List[PointF] = []
    for t in (t1, t2):
        p = p0r + t * dr
        pw = inv_rot @ p + np.array([cx, cy], dtype=np.float64)
        pts.append((float(pw[0]), float(pw[1])))
    return pts


def _collect_line_intersections(lines: Sequence[dict], ellipse: Ellipse) -> List[dict]:
    lines_with_points: List[dict] = []
    for line in lines:
        rho = float(line["rho"])
        theta = float(line["theta_rad"])
        pts = _line_ellipse_intersections(rho, theta, ellipse)
        if len(pts) < 2:
            continue
        phi = (theta + math.pi / 2.0) % math.pi
        lines_with_points.append(
            {
                "rho": rho,
                "theta": theta,
                "phi": phi,
                "points": pts,
                "accuracy": float(line.get("accuracy", 1.0)),
            }
        )
    return lines_with_points


def compute_keypoints(img_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    if img_bgr is None:
        return None

    white_mask = build_white_mask(img_bgr)
    lines = detect_lines_from_mask(white_mask).get("lines", [])

    rg_mask = build_red_green_mask(img_bgr)
    ellipse = detect_outer_ellipse(rg_mask)
    if ellipse is None or not lines:
        return img_bgr.copy(), None

    lines_with_points = _collect_line_intersections(lines, ellipse)
    if not lines_with_points:
        return img_bgr.copy(), None

    (cx, cy), _axes, _angle = ellipse
    center = np.array([cx, cy], dtype=np.float64)

    result = warp_to_reference_with_matrix(img_bgr, lines_with_points, center)
    if result is None:
        return img_bgr.copy(), None

    warped, total_matrix = result

    return draw_reference_overlay(warped), total_matrix
