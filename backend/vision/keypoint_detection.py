import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from backend.vision.ellipse_detection import build_red_green_mask, detect_outer_ellipse
from backend.vision.homography import warp_to_reference_with_matrix
from backend.vision.line_detection import build_white_mask, detect_lines_from_mask
from backend.vision.reference import REFERENCE_OUTPUT_SIZE, draw_reference_overlay

PointF = Tuple[float, float]
Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]


def _line_ellipse_intersections(rho: float, theta: float, ellipse: Ellipse) -> List[PointF]:
    # Compute intersections between a Hough line (rho, theta) and a rotated ellipse.
    # Used to convert detected "board lines" into concrete point constraints on the board boundary.
    (cx, cy), (major, minor), angle_deg = ellipse
    a = float(major) * 0.5
    b = float(minor) * 0.5
    if a <= 0.0 or b <= 0.0:
        return []

    ct = math.cos(theta)
    st = math.sin(theta)
    p0 = np.array([ct * rho, st * rho], dtype=np.float64)
    d = np.array([-st, ct], dtype=np.float64)

    # Transform into ellipse-aligned coordinates, solve intersection with unit ellipse,
    # then transform back to image coordinates.
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
    # Enrich each detected line with its two boundary intersection points on the outer ellipse.
    # The downstream homography solver expects line constraints expressed via points.
    lines_with_points: List[dict] = []
    for line in lines:
        rho = float(line["rho"])
        theta = float(line["theta_rad"])
        pts = _line_ellipse_intersections(rho, theta, ellipse)
        if len(pts) < 2:
            continue
        # "phi" is the line direction angle (normal theta rotated by 90°), normalized to [0, pi).
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


def _compute_warp_result(img_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # Core pipeline:
    # 1) detect white board lines
    # 2) detect outer board ellipse from red/green regions
    # 3) convert lines to intersection constraints with ellipse
    # 4) estimate warp into a canonical "reference" view + return the homography
    if img_bgr is None:
        return None

    white_mask = build_white_mask(img_bgr)
    lines = detect_lines_from_mask(white_mask).get("lines", [])

    rg_mask = build_red_green_mask(img_bgr)
    ellipse = detect_outer_ellipse(rg_mask)
    if ellipse is None or not lines:
        return None

    lines_with_points = _collect_line_intersections(lines, ellipse)
    if not lines_with_points:
        return None

    (cx, cy), _axes, _angle = ellipse
    center = np.array([cx, cy], dtype=np.float64)

    # Output size is capped to keep computation predictable while staying within the
    # input image bounds. This also implicitly defines the coordinate system used
    # by the returned homography matrix.
    output_size = min(
        int(img_bgr.shape[0]),
        int(img_bgr.shape[1]),
        REFERENCE_OUTPUT_SIZE,
    )
    return warp_to_reference_with_matrix(
        img_bgr,
        lines_with_points,
        center,
        output_size=output_size,
    )


def compute_homography_matrix(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    # Public API for calibration: return only the estimated homography (if available).
    result = _compute_warp_result(img_bgr)
    if result is None:
        return None
    _warped, total_matrix = result
    return total_matrix


def compute_keypoints(img_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    # Public API for UI/debug: return an annotated "reference view" image plus the homography.
    if img_bgr is None:
        return None

    result = _compute_warp_result(img_bgr)
    if result is None:
        return img_bgr.copy(), None

    warped, total_matrix = result
    return draw_reference_overlay(warped), total_matrix
