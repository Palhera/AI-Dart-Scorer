import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.vision.reference import (
    BOARD_DIAMETER_MM,
    CANONICAL_ANGLES,
    REFERENCE_LINE_OUTER_MM,
    REFERENCE_OUTPUT_SIZE,
    REFERENCE_ROTATION_RAD,
)

PointF = Tuple[float, float]


def _angle_diff(a: float, b: float) -> float:
    return abs(((a - b + math.pi / 2.0) % math.pi) - (math.pi / 2.0))


def _best_offset(lines: Sequence[dict]) -> float:
    offsets: List[float] = []
    for line in lines:
        phi = float(line["phi"])
        for angle in CANONICAL_ANGLES:
            offsets.append((phi - angle) % math.pi)

    if not offsets:
        return 0.0

    best_offset = offsets[0]
    best_score = float("inf")
    for offset in offsets:
        score = 0.0
        for line in lines:
            phi = float(line["phi"])
            accuracy = float(line.get("accuracy", 1.0))
            diffs = [_angle_diff(phi, angle + offset) for angle in CANONICAL_ANGLES]
            score += accuracy * min(diffs) ** 2
        if score < best_score:
            best_score = score
            best_offset = offset
    return best_offset


def _assign_canonical_indices(lines: Sequence[dict]) -> List[int]:
    offset = _best_offset(lines)
    indices: List[int] = []
    for line in lines:
        phi = float(line["phi"])
        diffs = [_angle_diff(phi, angle + offset) for angle in CANONICAL_ANGLES]
        indices.append(int(np.argmin(diffs)))
    return indices


def _build_correspondences(
    lines_with_points: Sequence[dict],
    center: np.ndarray,
    output_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = _assign_canonical_indices(lines_with_points)
    radius = REFERENCE_LINE_OUTER_MM * output_size / BOARD_DIAMETER_MM
    center_dst = (output_size * 0.5, output_size * 0.5)

    src_pts: List[PointF] = []
    dst_pts: List[PointF] = []
    for line, idx in zip(lines_with_points, indices):
        pts = line["points"]
        phi = float(line["phi"])
        d = np.array([math.cos(phi), math.sin(phi)], dtype=np.float64)
        dots = [float(np.dot(np.array(p, dtype=np.float64) - center, d)) for p in pts]
        plus_idx = int(np.argmax(dots))
        minus_idx = 1 - plus_idx
        plus = pts[plus_idx]
        minus = pts[minus_idx]

        angle = CANONICAL_ANGLES[idx] + REFERENCE_ROTATION_RAD
        dst_plus = (
            center_dst[0] + radius * math.cos(angle),
            center_dst[1] + radius * math.sin(angle),
        )
        dst_minus = (
            center_dst[0] + radius * math.cos(angle + math.pi),
            center_dst[1] + radius * math.sin(angle + math.pi),
        )

        repeats = 1 + int(round(float(line.get("accuracy", 1.0)) * 2.0))
        for _ in range(repeats):
            src_pts.append(plus)
            dst_pts.append(dst_plus)
            src_pts.append(minus)
            dst_pts.append(dst_minus)

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    return src, dst


def compute_homography(
    lines_with_points: Sequence[dict],
    center: np.ndarray,
    output_size: int = REFERENCE_OUTPUT_SIZE,
) -> Optional[np.ndarray]:
    if not lines_with_points:
        return None

    src, dst = _build_correspondences(lines_with_points, center, output_size)
    if len(src) < 4:
        return None

    homography, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return homography


def warp_to_reference(
    img_bgr: np.ndarray,
    lines_with_points: Sequence[dict],
    center: np.ndarray,
    output_size: int = REFERENCE_OUTPUT_SIZE,
) -> Optional[np.ndarray]:
    homography = compute_homography(lines_with_points, center, output_size=output_size)
    if homography is None:
        return None

    return cv2.warpPerspective(img_bgr, homography, (output_size, output_size))
