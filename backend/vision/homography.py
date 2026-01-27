import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

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
EccCache = Tuple[np.ndarray, np.ndarray]

ECC_CANNY_LOW = 40
ECC_CANNY_HIGH = 140
ECC_MASK_MARGIN_MM = 2.0
ECC_MAX_ITERS = 80
ECC_EPS = 1e-6
ECC_GAUSS_SIZE = 5

_ECC_CACHE: Dict[int, EccCache] = {}


def _build_ecc_mask(output_size: int) -> np.ndarray:
    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    center = (int(round((output_size - 1) * 0.5)), int(round((output_size - 1) * 0.5)))
    radius_mm = REFERENCE_LINE_OUTER_MM + ECC_MASK_MARGIN_MM
    radius_px = int(round(radius_mm * output_size / BOARD_DIAMETER_MM))
    if radius_px > 0:
        cv2.circle(mask, center, radius_px, 255, thickness=-1)
    return mask


def _prepare_ecc_image(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, ECC_CANNY_LOW, ECC_CANNY_HIGH)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    return edges.astype(np.float32) / 255.0


def _get_reference_ecc(output_size: int) -> Optional[EccCache]:
    cached = _ECC_CACHE.get(output_size)
    if cached is not None:
        return cached

    ref_path = os.path.join(os.path.dirname(__file__), "reference.png")
    ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    if ref is None:
        return None

    if ref.shape[0] != output_size or ref.shape[1] != output_size:
        ref = cv2.resize(ref, (output_size, output_size), interpolation=cv2.INTER_AREA)

    ref_edges = _prepare_ecc_image(ref)
    mask = _build_ecc_mask(output_size)
    _ECC_CACHE[output_size] = (ref_edges, mask)
    return _ECC_CACHE[output_size]


def _refine_with_ecc(warped_bgr: np.ndarray, output_size: int) -> Tuple[np.ndarray, np.ndarray]:
    ecc_ref = _get_reference_ecc(output_size)
    if ecc_ref is None:
        return warped_bgr, np.eye(3, dtype=np.float64)

    ref_edges, mask = ecc_ref
    moving_edges = _prepare_ecc_image(warped_bgr)

    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITERS, ECC_EPS)

    try:
        _cc, warp_matrix = cv2.findTransformECC(
            ref_edges,
            moving_edges,
            warp_matrix,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            inputMask=mask,
            gaussFiltSize=ECC_GAUSS_SIZE,
        )
    except cv2.error:
        return warped_bgr, np.eye(3, dtype=np.float64)

    refined = cv2.warpPerspective(
        warped_bgr,
        warp_matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    try:
        ecc_forward = np.linalg.inv(warp_matrix.astype(np.float64))
    except np.linalg.LinAlgError:
        ecc_forward = np.eye(3, dtype=np.float64)
    return refined, ecc_forward


def warp_to_reference_with_matrix(
    img_bgr: np.ndarray,
    lines_with_points: Sequence[dict],
    center: np.ndarray,
    output_size: int = REFERENCE_OUTPUT_SIZE,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    homography = compute_homography(lines_with_points, center, output_size=output_size)
    if homography is None:
        return None

    warped = cv2.warpPerspective(img_bgr, homography, (output_size, output_size))
    refined, ecc_forward = _refine_with_ecc(warped, output_size)
    total = ecc_forward @ homography.astype(np.float64)
    return refined, total


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
    result = warp_to_reference_with_matrix(
        img_bgr,
        lines_with_points,
        center,
        output_size=output_size,
    )
    if result is None:
        return None
    return result[0]
