import math

import cv2
import numpy as np

BOARD_DIAMETER_MM = 451.0
DOUBLE_RING_RADIUS_MM = 170.0
DOUBLE_RING_INNER_MM = 162.0
TRIPLE_RING_OUTER_MM = 107.0
TRIPLE_RING_INNER_MM = 99.0
BULL_OUTER_RADIUS_MM = 15.9
BULL_INNER_RADIUS_MM = 6.35

REFERENCE_RING_RADII_MM = (
    DOUBLE_RING_RADIUS_MM,
    DOUBLE_RING_INNER_MM,
    TRIPLE_RING_OUTER_MM,
    TRIPLE_RING_INNER_MM,
    BULL_OUTER_RADIUS_MM,
    BULL_INNER_RADIUS_MM,
)
REFERENCE_LINE_INNER_MM = BULL_OUTER_RADIUS_MM
REFERENCE_LINE_OUTER_MM = DOUBLE_RING_RADIUS_MM
REFERENCE_COLOR = (255, 255, 255)
REFERENCE_THICKNESS = 1
REFERENCE_ROTATION_DEG = 9.0
REFERENCE_ROTATION_RAD = -math.radians(REFERENCE_ROTATION_DEG)
REFERENCE_OUTPUT_SIZE = int(round(BOARD_DIAMETER_MM))
CANONICAL_ANGLES = [i * math.pi / 10.0 for i in range(10)]


def draw_reference_overlay(img_bgr: np.ndarray) -> np.ndarray:
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]
    size = float(min(h, w) - 1)
    if size <= 0.0:
        return overlay

    scale = size / BOARD_DIAMETER_MM
    center_f = (float(w - 1) * 0.5, float(h - 1) * 0.5)
    center = (int(round(center_f[0])), int(round(center_f[1])))

    for radius_mm in REFERENCE_RING_RADII_MM:
        radius_px = int(round(radius_mm * scale))
        if radius_px > 0:
            cv2.circle(overlay, center, radius_px, REFERENCE_COLOR, REFERENCE_THICKNESS)

    outer_radius_px = int(round(REFERENCE_LINE_OUTER_MM * scale))
    inner_radius_px = int(round(REFERENCE_LINE_INNER_MM * scale))
    for angle in CANONICAL_ANGLES:
        angle_rot = angle + REFERENCE_ROTATION_RAD
        dx = math.cos(angle_rot)
        dy = math.sin(angle_rot)

        x0 = int(round(center_f[0] + dx * inner_radius_px))
        y0 = int(round(center_f[1] + dy * inner_radius_px))
        x1 = int(round(center_f[0] + dx * outer_radius_px))
        y1 = int(round(center_f[1] + dy * outer_radius_px))
        cv2.line(overlay, (x0, y0), (x1, y1), REFERENCE_COLOR, REFERENCE_THICKNESS)

        x2 = int(round(center_f[0] - dx * inner_radius_px))
        y2 = int(round(center_f[1] - dy * inner_radius_px))
        x3 = int(round(center_f[0] - dx * outer_radius_px))
        y3 = int(round(center_f[1] - dy * outer_radius_px))
        cv2.line(overlay, (x2, y2), (x3, y3), REFERENCE_COLOR, REFERENCE_THICKNESS)

    return overlay
