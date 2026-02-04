import math

import cv2
import numpy as np

# Physical dartboard dimensions (mm). These define the canonical geometry used for calibration/warping.
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
REFERENCE_COLOR = (255, 0, 0)
REFERENCE_THICKNESS = 1

# Rotation compensates for how the "canonical" sector directions are defined vs. how the reference art is oriented.
REFERENCE_ROTATION_DEG = 9.0
REFERENCE_ROTATION_RAD = -math.radians(REFERENCE_ROTATION_DEG)

# Canonical reference image size used across the pipeline (warp, ECC, overlays).
REFERENCE_OUTPUT_SIZE = 720

# Ten unique line directions modulo pi (each direction covers a pair of opposite radial lines).
CANONICAL_ANGLES = [i * math.pi / 10.0 for i in range(10)]


def draw_reference_overlay(img_bgr: np.ndarray) -> np.ndarray:
    # Draws the canonical board geometry in the *current image coordinates*.
    # Used for visual debugging after warping into reference space.
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

        # Draw both directions (angle and angle+pi) explicitly so the overlay shows full diameters.
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
