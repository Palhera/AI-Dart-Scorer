from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

U8 = np.uint8


@dataclass(frozen=True, slots=True)
class TransformParams:
    """Tunable parameters for mask extraction."""
    # White mask cleanup
    min_area_ratio: float = 0.0015
    morph_kernel: int = 3
    open_iters: int = 1
    close_iters: int = 2

    # HSV thresholds (OpenCV HSV ranges: H=[0..179], S,V=[0..255])
    low_green: Tuple[int, int, int] = (40, 60, 60)
    high_green: Tuple[int, int, int] = (80, 255, 255)

    low_red_1: Tuple[int, int, int] = (0, 60, 60)
    high_red_1: Tuple[int, int, int] = (10, 255, 255)
    low_red_2: Tuple[int, int, int] = (170, 60, 60)
    high_red_2: Tuple[int, int, int] = (180, 255, 255)

    # Safety / performance knobs
    min_neutral_samples: int = 200


# ---------------------------
# Internal utilities
# ---------------------------

def _kernel(params: TransformParams) -> np.ndarray:
    k = max(1, int(params.morph_kernel))
    if k == 1:
        # Morphology with 1x1 is a no-op; still return a valid kernel.
        return np.ones((1, 1), dtype=U8)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _ensure_bgr_u8(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H, W, 3), got {img_bgr.shape}.")
    if img_bgr.dtype != U8:
        # OpenCV works best with uint8 for these operations.
        img_bgr = img_bgr.astype(U8, copy=False)
    return img_bgr


def _to_3ch(mask_u8: np.ndarray) -> np.ndarray:
    """Convert single-channel uint8 mask to 3-channel BGR mask."""
    if mask_u8.ndim != 2:
        raise ValueError(f"Expected single-channel mask, got {mask_u8.shape}.")
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def _filter_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components with area < min_area. mask_u8 must be 0/255."""
    if min_area <= 0:
        return mask_u8

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    # Labels start at 1; 0 is background.
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_labels = (np.where(areas >= min_area)[0] + 1).astype(np.int32)

    if keep_labels.size == 0:
        return np.zeros_like(mask_u8)

    # Vectorized membership test. For many labels, using a boolean LUT is faster.
    lut = np.zeros(num_labels, dtype=U8)
    lut[keep_labels] = 1
    kept = lut[labels]  # 0/1
    return (kept * 255).astype(U8, copy=False)


# ---------------------------
# Public API
# ---------------------------

def build_white_mask(img_bgr: np.ndarray, params: Optional[TransformParams] = None) -> np.ndarray:
    """
    Build a 3-channel (BGR) binary mask for "white" / high-luminance neutral regions.

    Approach:
      1) Convert BGR -> LAB
      2) Identify near-neutral pixels using chroma distance to (a=128,b=128) with Otsu
      3) Compute an Otsu threshold on L over neutral pixels to separate brighter neutrals
      4) Morphological open+close
      5) Remove small connected components (area threshold relative to image size)
    """
    params = params or TransformParams()
    img_bgr = _ensure_bgr_u8(img_bgr)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]  # uint8
    a = lab[:, :, 1].astype(np.int16)
    b = lab[:, :, 2].astype(np.int16)

    # Squared chroma distance to neutral (128,128); avoid sqrt for speed.
    da = a - 128
    db = b - 128
    chroma2 = (da * da + db * db).astype(np.uint16)  # fits: max ~ 2*(127^2)=32258

    # Map to uint8 for Otsu; using sqrt is unnecessary; monotonic transform is fine.
    # Scale down to [0..255] approximately by right-shifting.
    # 32258 >> 7 ~= 252
    chroma_u8 = (chroma2 >> 7).astype(U8)

    # Neutral pixels: low chroma
    _, neutral_mask = cv2.threshold(chroma_u8, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    neutral = neutral_mask.astype(bool)

    neutral_l = l[neutral]
    if neutral_l.size < params.min_neutral_samples:
        # Fallback: use the full image if neutral sample is insufficient.
        neutral_l = l.reshape(-1)

    # Otsu on luminance to get "white" cutoff (high L)
    # Provide a 1D array as a single-column image.
    neutral_l_1c = neutral_l.reshape(-1, 1)
    white_thresh, _ = cv2.threshold(neutral_l_1c, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    mask_u8 = np.zeros_like(l, dtype=U8)
    mask_u8[(l >= white_thresh) & neutral] = 255

    k = _kernel(params)
    if params.open_iters > 0:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=int(params.open_iters))
    if params.close_iters > 0:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=int(params.close_iters))

    min_area = int(mask_u8.size * float(params.min_area_ratio))
    mask_u8 = _filter_small_components(mask_u8, min_area=min_area)

    return _to_3ch(mask_u8)


def build_red_green_mask(img_bgr: np.ndarray, params: Optional[TransformParams] = None) -> np.ndarray:
    """
    Build a 3-channel (BGR) binary mask for red and green regions in HSV space.
    """
    params = params or TransformParams()
    img_bgr = _ensure_bgr_u8(img_bgr)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    low_green = np.array(params.low_green, dtype=U8)
    high_green = np.array(params.high_green, dtype=U8)
    low_red_1 = np.array(params.low_red_1, dtype=U8)
    high_red_1 = np.array(params.high_red_1, dtype=U8)
    low_red_2 = np.array(params.low_red_2, dtype=U8)
    high_red_2 = np.array(params.high_red_2, dtype=U8)

    green = cv2.inRange(hsv, low_green, high_green)
    red1 = cv2.inRange(hsv, low_red_1, high_red_1)
    red2 = cv2.inRange(hsv, low_red_2, high_red_2)

    mask_u8 = cv2.bitwise_or(green, cv2.bitwise_or(red1, red2))
    return _to_3ch(mask_u8)
