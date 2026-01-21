from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

U8 = np.uint8
ImageInput = Union[str, np.ndarray]  # base64 string (optionally data URL) OR OpenCV BGR image


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


__all__ = ["ImageInput", "TransformParams", "U8"]
