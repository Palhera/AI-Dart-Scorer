from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.vision.vision_types import ImageInput, TransformParams, U8
from app.vision.vision_utils import decode_base64_image, ensure_bgr_u8, line_border_points, to_3ch

# Optional dependency: Hungarian assignment for best matching (preferred).
try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment  # type: ignore
except Exception:
    _linear_sum_assignment = None


F64 = np.float64

# Model: 10 spokes with fixed offsets (pi-periodic angles)
PHI0_DEG: float = 9.0
DELTA_DEG: float = 18.0
PHI_GRID: np.ndarray = (np.deg2rad(PHI0_DEG) + np.deg2rad(DELTA_DEG) * np.arange(10, dtype=F64)) % np.pi

@dataclass(frozen=True, slots=True)
class DetectConfig:
    """Detection tuning parameters (explicit, testable, production-friendly)."""
    max_lines: int = 10

    # Edge + Hough
    canny_low: int = 40
    canny_high: int = 120
    hough_rho: float = 1.0
    hough_theta: float = float(np.pi / 180.0)
    hough_min_votes: int = 80
    hough_votes_frac: float = 0.2  # votes = max(min_votes, min(H,W)*frac)

    # Line selection / de-dup
    unique_rho_tol: float = 20.0
    unique_theta_tol: float = float(np.deg2rad(6.0))

    # Consensus filtering
    consensus_tol_pass1: float = 5.0
    consensus_tol_pass2: float = 8.0

    # Contrast scoring
    color_offset_px: float = 3.0
    color_samples: int = 40

    # Composite scoring weights
    score_color_w: float = 0.7
    score_center_w: float = 0.3
    center_sigma_px: float = 8.0  # scale for center_score distance normalization

    # Inference (grid search)
    fit_fast: bool = True
    fit_max_evals: int = 15_000


# Internal line tuple: (rho, theta, score/confidence, found)
ScoredLine = Tuple[float, float, float, bool]
LineRT = Tuple[float, float]
PointF = Tuple[float, float]
PointI = Tuple[int, int]


# ---------------------------
# Image transform helpers
# ---------------------------

def _kernel(params: TransformParams) -> np.ndarray:
    k = max(1, int(params.morph_kernel))
    if k == 1:
        # Morphology with 1x1 is a no-op; still return a valid kernel.
        return np.ones((1, 1), dtype=U8)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


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
    img_bgr = ensure_bgr_u8(img_bgr)

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

    return to_3ch(mask_u8)


# ---------------------------
# Angle math (mod pi)
# ---------------------------

def _wrap_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) into [0, pi)."""
    return np.mod(angle, np.pi)


def _diff_pi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed difference a-b on a pi-periodic circle, mapped to [-pi/2, +pi/2)."""
    return np.mod(a - b + (np.pi / 2.0), np.pi) - (np.pi / 2.0)


def _predict_angles(k: float, rho: float, beta: float) -> np.ndarray:
    """
    Predict 10 pi-periodic angles using a stable formulation:
      atan(k*tan(x)) = atan2(k*sin(x), cos(x))
    """
    x = PHI_GRID - beta
    return _wrap_pi(rho + np.arctan2(k * np.sin(x), np.cos(x))).astype(F64, copy=False)


# ---------------------------
# Assignment / inference model
# ---------------------------

def _assignment_cost(theta_obs: np.ndarray, theta_pred: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Minimum-sum squared angular error assignment between observations and 10 predictions.
    Uses SciPy Hungarian algorithm if available; otherwise subset-DP fallback.
    """
    n_obs = int(theta_obs.shape[0])
    if n_obs == 0:
        return 0.0, []

    diffs = _diff_pi(theta_obs[:, None], theta_pred[None, :])
    cost = diffs * diffs

    if _linear_sum_assignment is not None:
        rows, cols = _linear_sum_assignment(cost)
        return float(cost[rows, cols].sum()), list(zip(rows.tolist(), cols.tolist()))

    # DP over bitmasks; practical because there are exactly 10 predictions.
    best: dict[int, Tuple[float, List[Tuple[int, int]]]] = {0: (0.0, [])}
    for i in range(n_obs):
        row = cost[i]
        next_best: dict[int, Tuple[float, List[Tuple[int, int]]]] = {}
        for mask, (score, path) in best.items():
            for j in range(10):
                bit = 1 << j
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_score = score + float(row[j])
                prev = next_best.get(new_mask)
                if prev is None or new_score < prev[0]:
                    next_best[new_mask] = (new_score, path + [(i, j)])
        best = next_best

    best_score, best_path = min(best.values(), key=lambda item: item[0])
    return float(best_score), best_path


def _fit_model(obs_deg: Sequence[float], *, fast: bool, max_evals: int) -> dict:
    """
    Coarse grid-search fit for (k, rho, beta) given observed angles in degrees.
    Returns inferred missing angles (degrees) among the 10 expected spokes.
    """
    theta_obs = _wrap_pi(np.deg2rad(np.asarray(obs_deg, dtype=F64)))

    if fast:
        k_grid = np.arange(0.4, 1.0 + 1e-9, 0.1, dtype=F64)
        angle_step = float(np.deg2rad(10.0))
    else:
        k_grid = np.arange(0.2, 1.0 + 1e-9, 0.02, dtype=F64)
        angle_step = float(np.deg2rad(2.0))

    rho_grid = np.arange(0.0, np.pi, angle_step, dtype=F64)
    beta_grid = np.arange(0.0, np.pi, angle_step, dtype=F64)

    best_cost = float("inf")
    best_params: Tuple[float, float, float] | None = None
    best_assigned: List[Tuple[int, int]] = []

    # Local bindings
    atan2 = np.arctan2
    sin = np.sin
    cos = np.cos
    phi = PHI_GRID
    pi = np.pi

    evals = 0
    for k in k_grid:
        for rho in rho_grid:
            for beta in beta_grid:
                x = phi - beta
                theta_pred = np.mod(rho + atan2(k * sin(x), cos(x)), pi)

                cost, assigned = _assignment_cost(theta_obs, theta_pred)
                if cost < best_cost:
                    best_cost = cost
                    best_params = (float(k), float(rho), float(beta))
                    best_assigned = assigned

                evals += 1
                if fast and evals >= max_evals:
                    break
            if fast and evals >= max_evals:
                break
        if fast and evals >= max_evals:
            break

    if best_params is None:
        return {"k": 0.0, "rho": 0.0, "beta": 0.0, "assigned": [], "missing_angles": []}

    k, rho, beta = best_params
    theta_pred = _predict_angles(k, rho, beta)

    used = {j for _, j in best_assigned}
    missing = [j for j in range(10) if j not in used]
    missing_angles = [float(np.rad2deg(theta_pred[j])) for j in missing]

    return {"k": k, "rho": rho, "beta": beta, "assigned": best_assigned, "missing_angles": missing_angles}


# ---------------------------
# Geometry / line helpers
# ---------------------------

def _select_unique_lines(
    raw_lines: Optional[np.ndarray],
    *,
    max_lines: int,
    rho_tol: float,
    theta_tol: float,
) -> List[LineRT]:
    """
    Greedy selection of Hough lines with rho/theta proximity suppression.
    raw_lines format: cv2.HoughLines => (N,1,2) with (rho,theta).
    """
    if raw_lines is None:
        return []
    selected: List[LineRT] = []
    for rho, theta in raw_lines[:, 0]:
        if len(selected) >= max_lines:
            break
        r, t = float(rho), float(theta)
        if any(abs(r - sr) <= rho_tol and abs(t - st) <= theta_tol for sr, st in selected):
            continue
        selected.append((r, t))
    return selected


def _consensus_point(lines: Sequence[LineRT]) -> Optional[PointF]:
    """
    Least-squares point (x,y) that best satisfies rho = x*cos(theta)+y*sin(theta) for all lines.
    """
    if len(lines) < 2:
        return None
    thetas = np.array([t for _, t in lines], dtype=np.float32)
    rhos = np.array([r for r, _ in lines], dtype=np.float32)
    A = np.column_stack((np.cos(thetas), np.sin(thetas))).astype(np.float32, copy=False)
    point, *_ = np.linalg.lstsq(A, rhos, rcond=None)
    return float(point[0]), float(point[1])


def _filter_by_consensus(lines: Sequence[LineRT], consensus: Optional[PointF], tol_px: float) -> List[LineRT]:
    """Keep lines whose distance to the consensus point is <= tol_px."""
    if consensus is None:
        return list(lines)
    cx, cy = consensus
    kept: List[LineRT] = []
    for rho, theta in lines:
        dist = abs(math.cos(theta) * cx + math.sin(theta) * cy - rho)
        if dist <= tol_px:
            kept.append((rho, theta))
    return kept


def _line_color_contrast(
    lab_img: np.ndarray,
    rho: float,
    theta: float,
    *,
    offset_px: float,
    samples: int,
) -> float:
    """
    Normalized [0..1] contrast across a line based on LAB differences sampled on both sides.
    Vectorized for performance.
    """
    h, w = lab_img.shape[:2]
    pts = line_border_points(rho, theta, w, h)
    if len(pts) < 2:
        return 0.0

    (x0, y0), (x1, y1) = pts
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    length = float(math.hypot(dx, dy))
    if length < 1.0:
        return 0.0

    nx = -(dy / length)
    ny = (dx / length)

    t = (np.arange(samples, dtype=np.float32) + 0.5) / float(samples)
    cx = x0 + t * dx
    cy = y0 + t * dy

    ax = np.rint(cx + nx * offset_px).astype(np.int32)
    ay = np.rint(cy + ny * offset_px).astype(np.int32)
    bx = np.rint(cx - nx * offset_px).astype(np.int32)
    by = np.rint(cy - ny * offset_px).astype(np.int32)

    valid = (0 <= ax) & (ax < w) & (0 <= ay) & (ay < h) & (0 <= bx) & (bx < w) & (0 <= by) & (by < h)
    if not np.any(valid):
        return 0.0

    a = lab_img[ay[valid], ax[valid]].astype(np.float32, copy=False)
    b = lab_img[by[valid], bx[valid]].astype(np.float32, copy=False)
    deltas = np.linalg.norm(a - b, axis=1)

    avg = float(deltas.mean()) if deltas.size else 0.0
    return min(avg / 60.0, 1.0)  # empirical scaling retained from original


def _infer_missing_lines(
    scored_lines: Sequence[ScoredLine],
    consensus: Optional[PointF],
    *,
    max_lines: int,
    fast: bool,
    max_evals: int,
) -> List[ScoredLine]:
    """
    Infer missing lines using the parametric 10-spoke model anchored at consensus point.
    Adds (rho, theta, 0.0, False) for inferred entries.
    """
    if consensus is None:
        return list(scored_lines)[:max_lines]

    observed_deg = [float(np.rad2deg(theta) % 180.0) for _, theta, _, found in scored_lines if found]
    if len(observed_deg) == 0 or len(observed_deg) >= max_lines:
        return list(scored_lines)[:max_lines]

    cx, cy = consensus
    model = _fit_model(observed_deg, fast=fast, max_evals=max_evals)

    existing = list(scored_lines)
    for angle_deg in model["missing_angles"]:
        if len(existing) >= max_lines:
            break
        theta = float(_wrap_pi(np.deg2rad(angle_deg)))
        rho = float(cx * math.cos(theta) + cy * math.sin(theta))
        existing.append((rho, theta, 0.0, False))
    return existing[:max_lines]


# ---------------------------
# Image decode
# ---------------------------

# ---------------------------
# Detection core
# ---------------------------

def _detect_lines(mask_bgr: np.ndarray, img_bgr: np.ndarray, cfg: DetectConfig) -> Tuple[List[ScoredLine], Optional[PointF]]:
    gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)

    h, w = gray.shape[:2]
    votes = max(cfg.hough_min_votes, int(min(h, w) * cfg.hough_votes_frac))

    raw_lines = cv2.HoughLines(edges, cfg.hough_rho, cfg.hough_theta, votes)
    selected = _select_unique_lines(
        raw_lines,
        max_lines=cfg.max_lines,
        rho_tol=cfg.unique_rho_tol,
        theta_tol=cfg.unique_theta_tol,
    )
    if not selected:
        return [], None

    consensus = _consensus_point(selected)
    precise = _filter_by_consensus(selected, consensus, cfg.consensus_tol_pass1)

    if len(precise) >= 2:
        consensus = _consensus_point(precise)
        precise = _filter_by_consensus(precise, consensus, cfg.consensus_tol_pass1)
    else:
        precise = _filter_by_consensus(selected, consensus, cfg.consensus_tol_pass2)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    scored: List[ScoredLine] = []
    for rho, theta in precise:
        color_score = _line_color_contrast(
            lab,
            rho,
            theta,
            offset_px=cfg.color_offset_px,
            samples=cfg.color_samples,
        )

        if consensus is None:
            center_score = 1.0
        else:
            cx, cy = consensus
            dist = abs(math.cos(theta) * cx + math.sin(theta) * cy - rho)
            center_score = 1.0 - min(dist / cfg.center_sigma_px, 1.0)

        score = cfg.score_color_w * color_score + cfg.score_center_w * center_score
        scored.append((rho, theta, float(score), True))

    scored.sort(key=lambda item: item[2], reverse=True)
    scored = scored[: cfg.max_lines]

    if len(scored) < cfg.max_lines and consensus is not None:
        scored = _infer_missing_lines(
            scored,
            consensus,
            max_lines=cfg.max_lines,
            fast=cfg.fit_fast,
            max_evals=cfg.fit_max_evals,
        )

    return scored, consensus


def _normalize_confidence(lines: Sequence[ScoredLine]) -> List[ScoredLine]:
    """
    Normalize scores of found lines to [0.01..1.0], inferred lines to 0.0.
    """
    found_scores = [s for _, _, s, found in lines if found]
    if not found_scores:
        return [(rho, theta, 0.0, found) for rho, theta, _, found in lines]

    smin = min(found_scores)
    smax = max(found_scores)
    denom = (smax - smin) if smax > smin else None

    out: List[ScoredLine] = []
    for rho, theta, score, found in lines:
        if not found:
            conf = 0.0
        elif denom is None:
            conf = 1.0
        else:
            conf = max(0.01, float((score - smin) / denom))
        out.append((rho, theta, float(conf), found))
    return out


# ---------------------------
# Public API
# ---------------------------

def detect_lines_from_image(
    image_input: ImageInput,
    *,
    params: Optional[TransformParams] = None,
    cfg: Optional[DetectConfig] = None,
) -> dict:
    """
    Production API: returns ONLY lines and their normalized accuracy.

    Input:
      - image_input: BGR numpy array OR base64 string (optionally data URL)

    Output:
      {
        "lines": [
          {"rho": float, "theta_rad": float, "accuracy": float},
          ...
        ]
      }
    """
    img_bgr = decode_base64_image(image_input) if isinstance(image_input, str) else image_input
    if img_bgr is None:
        return {"lines": []}

    if params is None:
        params = TransformParams()
    if cfg is None:
        cfg = DetectConfig()

    mask_bgr = build_white_mask(img_bgr, params=params)
    return detect_lines_from_mask(mask_bgr, img_bgr, cfg=cfg)


def detect_lines_from_mask(
    mask_bgr: np.ndarray,
    img_bgr: np.ndarray,
    *,
    cfg: Optional[DetectConfig] = None,
) -> dict:
    """
    Detect lines from a precomputed mask + original image.

    Returns:
      {"lines": [{"rho": float, "theta_rad": float, "accuracy": float}, ...]}
    """
    if cfg is None:
        cfg = DetectConfig()

    scored, _consensus = _detect_lines(mask_bgr, img_bgr, cfg)
    normalized = _normalize_confidence(scored)

    lines_out = [
        {"rho": float(rho), "theta_rad": float(theta), "accuracy": float(acc)}
        for rho, theta, acc, _found in normalized
    ]
    return {"lines": lines_out}
