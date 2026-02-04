import asyncio
import base64
from pathlib import Path
from typing import Optional, Tuple
from pydantic import BaseModel

import cv2
import numpy as np
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from backend.vision.keypoint_detection import compute_keypoints
from backend.vision.calibration_store import load_calibration, update_warp_matrix
from backend.vision.reference import BOARD_DIAMETER_MM, REFERENCE_LINE_OUTER_MM

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

router = APIRouter()
ROTATE_STEP_DEG = 18.0


def _process_keypoints_image(image: np.ndarray) -> Tuple[dict, Optional[np.ndarray]]:
    # Computes the overlay image for UI/debug and returns an API-friendly payload.
    result = compute_keypoints(image)
    if result is None:
        raise HTTPException(status_code=500, detail="No result generated")

    result_img, total_matrix = result
    ok_result, result_png = cv2.imencode(".png", result_img)
    if not ok_result:
        raise HTTPException(status_code=500, detail="Failed to encode result")

    payload = {"image": base64.b64encode(result_png.tobytes()).decode("ascii")}
    payload["total_warp_matrix"] = (
        total_matrix.tolist() if isinstance(total_matrix, np.ndarray) else None
    )
    return payload, total_matrix


def _process_keypoints_bytes(data: bytes) -> Tuple[dict, Optional[np.ndarray], np.ndarray]:
    # Helper for upload/debug routes: decode bytes -> run keypoint pipeline.
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    payload, total_matrix = _process_keypoints_image(image)
    return payload, total_matrix, image


@router.get("/api/status")
def status(request: Request):
    # "ready" is controlled by the lifespan startup sequence (cameras init).
    return {"ready": bool(getattr(request.app.state, "ready", False))}


@router.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
@router.get("/api/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
def chrome_devtools_manifest():
    # Some Chrome tooling probes for this path; returning {} avoids noisy 404s in dev.
    return JSONResponse({})


@router.get("/game", include_in_schema=False)
def game():
    return FileResponse(FRONTEND_DIR / "game.html", media_type="text/html")


@router.get("/game/data-collection", include_in_schema=False)
def game_data_collection():
    return FileResponse(
        FRONTEND_DIR / "game-modes" / "data-collection.html",
        media_type="text/html",
    )


@router.get("/settings", include_in_schema=False)
def settings():
    return FileResponse(FRONTEND_DIR / "settings.html", media_type="text/html")


# -----------------------------
# Streaming MJPEG
# -----------------------------
BOUNDARY = "frame"


@router.get("/api/stream/{cam_id}", include_in_schema=False)
async def stream_camera(cam_id: str, request: Request):
    mgr = getattr(request.app.state, "camera_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Camera manager not initialized")

    async def gen():
        # MJPEG is implemented as a multipart stream that yields successive JPEG frames.
        # The camera thread provides "seq" values so we can wait for new frames efficiently.
        last_seq = -1
        while True:
            if await request.is_disconnected():
                break

            # wait_for_jpeg is blocking (thread/condition-based), so it must run off the event loop.
            jpeg, seq = await asyncio.to_thread(mgr.wait_for_jpeg, cam_id, last_seq, 0.5)
            if jpeg is None or seq == last_seq:
                await asyncio.sleep(0.005)
                continue

            last_seq = seq
            yield (
                f"--{BOUNDARY}\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n\r\n"
            ).encode("utf-8") + jpeg + b"\r\n"

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={
            # Streaming endpoints should not be cached; X-Accel-Buffering disables proxy buffering (e.g. nginx).
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# -----------------------------
# Camera actions
# -----------------------------
class CamAction(BaseModel):
    cam_id: str
    action: str  # "rotate_left" | "rotate_right" | "calibrate" | "reset"


class ManualWarpPoint(BaseModel):
    x: float
    y: float


class ManualWarpPayload(BaseModel):
    cam_id: str
    points: Tuple[ManualWarpPoint, ManualWarpPoint, ManualWarpPoint, ManualWarpPoint]


@router.post("/api/camera/action")
async def camera_action(payload: CamAction, request: Request):
    mgr = getattr(request.app.state, "camera_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Camera manager not initialized")

    cam_id = payload.cam_id
    action = payload.action

    if cam_id not in mgr.cams:
        raise HTTPException(status_code=404, detail="Unknown camera")

    if action == "rotate_left":
        # Rotation is applied to the persisted homography (useful when the board is slightly rotated).
        rotated = mgr.rotate_homography(cam_id, -ROTATE_STEP_DEG)
        if rotated is None:
            raise HTTPException(status_code=409, detail="No homography available to rotate")
        return {"ok": True}

    if action == "rotate_right":
        rotated = mgr.rotate_homography(cam_id, ROTATE_STEP_DEG)
        if rotated is None:
            raise HTTPException(status_code=409, detail="No homography available to rotate")
        return {"ok": True}

    if action == "calibrate":
        # Captures a prepared frame (undistorted/cropped/resized) to keep calibration consistent.
        print(f"Calibrating camera {cam_id}")
        frame = await asyncio.to_thread(mgr.capture_prepared_frame, cam_id, 1.0)
        if frame is None:
            raise HTTPException(status_code=503, detail="Camera frame unavailable")
        payload, total_matrix = _process_keypoints_image(frame)
        if total_matrix is not None:
            update_warp_matrix(cam_id, frame.shape[1], frame.shape[0], total_matrix)
        return payload

    if action == "reset":
        if not mgr.reset_homography(cam_id):
            raise HTTPException(status_code=500, detail="Failed to reset homography")
        return {"ok": True}

    raise HTTPException(status_code=400, detail="Unknown action")


@router.post("/api/camera/manual-warp")
async def camera_manual_warp(payload: ManualWarpPayload, request: Request):
    mgr = getattr(request.app.state, "camera_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Camera manager not initialized")

    cam_id = payload.cam_id
    if cam_id not in mgr.cams:
        raise HTTPException(status_code=404, detail="Unknown camera")

    frame = await asyncio.to_thread(mgr.capture_prepared_frame, cam_id, 1.0)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera frame unavailable")

    height, width = frame.shape[:2]
    size = min(width, height) - 1.0
    if size <= 0:
        raise HTTPException(status_code=400, detail="Invalid frame size")

    # Build a "canonical" set of source points on the reference circle in image space,
    # then map them to user-specified destination points to define a manual correction warp.
    scale = float(size) / float(BOARD_DIAMETER_MM)
    outer_px = float(REFERENCE_LINE_OUTER_MM) * scale
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5

    angles = np.radians([225.0, 315.0, 45.0, 135.0])
    src = np.array(
        [
            [cx + outer_px * np.cos(theta), cy + outer_px * np.sin(theta)]
            for theta in angles
        ],
        dtype=np.float32,
    )

    # UI provides points normalized to [0,1]; convert to pixel coordinates.
    dst = np.array(
        [
            [float(pt.x) * float(width), float(pt.y) * float(height)]
            for pt in payload.points
        ],
        dtype=np.float32,
    )

    try:
        user_warp = cv2.getPerspectiveTransform(src, dst)
    except cv2.error as exc:
        raise HTTPException(status_code=400, detail="Invalid adjustment") from exc

    try:
        # We store the correction in forward direction consistent with the runtime warp application.
        correction = np.linalg.inv(user_warp.astype(np.float64))
    except np.linalg.LinAlgError as exc:
        raise HTTPException(status_code=400, detail="Invalid adjustment (non-invertible)") from exc

    data = load_calibration(cam_id, width, height)
    warp = data.get("warp", {}) if isinstance(data, dict) else {}
    h_raw = warp.get("matrix")
    if h_raw is None:
        base = np.eye(3, dtype=np.float64)
    else:
        base = np.asarray(h_raw, dtype=np.float64)
        if base.shape != (3, 3):
            base = np.eye(3, dtype=np.float64)

    updated = correction @ base
    update_warp_matrix(cam_id, width, height, updated)
    return {"ok": True, "matrix": updated.tolist()}


# -----------------------------
# TEMP DEBUG UPLOAD - REMOVE BEFORE RELEASE
# -----------------------------
@router.post("/keypoints", include_in_schema=False)
async def keypoints_debug(file: UploadFile = File(...)):
    # Debug-only: allows running the vision pipeline on arbitrary uploads.
    data = await file.read()
    payload, _, _ = _process_keypoints_bytes(data)
    return payload
