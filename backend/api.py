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
from backend.vision.calibration_store import update_warp_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

router = APIRouter()
ROTATE_STEP_DEG = 18.0

def _process_keypoints_image(image: np.ndarray) -> Tuple[dict, Optional[np.ndarray]]:
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
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    payload, total_matrix = _process_keypoints_image(image)
    return payload, total_matrix, image

@router.get("/api/status")
def status(request: Request):
    return {"ready": bool(getattr(request.app.state, "ready", False))}

@router.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
@router.get("/api/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
def chrome_devtools_manifest():
    return JSONResponse({})

@router.get("/game", include_in_schema=False)
def game():
    return FileResponse(FRONTEND_DIR / "game.html", media_type="text/html")

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
        while True:
            if await request.is_disconnected():
                break

            jpeg = mgr.get_jpeg(cam_id)
            if jpeg is None:
                await asyncio.sleep(0.05)
                continue

            yield (
                f"--{BOUNDARY}\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n\r\n"
            ).encode("utf-8") + jpeg + b"\r\n"

            await asyncio.sleep(0.03)  # ~30 fps

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-cache"},
    )


# -----------------------------
# Camera actions
# -----------------------------
class CamAction(BaseModel):
    cam_id: str
    action: str  # "rotate_left" | "rotate_right" | "calibrate" | "reset"

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


# -----------------------------
# TEMP DEBUG UPLOAD - REMOVE BEFORE RELEASE
# -----------------------------
@router.post("/keypoints", include_in_schema=False)
async def keypoints_debug(file: UploadFile = File(...)):
    data = await file.read()
    payload, _, _ = _process_keypoints_bytes(data)
    return payload
