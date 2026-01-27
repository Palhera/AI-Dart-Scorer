import base64
import base64
import re
import threading
import time

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.camera_manager import CameraManager
from backend.vision.keypoint_detection import compute_keypoints
from backend.vision.reference import draw_reference_overlay

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_IMAGE_PATH = BASE_DIR / "backend" / "vision" / "reference.png"

api = FastAPI()
api.mount("/frontend", StaticFiles(directory=str(BASE_DIR / "frontend")), name="frontend")

camera_manager = CameraManager(cam_ids=(1, 2, 3))
capture_lock = threading.Lock()

@api.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_manifest():
    return JSONResponse({})

@api.get("/")
def index():
    return FileResponse(BASE_DIR / "frontend" / "index.html", media_type="text/html")

@api.get("/game")
def game():
    return FileResponse(BASE_DIR / "frontend" / "game-page.html", media_type="text/html")

@api.get("/data-collection")
def data_collection_page():
    return FileResponse(BASE_DIR / "frontend" / "data-collection.html", media_type="text/html")

@api.get("/calibration")
def calibration():
    return FileResponse(BASE_DIR / "frontend" / "calibration-page.html", media_type="text/html")

def _build_reference_overlay_png() -> bytes:
    image = cv2.imread(str(REFERENCE_IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=500, detail="Reference image not found")

    overlay = draw_reference_overlay(image)
    ok_overlay, overlay_png = cv2.imencode(".png", overlay)
    if not ok_overlay:
        raise HTTPException(status_code=500, detail="Failed to encode reference overlay")

    return overlay_png.tobytes()

@api.get("/reference-overlay.png")
def reference_overlay():
    return Response(content=_build_reference_overlay_png(), media_type="image/png")

@api.get("/cameras")
def cameras():
    available = camera_manager.get_available()
    return {"available": available}

def _mjpeg_stream(cam_id: int):
    while True:
        frame = camera_manager.get_frame(cam_id)
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )
        time.sleep(0.03)

@api.get("/camera/{cam_id}/stream")
def camera_stream(cam_id: int):
    if cam_id not in (1, 2, 3):
        raise HTTPException(status_code=404, detail="Unknown camera")
    if cam_id not in camera_manager.get_available():
        raise HTTPException(status_code=404, detail="Camera not available")
    return StreamingResponse(
        _mjpeg_stream(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )

def _find_next_capture_id(folder: Path) -> int:
    if not folder.exists():
        return 1
    pattern = re.compile(r"^(\d+)_cam\d+\.png$", re.IGNORECASE)
    max_id = 0
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        match = pattern.match(entry.name)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except ValueError:
            continue
        max_id = max(max_id, value)
    return max_id + 1

@api.post("/data-collection/capture")
def data_collection_capture(payload: dict = Body(...)):
    save_path = payload.get("save_path") if isinstance(payload, dict) else None
    if not save_path or not isinstance(save_path, str):
        raise HTTPException(status_code=400, detail="Missing save_path")

    target_dir = Path(save_path).expanduser()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - OS-specific failure
        raise HTTPException(status_code=400, detail="Unable to access save_path") from exc

    with capture_lock:
        available = camera_manager.get_available()
        if not available:
            raise HTTPException(status_code=400, detail="No cameras available")

        frames = {}
        for cam_id in available:
            frame = camera_manager.get_frame(cam_id)
            if frame is None:
                raise HTTPException(status_code=500, detail=f"Camera {cam_id} not ready")
            frames[cam_id] = frame

        capture_id = _find_next_capture_id(target_dir)
        id_str = f"{capture_id:05d}"
        saved_files = []

        try:
            for cam_id, frame in frames.items():
                filename = f"{id_str}_cam{cam_id}.png"
                file_path = target_dir / filename
                ok = cv2.imwrite(str(file_path), frame)
                if not ok:
                    raise RuntimeError(f"Failed to save camera {cam_id}")
                saved_files.append(filename)
        except Exception as exc:
            for filename in saved_files:
                try:
                    (target_dir / filename).unlink()
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail="Failed to save capture") from exc

        return {"id": id_str, "cameras": available, "saved": saved_files}

""" TEMPORARY UPLOAD TESTING ENDPOINT - REMOVE LATER """
@api.post("/keypoints")
async def keypoints(
    file: UploadFile = File(...),
):
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

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
    return payload
