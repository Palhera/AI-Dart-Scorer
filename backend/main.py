import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.vision.keypoint_detection import compute_keypoints
from backend.vision.reference import draw_reference_overlay

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_IMAGE_PATH = BASE_DIR / "backend" / "vision" / "reference.png"

api = FastAPI()
api.mount("/frontend", StaticFiles(directory=str(BASE_DIR / "frontend")), name="frontend")

@api.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_manifest():
    return JSONResponse({})

@api.get("/")
def index():
    return FileResponse(BASE_DIR / "frontend" / "index.html", media_type="text/html")

@api.get("/game")
def game():
    return FileResponse(BASE_DIR / "frontend" / "game-page.html", media_type="text/html")

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
