import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.vision.keypoint_detection import compute_keypoints

BASE_DIR = Path(__file__).resolve().parent.parent

api = FastAPI()
api.mount("/frontend", StaticFiles(directory=str(BASE_DIR / "frontend")), name="frontend")

@api.get("/")
def index():
    return FileResponse(BASE_DIR / "frontend" / "index.html", media_type="text/html")

@api.get("/game")
def game():
    return FileResponse(BASE_DIR / "frontend" / "game-page.html", media_type="text/html")

@api.get("/calibration")
def calibration():
    return FileResponse(BASE_DIR / "frontend" / "calibration-page.html", media_type="text/html")

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

    ok_result, result_png = cv2.imencode(".png", result)
    if not ok_result:
        raise HTTPException(status_code=500, detail="Failed to encode result")

    return {"image": base64.b64encode(result_png.tobytes()).decode("ascii")}
