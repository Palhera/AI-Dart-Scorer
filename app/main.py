import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.vision.keypoint_detection import compute_keypoints

BASE_DIR = Path(__file__).resolve().parent.parent

api = FastAPI()
api.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@api.get("/")
def index():
    return FileResponse(BASE_DIR / "templates" / "index.html", media_type="text/html")

@api.post("/keypoints")
async def keypoints(
    file: UploadFile = File(...),
    points: bool = True,
    lines: bool = False,
    circles: bool = False,
):
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    overlay = compute_keypoints(
        image,
        overlay_points=points,
        overlay_lines=lines,
        overlay_circles=circles,
    )
    if overlay is None:
        raise HTTPException(status_code=500, detail="No overlay generated")

    ok_overlay, overlay_png = cv2.imencode(".png", overlay)
    if not ok_overlay:
        raise HTTPException(status_code=500, detail="Failed to encode overlay")

    return {"overlay": base64.b64encode(overlay_png.tobytes()).decode("ascii")}
