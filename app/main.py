import base64

import cv2
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.vision.ellipse_detection import build_red_green_mask
from app.vision.line_detection import build_white_mask
from app.vision.keypoint_detection import compute_keypoints

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

api = FastAPI()
api.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@api.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@api.post("/transform")
async def transform(file: UploadFile = File(...)):
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    white_mask = build_white_mask(image)
    red_green_mask = build_red_green_mask(image)

    ok_white, white_png = cv2.imencode(".png", white_mask)
    ok_rg, rg_png = cv2.imencode(".png", red_green_mask)
    if not ok_white or not ok_rg:
        return JSONResponse({"error": "Failed to encode images"}, status_code=500)

    return JSONResponse(
        {
            "white_mask": base64.b64encode(white_png.tobytes()).decode("ascii"),
            "red_green_mask": base64.b64encode(rg_png.tobytes()).decode("ascii"),
        }
    )


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
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    result = compute_keypoints(
        image,
        overlay_points=points,
        overlay_lines=lines,
        overlay_circles=circles,
    )
    overlay = result.get("overlay")
    if overlay is None:
        return JSONResponse({"error": "No overlay generated"}, status_code=500)

    ok_overlay, overlay_png = cv2.imencode(".png", overlay)
    if not ok_overlay:
        return JSONResponse({"error": "Failed to encode overlay"}, status_code=500)

    return JSONResponse(
        {
            "overlay": base64.b64encode(overlay_png.tobytes()).decode("ascii"),
            "keypoints": result.get("keypoints", []),
        }
    )
