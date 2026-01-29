from pathlib import Path
import asyncio
from pydantic import BaseModel

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

router = APIRouter()

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
    action: str  # "rotate_left" | "rotate_right" | "calibrate"

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
        # Do something
        print(f"Rotating camera {cam_id} left")
        return {"ok": True}

    if action == "rotate_right":
        # Do something
        print(f"Rotating camera {cam_id} right")
        return {"ok": True}

    if action == "calibrate":
        # Do something
        print(f"Calibrating camera {cam_id}")
        return {"ok": True}

    raise HTTPException(status_code=400, detail="Unknown action")