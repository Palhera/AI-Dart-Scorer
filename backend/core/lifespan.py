from contextlib import asynccontextmanager
from fastapi import FastAPI

from backend.vision.cameras import CameraManager, CameraConfig

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting up...")
    app.state.ready = False

    # CameraManager is stored on app.state to make it globally accessible
    # to request handlers without relying on globals.
    app.state.camera_manager = CameraManager(
        {
            "cam1": CameraConfig(
                device="/dev/v4l/by-path/platform-xhci-hcd.2.auto-usb-0:1.2.4:1.0-video-index0",
                width=1280, height=720, fps=30
            ),
            "cam2": CameraConfig(
                device="/dev/v4l/by-path/platform-xhci-hcd.2.auto-usb-0:1.3:1.0-video-index0",
                width=1280, height=720, fps=30
            ),
            "cam3": CameraConfig(
                device="/dev/v4l/by-path/platform-xhci-hcd.2.auto-usb-0:1.4:1.0-video-index0",
                width=1280, height=720, fps=30
            ),
        }
    )
    app.state.ready = True

    try:
        yield
    finally:
        print("Shutting down...")
        app.state.ready = False

        mgr = getattr(app.state, "camera_manager", None)
        if mgr:
            mgr.close_all()

        print("Backend stopped")
