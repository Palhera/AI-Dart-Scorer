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
            "cam1": CameraConfig(index=0, width=1280, height=720, fps=30),
            "cam2": CameraConfig(index=4, width=1280, height=720, fps=30),
            "cam3": CameraConfig(index=6, width=1280, height=720, fps=30),
        }
    )
    app.state.ready = True

    try:
        yield
    finally:
        print("Shutting down...")
        app.state.ready = False

        print("Backend stopped")
