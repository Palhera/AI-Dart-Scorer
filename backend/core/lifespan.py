import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from backend.vision.cameras import CameraManager, CameraConfig

async def _init_cameras(app: FastAPI):
    app.state.ready = False

    app.state.camera_manager = CameraManager(
        {
            "cam1": CameraConfig(index=0, width=720, height=720, fps=30),
            "cam2": CameraConfig(index=1, width=720, height=720, fps=30),
            "cam3": CameraConfig(index=2, width=720, height=720, fps=30),
        }
    )
    app.state.camera_manager.start_all()

    timeout_s = 10.0
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    while not app.state.camera_manager.all_ready():
        if loop.time() - t0 > timeout_s:
            print("WARNING: cameras not all ready within timeout")
            return
        await asyncio.sleep(0.1)

    app.state.ready = True
    print("Cameras ready -> state.ready = True")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting up...")
    app.state.ready = False

    cam_task = asyncio.create_task(_init_cameras(app))

    try:
        yield
    finally:
        print("Shutting down...")
        app.state.ready = False

        cam_task.cancel()
        try:
            await cam_task
        except asyncio.CancelledError:
            pass

        mgr = getattr(app.state, "camera_manager", None)
        if mgr:
            mgr.stop_all()

        print("Backend stopped")
