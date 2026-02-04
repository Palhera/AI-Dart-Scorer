import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from backend.vision.cameras import CameraManager, CameraConfig

async def _init_cameras(app: FastAPI):
    app.state.ready = False

    app.state.camera_manager = CameraManager(
        {
            "cam1": CameraConfig(index=0, width=860, height=640, fps=30),
            "cam2": CameraConfig(index=4, width=860, height=640, fps=30),
            "cam3": CameraConfig(index=6, width=860, height=640, fps=30),
        }
    )
    app.state.camera_manager.start_all()

    timeout_s = 50.0
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    timed_out = False
    while not app.state.camera_manager.all_ready():
        if loop.time() - t0 > timeout_s:
            missing = [
                cam_id
                for cam_id, cam in app.state.camera_manager.cams.items()
                if not cam.is_open()
            ]
            if missing:
                print(
                    "WARNING: cameras not all ready within timeout "
                    f"(missing: {', '.join(missing)})"
                )
            else:
                print("WARNING: cameras not all ready within timeout")
            timed_out = True
            break
        await asyncio.sleep(0.1)

    app.state.ready = True
    if timed_out:
        print("Backend ready with missing cameras -> state.ready = True")
    else:
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
