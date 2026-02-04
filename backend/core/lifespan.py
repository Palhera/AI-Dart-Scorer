import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from backend.vision.cameras import CameraManager, CameraConfig


async def _init_cameras(app: FastAPI):
    # Exposed readiness flag used by the API to signal operational state
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
    app.state.camera_manager.start_all()

    # Wait for all cameras to be opened, but do not block startup indefinitely.
    # The backend is allowed to become "ready" even if some cameras are missing.
    timeout_s = 10.0
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

    # The backend is considered usable from this point on,
    # even if some cameras failed to initialize.
    app.state.ready = True
    if timed_out:
        print("Backend ready with missing cameras -> state.ready = True")
    else:
        print("Cameras ready -> state.ready = True")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting up...")
    app.state.ready = False

    # Camera initialization runs in the background to avoid blocking startup.
    cam_task = asyncio.create_task(_init_cameras(app))

    try:
        yield
    finally:
        print("Shutting down...")
        app.state.ready = False

        # Ensure background initialization does not outlive the application
        cam_task.cancel()
        try:
            await cam_task
        except asyncio.CancelledError:
            pass

        # Gracefully stop all camera threads/resources if they were created
        mgr = getattr(app.state, "camera_manager", None)
        if mgr:
            mgr.stop_all()

        print("Backend stopped")
