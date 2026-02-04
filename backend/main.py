from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.core.lifespan import lifespan
from backend.api import router as api_router


# The application relies on explicit startup/shutdown logic
# to initialize and release shared resources safely.
app = FastAPI(lifespan=lifespan)


# Register all backend routes in a single place
app.include_router(api_router)


# Serve the frontend at the root path.
# Backend API and frontend UI are intentionally hosted by the same app.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
