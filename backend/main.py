from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.core.lifespan import lifespan
from backend.api import router as api_router

app = FastAPI(lifespan=lifespan)

# API surface
app.include_router(api_router)

# Frontend (served at "/")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
