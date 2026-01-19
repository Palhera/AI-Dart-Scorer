from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

api = FastAPI()

@api.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@api.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    return JSONResponse({
        "filename": file.filename,
        "content_type": file.content_type,
        "bytes_received": len(data),
    })
