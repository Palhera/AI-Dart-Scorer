# AI Dart Scorer

Local FastAPI app that serves a web UI, streams dartboard cameras, and detects keypoints for calibration.

## Quick start
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --workers 1
```

Open:
- http://localhost:8000/ (UI)
- http://localhost:8000/settings
- http://localhost:8000/game

## Configure cameras
Edit camera indices and resolution in `backend/core/lifespan.py`.

## Documentation
- `PROJECT_GUIDE.md` (architecture, flows, schemas, and maintenance notes)

## LAN run
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```
