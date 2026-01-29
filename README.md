# AI Dart Scorer

Minimal FastAPI app that uploads a dartboard image, detects keypoints, and returns a processed preview image.

## Setup
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Run
```bash
uvicorn backend.main:app --reload
```

## Run and expose to LAN
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
