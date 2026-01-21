# AI Dart Scorer

Minimal FastAPI app that uploads a dartboard image, detects keypoints, and returns an overlay preview.

## Structure
```text
app/
  main.py
  vision/
templates/
  index.html
static/
  style.css
requirements.txt
```

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
uvicorn app.main:api --reload
```

## Run and expose to LAN
```bash
uvicorn app.main:api --host 0.0.0.0 --port 8000
```
