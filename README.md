# AI Darts Scorer

## Project structure
```text
.
├── .venv/
├── app/
│   ├── __init__.py
│   └── main.py
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

## Setup (Linux)
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

## Setup (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt

## Run
uvicorn app.main:api --reload

## Run and expose to lan
uvicorn app.main:api --host 0.0.0.0 --port 8000
