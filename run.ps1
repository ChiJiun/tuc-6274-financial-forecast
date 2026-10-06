$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

& ".\.venv\Scripts\python.exe" src/fetch_raw.py
& ".\.venv\Scripts\python.exe" src/pipeline.py
& ".\.venv\Scripts\python.exe" -m pytest -q
