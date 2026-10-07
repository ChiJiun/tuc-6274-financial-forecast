param(
    [string]$AsOf = "",
    [string]$ForecastEnd = "2028-12"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

$fetchArgs = @("src/fetch_raw.py")
if ($AsOf) { $fetchArgs += @("--as-of", $AsOf) }
& ".\.venv\Scripts\python.exe" @fetchArgs
& ".\.venv\Scripts\python.exe" src/build_assignment_financials.py

& ".\.venv\Scripts\python.exe" src/pipeline.py --forecast-end $ForecastEnd
& ".\.venv\Scripts\python.exe" -m pytest -q
