param(
    [string]$AsOf = "",
    [string]$ForecastEnd = "2028-12"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

$fetchArgs = @("src/fetch_raw.py")
if ($AsOf) { $fetchArgs += @("--as-of", $AsOf) }
& $venvPython @fetchArgs
& $venvPython src/build_assignment_financials.py

& $venvPython src/pipeline.py --forecast-end $ForecastEnd
& $venvPython src/validate_models.py
& $venvPython src/build_scenarios.py
& $venvPython -m pytest -q
