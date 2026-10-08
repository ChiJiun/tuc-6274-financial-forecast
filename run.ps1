param(
    [string]$AsOf = "",
    [string]$ForecastEnd = "2028-12"
)

$ErrorActionPreference = "Stop"
$venvPython = ".\.venv\Scripts\python.exe"
$requiredPython = "3.12.15"

if (-not (Test-Path ".venv")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
    } else {
        python -m venv .venv
    }
}

$pythonVersion = & $venvPython -c "import platform; print(platform.python_version())"
if ($pythonVersion -ne $requiredPython) {
    throw "Expected Python $requiredPython, found $pythonVersion. Remove .venv, install Python $requiredPython, and rerun."
}

& $venvPython -m pip install -r requirements.lock.txt
& $venvPython -m pip check

$fetchArgs = @("src/fetch_raw.py")
if ($AsOf) { $fetchArgs += @("--as-of", $AsOf) }
& $venvPython @fetchArgs
& $venvPython src/build_assignment_financials.py

& $venvPython src/pipeline.py --forecast-end $ForecastEnd
& $venvPython src/validate_models.py
& $venvPython src/build_scenarios.py
& $venvPython -m pytest -q
