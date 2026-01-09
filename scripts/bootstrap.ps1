# Bootstrap script for Windows PowerShell
# Creates .venv, installs dependencies, and installs pre-commit hooks.
$ErrorActionPreference = "Stop"

function Find-Python {
  if (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
  if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
  throw "Python not found. Install Python 3.10+ and ensure it's on PATH."
}

$py = Find-Python

if (-not (Test-Path ".venv")) {
  Write-Host "Creating venv in .venv..."
  & $py -m venv .venv
} else {
  Write-Host ".venv already exists."
}

$venvPy = Join-Path ".venv" "Scripts\python.exe"

Write-Host "Upgrading pip..."
& $venvPy -m pip install --upgrade pip

Write-Host "Installing package + dev dependencies..."
& $venvPy -m pip install -e ".[dev]"

Write-Host "Installing pre-commit hooks..."
& $venvPy -m pre_commit install

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Activate venv:  .\.venv\Scripts\Activate.ps1"
Write-Host "Run checks:     .\.venv\Scripts\python.exe -m ruff check .; .\.venv\Scripts\python.exe -m pytest -q"
Write-Host "Run app:        .\.venv\Scripts\python.exe -m streamlit run src/client_portal/app/streamlit_app.py"
