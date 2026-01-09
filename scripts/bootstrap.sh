#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  echo "Creating venv in .venv..."
  python3 -m venv .venv
else
  echo ".venv already exists."
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install

echo ""
echo "Bootstrap complete."
echo "Activate venv:  source .venv/bin/activate"
echo "Run checks:     ruff check . && pytest -q"
echo "Run app:        streamlit run src/client_portal/app/streamlit_app.py"
