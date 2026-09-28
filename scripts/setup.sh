#!/usr/bin/env bash
# Fresh-clone setup for DriftTrace (Linux/macOS). Local-first, no Docker.
# Mirrors scripts/setup.ps1. Safe to re-run. Creates no model - upload a bundle to begin.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
echo "== DriftTrace setup =="
echo "Project root: $repo"

python="${PYTHON:-python3}"
if ! command -v "$python" >/dev/null 2>&1; then
  echo "ERROR: Python 3.11+ not found on PATH. Install Python and re-run." >&2
  exit 1
fi

venv_py="$repo/.venv/bin/python"
if [ ! -x "$venv_py" ]; then
  echo "Creating virtual environment at .venv ..."
  "$python" -m venv "$repo/.venv"
fi

echo "Installing DriftTrace (serving + streaming + explain extras) ..."
"$venv_py" -m pip install --upgrade pip --quiet
"$venv_py" -m pip install -e ".[serving,streaming,explain]"

echo "Preparing runtime directories (no model is created) ..."
"$venv_py" -m drifttrace.bootstrap

if command -v npm >/dev/null 2>&1 && [ -d "$repo/frontend" ]; then
  echo "Installing frontend dependencies (npm install) ..."
  (cd "$repo/frontend" && npm install)
fi

echo ""
echo "== Setup complete =="
echo "Start the backend:"
echo "    ./.venv/bin/python -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000"
echo "Start the frontend (second terminal):"
echo "    cd frontend && npm run dev"
echo "Then open http://localhost:5173 and upload a model bundle to begin (no model is active yet)."
