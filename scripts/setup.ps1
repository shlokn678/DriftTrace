<#
.SYNOPSIS
    Fresh-clone setup for DriftTrace (Windows / PowerShell). Local-first, no Docker.

.DESCRIPTION
    From a fresh git clone this script:
      1. creates a .venv virtual environment (if missing),
      2. installs the project with the serving + streaming + explain extras,
      3. prepares runtime directories (NO model is created - upload one to begin),
      4. prints the commands to start the backend and frontend.

    Safe to re-run.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1
#>
param(
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "== DriftTrace setup ==" -ForegroundColor Cyan
Write-Host "Project root: $repo"

# --- 1. Locate a Python 3.11+ interpreter -------------------------------------------
$python = $null
# Prefer the py launcher with an explicit 3.11+ request, then fall back to PATH.
foreach ($candidate in @("py -3.13", "py -3.12", "py -3.11", "python", "py -3")) {
    $exe = $candidate.Split(" ")[0]
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
        $probe = "$candidate -c `"import sys;print('%d.%d'%sys.version_info[:2])`""
        $v = Invoke-Expression "& $probe 2>`$null"
        if ($LASTEXITCODE -eq 0 -and $v) {
            $parts = $v.Trim().Split(".")
            if (([int]$parts[0] -gt 3) -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 11)) {
                $python = $candidate
                break
            }
        }
    }
}
if (-not $python) {
    Write-Error "Python 3.11+ is required but was not found. Install Python 3.11+ and re-run (the project's requires-python is >=3.11)."
    exit 1
}
Write-Host "Using interpreter: $python"

# --- 2. Create the virtual environment ----------------------------------------------
$venv = Join-Path $repo ".venv"
$venvPy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Creating virtual environment at .venv ..." -ForegroundColor Cyan
    Invoke-Expression "& $python -m venv `"$venv`""
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create the virtual environment."; exit 1 }
}
if (-not (Test-Path $venvPy)) { Write-Error "Virtual environment python not found at $venvPy."; exit 1 }

# --- 3. Install the package with the required extras --------------------------------
Write-Host "Installing DriftTrace (serving + streaming + explain extras) ..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -e ".[serving,streaming,explain]"
if ($LASTEXITCODE -ne 0) { Write-Error "Dependency installation failed."; exit 1 }

# --- 4. Prepare runtime directories (no model is created) ---------------------------
Write-Host "Preparing runtime directories (no model is created) ..." -ForegroundColor Cyan
& $venvPy -m drifttrace.bootstrap
if ($LASTEXITCODE -ne 0) { Write-Error "Bootstrap failed. See the message above."; exit 1 }

# --- 5. Frontend dependencies (optional) --------------------------------------------
if (-not $SkipFrontend) {
    $frontend = Join-Path $repo "frontend"
    if ((Test-Path $frontend) -and (Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Host "Installing frontend dependencies (npm install) ..." -ForegroundColor Cyan
        Push-Location $frontend
        try { npm install } finally { Pop-Location }
    }
    else {
        Write-Host "Skipping frontend deps (npm not found or no frontend/)." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "== Setup complete ==" -ForegroundColor Green
Write-Host "Start the backend:" -ForegroundColor Cyan
Write-Host "    .\.venv\Scripts\python.exe -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000"
Write-Host "Start the frontend (second terminal):" -ForegroundColor Cyan
Write-Host "    cd frontend; npm run dev"
Write-Host "Then open http://localhost:5173 and upload a model bundle to begin (no model is active yet)."
