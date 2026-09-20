$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
Set-Location $repo
$env:PYTHONIOENCODING = "utf-8"

$fail = 0
Write-Output "=== ruff ==="
& $py -m ruff check src tests
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Output "=== black --check ==="
& $py -m black --check src tests
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Output "=== mypy ==="
& $py -m mypy
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Output "CHECKS_FAIL=$fail"
exit $fail
