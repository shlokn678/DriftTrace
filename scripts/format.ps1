$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
Set-Location $repo
& $py -m black src tests
& $py -m ruff check --fix src tests
Write-Output "FORMAT_DONE"
