$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$venvScripts = Join-Path $repo ".venv\Scripts"
$py = Join-Path $venvScripts "python.exe"
Set-Location $repo
$env:PYTHONIOENCODING = "utf-8"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:DVC_NO_ANALYTICS = "1"
# Ensure DVC stage subprocesses that call bare `python` use the venv interpreter.
$env:PATH = "$venvScripts;$env:PATH"

& $py -m dvc repro
Write-Output "DVC_REPRO_EXIT=$LASTEXITCODE"
exit $LASTEXITCODE
