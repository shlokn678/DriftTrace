$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
Set-Location $repo

if (-not (Test-Path ".dvc")) {
    & $py -m dvc init
    Write-Output "DVC_INIT_EXIT=$LASTEXITCODE"
}

# Local DVC remote (decision D-1): a directory inside the workspace, no cloud/credentials.
$remoteDir = Join-Path $repo ".dvc-storage"
New-Item -ItemType Directory -Path $remoteDir -Force | Out-Null
& $py -m dvc remote add -d -f localremote $remoteDir
Write-Output "DVC_REMOTE_EXIT=$LASTEXITCODE"

& $py -m dvc remote list
exit 0
