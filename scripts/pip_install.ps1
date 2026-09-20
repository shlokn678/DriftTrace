param(
    [Parameter(Mandatory = $true)][string]$Spec
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
& $py -m pip install $Spec --no-input
Write-Output "PIP_EXIT=$LASTEXITCODE"
exit $LASTEXITCODE
