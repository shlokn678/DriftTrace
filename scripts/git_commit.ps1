param(
    [Parameter(Mandatory = $true)][string]$Message
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
git add -A
git -c user.name="DriftTrace" -c user.email="drifttrace@example.invalid" commit -m $Message
Write-Output "COMMIT_EXIT=$LASTEXITCODE"
git log --oneline -1
exit 0
