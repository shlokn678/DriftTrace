param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
Set-Location $repo
$env:PYTHONIOENCODING = "utf-8"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
& $py -m drifttrace.cli.main @CliArgs
Write-Output "CLI_EXIT=$LASTEXITCODE"
exit $LASTEXITCODE
