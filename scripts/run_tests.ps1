param(
    [string]$Target = "tests",
    [string]$Marker = ""
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"

$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"

$targets = $Target -split '\s+' | Where-Object { $_ -ne "" }
$args = @("-m", "pytest") + $targets + @("-p", "no:cacheprovider", "-q", "--junit-xml=$repo\test-results.xml")
if ($Marker -ne "") { $args += @("-m", $Marker) }

& $py @args
$code = $LASTEXITCODE
Write-Output "PYTEST_EXIT=$code"
exit $code
