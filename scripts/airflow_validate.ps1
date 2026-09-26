# Validate the DriftTrace Airflow DAG in an ISOLATED venv so the main environment
# (Phase 1/3 deps) is never disturbed. Proves the DAG builds under real Airflow (FR-6).
#
# NOTE: Airflow's full runtime (scheduler/DagBag) is POSIX-only and does not run on
# native Windows (it uses os.register_at_fork). We shim that single POSIX-only hook so
# we can still construct and inspect the real Airflow DAG object on this machine. On a
# Linux/WSL/container host, no shim is needed and `airflow dags list` works directly.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$afVenv = Join-Path $repo ".venv-airflow"
$afPy = Join-Path $afVenv "Scripts\python.exe"

if (-not (Test-Path $afPy)) {
    Write-Output "Creating isolated Airflow venv..."
    py -3.13 -m venv $afVenv
    & $afPy -m pip install --upgrade pip --quiet
    & $afPy -m pip install "apache-airflow>=2.8" --quiet
    & $afPy -m pip install -e $repo --quiet
}

& $afPy -c "print('airflow', __import__('airflow').__version__)"

$env:AIRFLOW__CORE__LOAD_EXAMPLES = "False"
# Do NOT put the repo root on PYTHONPATH (it shadows stdlib on Windows). The DAG module
# is loaded by file path inside the validation script; drifttrace is editable-installed.
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
$validateScript = Join-Path $PSScriptRoot "airflow_validate.py"
$dagFile = Join-Path $repo "pipelines\airflow\dags\drifttrace_dag.py"
# Run from a neutral working directory to avoid local-package shadowing.
Push-Location $env:TEMP
& $afPy $validateScript $dagFile
$code = $LASTEXITCODE
Pop-Location
Write-Output "AIRFLOW_VALIDATE_EXIT=$code"
exit $code
