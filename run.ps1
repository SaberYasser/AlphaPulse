# run.ps1 - start the scanner (or any ets subcommand) inside the project venv.
#   .\run.ps1                 -> live scanner (regular market hours only)
#   .\run.ps1 replay          -> synthetic replay
#   .\run.ps1 clock           -> market clock check
#   .\run.ps1 status          -> heartbeat
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPy = "$root\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host 'venv missing - run .\setup.ps1 first' -ForegroundColor Red
    exit 1
}
[string[]]$cmdArgs = if ($args.Count -gt 0) { $args } else { @('run') }
& $venvPy -m early_trend_scanner @cmdArgs
exit $LASTEXITCODE
