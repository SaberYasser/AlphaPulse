# telegram_test.ps1 - Telegram wiring.
#   .\telegram_test.ps1 -Setup   -> guided chat-id discovery for @YourTelegramUsername + test message
#   .\telegram_test.ps1          -> send a test message with stored credentials
param([switch]$Setup)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPy = "$root\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host 'venv missing - run .\setup.ps1 first' -ForegroundColor Red
    exit 1
}
if ($Setup) {
    & $venvPy -m early_trend_scanner telegram-setup
} else {
    & $venvPy -m early_trend_scanner telegram-test
}
exit $LASTEXITCODE
