# setup.ps1 - one-time environment setup for early_trend_scanner (Windows PowerShell 5.1+)
# Creates .venv, installs pinned dependencies, prepares .env.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Find-Python {
    $candidates = @()
    foreach ($name in @('python', 'python3')) {
        try {
            $cmd = Get-Command $name -ErrorAction Stop
            if ($cmd.Source -notlike '*WindowsApps*') { $candidates += $cmd.Source }
        } catch {}
    }
    $candidates += @(
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        'C:\Program Files\Python313\python.exe',
        'C:\Program Files\Python312\python.exe',
        'C:\Program Files\Python311\python.exe'
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $v = & $c -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($v -and [version]$v -ge [version]'3.11') { return $c }
        }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host 'Python 3.11+ was not found. Install it first, e.g.:' -ForegroundColor Yellow
    Write-Host '  winget install --id Python.Python.3.12 --scope user' -ForegroundColor Yellow
    Write-Host 'then re-run .\setup.ps1'
    exit 1
}
Write-Host "Using Python: $py"

if (-not (Test-Path "$root\.venv")) {
    & $py -m venv "$root\.venv"
    Write-Host 'Created .venv'
}
$venvPy = "$root\.venv\Scripts\python.exe"

& $venvPy -m pip install --upgrade pip --quiet
Write-Host 'Installing dependencies (pinned)...'
& $venvPy -m pip install -r "$root\requirements.txt" --quiet
& $venvPy -m pip install -e $root --no-deps --quiet
if ($LASTEXITCODE -ne 0) { Write-Host 'pip install failed' -ForegroundColor Red; exit 1 }

if (-not (Test-Path "$root\.env")) {
    Copy-Item "$root\.env.example" "$root\.env"
    Write-Host 'Created .env from template - EDIT IT with your Alpaca + Telegram keys.' -ForegroundColor Yellow
}

New-Item -ItemType Directory -Force "$root\data" | Out-Null
New-Item -ItemType Directory -Force "$root\logs" | Out-Null

& $venvPy -c "import early_trend_scanner, aiohttp, yaml; print('import check OK, version', early_trend_scanner.__version__)"
if ($LASTEXITCODE -ne 0) { Write-Host 'Import check failed' -ForegroundColor Red; exit 1 }

Write-Host ''
Write-Host 'Setup complete. Next steps:' -ForegroundColor Green
Write-Host '  1. Edit .env  (APCA_API_KEY_ID, APCA_API_SECRET_KEY, TELEGRAM_BOT_TOKEN)'
Write-Host '  2. .\telegram_test.ps1 -Setup     # discovers TELEGRAM_CHAT_ID + sends test'
Write-Host '  3. .\run.ps1 replay               # offline synthetic replay smoke test'
Write-Host '  4. .\install_task.ps1             # schedule daily start before the open'
