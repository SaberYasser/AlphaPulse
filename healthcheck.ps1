# healthcheck.ps1 - is the scanner alive and healthy right now?
# Exit codes: 0 healthy | 1 degraded/stale | 2 not running
$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$statusPath = Join-Path $root 'data\status.json'

if (-not (Test-Path $statusPath)) {
    Write-Host 'NOT RUNNING: no status file yet' -ForegroundColor Red
    exit 2
}
$status = Get-Content $statusPath -Raw | ConvertFrom-Json
$age = [int]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [double]$status.written_at)

Write-Host ("state           : {0}" -f $status.state)
Write-Host ("status age      : {0}s" -f $age)
if ($null -ne $status.session)        { Write-Host ("session         : {0}" -f $status.session) }
if ($null -ne $status.process_id)     { Write-Host ("process id      : {0}" -f $status.process_id) }
if ($null -ne $status.project_root)   { Write-Host ("project root    : {0}" -f $status.project_root) }
if ($null -ne $status.symbols)        { Write-Host ("universe        : {0}" -f ($status.symbols -join ', ')) }
if ($null -ne $status.ws_connected)   { Write-Host ("ws connected    : {0}" -f $status.ws_connected) }
if ($null -ne $status.alerts_enabled) { Write-Host ("alerts enabled  : {0}" -f $status.alerts_enabled) }
if ($null -ne $status.last_rx_age_s)  { Write-Host ("last feed event : {0}s ago" -f $status.last_rx_age_s) }
if ($null -ne $status.feed_latency_s) { Write-Host ("feed latency    : {0}s" -f $status.feed_latency_s) }
if ($null -ne $status.alerts_today)   { Write-Host ("alerts today    : {0}" -f $status.alerts_today) }
if ($null -ne $status.rss_mb)         { Write-Host ("memory (RSS)    : {0} MB" -f $status.rss_mb) }
if ($null -ne $status.telegram_sent)  { Write-Host ("telegram sent   : {0} (failed {1})" -f $status.telegram_sent, $status.telegram_failed) }
if ($null -ne $status.model_version)  { Write-Host ("model           : {0}" -f $status.model_version) }
if ($null -ne $status.efficacy_model_version) {
    Write-Host ("efficacy model  : {0}" -f $status.efficacy_model_version)
}
if ($null -ne $status.efficacy_gate_active) {
    Write-Host ("efficacy gate   : {0}" -f $status.efficacy_gate_active)
}

$task = Get-ScheduledTask -TaskName 'EarlyTrendScanner'
if ($task) { Write-Host ("scheduled task  : {0}" -f $task.State) }

if ($status.state -eq 'scanning') {
    if ($age -gt 60) {
        Write-Host 'DEGRADED: status heartbeat is stale' -ForegroundColor Yellow
        exit 1
    }
    if (-not $status.ws_connected) {
        Write-Host 'DEGRADED: websocket disconnected' -ForegroundColor Yellow
        exit 1
    }
    if ($null -ne $status.alerts_enabled -and -not $status.alerts_enabled) {
        Write-Host 'DEGRADED: alerts suppressed by feed health' -ForegroundColor Yellow
        exit 1
    }
    if (($status.dropped_trades -gt 0) -or ($status.dropped_quotes -gt 0)) {
        Write-Host 'DEGRADED: stream queue has dropped market-data events' -ForegroundColor Yellow
        exit 1
    }
    if ($null -ne $status.telegram_enabled -and -not $status.telegram_enabled) {
        Write-Host 'DEGRADED: Telegram is disabled (shadow mode)' -ForegroundColor Yellow
        exit 1
    }
    if ($status.rss_mb -gt 800) {
        Write-Host 'DEGRADED: memory above 800 MB' -ForegroundColor Yellow
        exit 1
    }
    Write-Host 'HEALTHY' -ForegroundColor Green
    exit 0
}
Write-Host ("Scanner not actively scanning (state={0}) - normal outside market hours." -f $status.state)
exit 0
