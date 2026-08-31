# install_task.ps1 - register a Task Scheduler job that starts the scanner
# shortly before the U.S. market open, waking the computer if needed.
#
# The task fires at a LOCAL time computed from 09:20 America/New_York at install
# time. The scanner itself uses the Alpaca market clock, so an off-by-an-hour
# trigger after a DST change is harmless (it waits or catches up) - re-run this
# script after DST switches or a timezone change to restore exact timing.
param(
    [string]$TaskName = 'EarlyTrendScanner',
    [int]$EtHour = 9,
    [int]$EtMinute = 20
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$et = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
$todayEt = [System.TimeZoneInfo]::ConvertTime([DateTimeOffset]::Now, $et).Date
$triggerEt = New-Object DateTimeOffset(
    $todayEt.AddHours($EtHour).AddMinutes($EtMinute), $et.GetUtcOffset($todayEt.AddHours(12)))
$triggerLocal = $triggerEt.ToLocalTime().DateTime
Write-Host ("{0:HH:mm} ET maps to {1:HH:mm} local time today" -f $triggerEt.DateTime, $triggerLocal)

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$root\run.ps1`"" `
    -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -Weekly -At $triggerLocal `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday

$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 10)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description 'Early trend scanner - starts before US market open' `
    -Force | Out-Null

Write-Host "Task '$TaskName' registered:" -ForegroundColor Green
Write-Host "  runs Mon-Fri at $($triggerLocal.ToString('HH:mm')) local (~$EtHour`:$($EtMinute.ToString('00')) ET)"
Write-Host '  "Wake the computer to run this task" is ENABLED'
Write-Host '  the app exits on non-trading days after checking the Alpaca calendar'
Write-Host ''
Write-Host 'Verify with:  Get-ScheduledTask -TaskName EarlyTrendScanner | Get-ScheduledTaskInfo'
Write-Host 'Run now with: Start-ScheduledTask -TaskName EarlyTrendScanner'
Write-Host 'Remove with:  Unregister-ScheduledTask -TaskName EarlyTrendScanner -Confirm:$false'
