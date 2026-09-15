# Capture one batch session at a scheduled UTC time and hand it to the frozen
# pipeline -- in a process that outlives the shell that started it.
#
#   powershell -File scripts\batch_orchestrate.ps1 -Date 2026-09-16 -StartUtc 2026-09-16T00:00:00Z
#
# What it does, in order, logging every step to runs\batch-1-orchestrator-<Date>.log:
#   1. detaches itself (hidden window) and holds the keep-awake request the whole time
#   2. waits until StartUtc
#   3. refuses if a capture is already running or the target partition exists
#   4. runs TickForge's scripts\run_capture.ps1 -Hours <Hours>, defaults unchanged, and waits for it
#   5. verifies the partition closed (files present, MicroFlux eligibility facts logged)
#   6. launches scripts\run_detached.ps1 -Date <Date> -Batch <Batch>, which gates on
#      eligibility and batch rules and then runs replication -> neural -> final test once
#   7. waits for that pipeline's exit marker and logs BATCH SESSION COMPLETE
# It changes no data setting and no scientific rule; the gate in run_session.py decides.

param(
    [Parameter(Mandatory = $true)] [string] $Date,
    [Parameter(Mandatory = $true)] [string] $StartUtc,
    [double] $Hours = 8.5,
    [string] $Symbol = "BTCUSDT",
    [string] $Root = "C:\tickforge-runs",
    [string] $TickForge = "C:\Users\iidab\OneDrive\Desktop\TickForge",
    [string] $Batch = "docs/collection-batch-1.json",
    [switch] $Inner
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repo "runs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "batch-1-orchestrator-$Date.log"
function Log([string] $m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

if (-not $Inner) {
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
              "-Date", $Date, "-StartUtc", $StartUtc, "-Hours", $Hours, "-Symbol", $Symbol,
              "-Root", $Root, "-TickForge", $TickForge, "-Batch", $Batch, "-Inner")
    $proc = Start-Process -FilePath "powershell" -ArgumentList $args -WorkingDirectory $repo -WindowStyle Hidden -PassThru
    Write-Host "orchestrator PID $($proc.Id)   session $Symbol-$Date   capture at $StartUtc for $Hours h"
    Write-Host "log  $log"
    exit 0
}

# --- inner ---------------------------------------------------------------------
$KEEP_AWAKE = [uint32] 2147483713   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED (as TickForge's run_capture.ps1)
$RELEASE = [uint32] 2147483648
Add-Type -Name Power -Namespace Win32 -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$null = [Win32.Power]::SetThreadExecutionState($KEEP_AWAKE)
Log "=== orchestrator start PID $PID  target $Symbol-$Date  capture at $StartUtc for $Hours h  keep-awake held"
$partition = Join-Path $Root "binance\$Symbol\$Date"
try {
    # 2. wait
    $start = [DateTime]::Parse($StartUtc, $null, [System.Globalization.DateTimeStyles]::AdjustToUniversal)
    $lastNote = Get-Date
    while ((Get-Date).ToUniversalTime() -lt $start) {
        if (((Get-Date) - $lastNote).TotalMinutes -ge 30) { Log "waiting; $([int]($start - (Get-Date).ToUniversalTime()).TotalMinutes) min to go"; $lastNote = Get-Date }
        Start-Sleep -Seconds 30
    }
    Log "start time reached ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') local)"

    # 3. guards
    $running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*tickforge*capture*" }
    if ($running) { Log "REFUSED: a capture is already running (PID $($running.ProcessId))"; exit 2 }
    if (Test-Path $partition) { Log "REFUSED: partition already exists: $partition"; exit 2 }

    # 4. capture, TickForge's own script, defaults unchanged
    $env:UV_PROJECT_ENVIRONMENT = "C:\venvs\tickforge"
    $capLog = Join-Path $logDir "batch-1-capture-$Date.wrapper.log"
    Log "launching run_capture.ps1 -Hours $Hours -Symbol $Symbol -Root $Root  (wrapper output -> $capLog)"
    $cap = Start-Process -FilePath "powershell" -WorkingDirectory $TickForge -WindowStyle Hidden -PassThru `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts\run_capture.ps1", "-Hours", $Hours, "-Symbol", $Symbol, "-Root", $Root) `
        -RedirectStandardOutput $capLog -RedirectStandardError "$capLog.err"
    Log "capture wrapper PID $($cap.Id)"
    $cap.WaitForExit()
    Log "capture wrapper exited with code $($cap.ExitCode) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    # 5. verify closure
    if (-not (Test-Path $partition)) { Log "FAILED: no partition at $partition after capture"; exit 3 }
    $files = Get-ChildItem $partition -Filter *.parquet
    Log "partition has $($files.Count) parquet files, $([int](($files | Measure-Object Length -Sum).Sum / 1MB)) MB"
    $py = Join-Path $repo ".venv\Scripts\python.exe"
    $env:PYTHONIOENCODING = "utf-8"
    $facts = & $py -c "import json; from microflux.experiment import session_eligibility; print(json.dumps(session_eligibility(r'$($Root -replace '\\','/')', '$Symbol', '$Date')))" 2>$null
    Log "eligibility facts: $facts"

    # 6. hand off to the frozen pipeline (its gate records acceptance or exclusion)
    Log "launching run_detached.ps1 -Date $Date -Batch $Batch"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\run_detached.ps1") -Date $Date -Symbol $Symbol -Root ($Root -replace '\\', '/') -Batch $Batch 2>&1 | ForEach-Object { Log "  $_" }

    # 7. wait for the pipeline's exit marker
    $runLog = Join-Path $repo "runs\$Symbol-$Date\run.log"
    while (-not ((Test-Path $runLog) -and (Select-String -Path $runLog -Pattern "=== .* exit code" -Quiet))) { Start-Sleep -Seconds 60 }
    $done = Test-Path (Join-Path $repo "runs\$Symbol-$Date\final_test.json")
    Log "pipeline exited; final_test.json present: $done"
    Log "=== BATCH SESSION COMPLETE $Symbol-$Date"
} finally {
    $null = [Win32.Power]::SetThreadExecutionState($RELEASE)
    Log "keep-awake released"
}
