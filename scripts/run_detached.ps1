# Run one session under PROTOCOL.md in a process that outlives the shell that
# started it, keeps the machine awake while it runs, and logs everything.
#
#   powershell -File scripts\run_detached.ps1 -Date 2026-09-14
#   powershell -File scripts\run_detached.ps1 -Date 2026-09-14 -Batch docs/collection-batch-1.json
#
# Launches itself detached (hidden window) unless -Inner is set; the inner
# instance holds ES_SYSTEM_REQUIRED for the duration -- the same call
# TickForge's run_capture.ps1 uses -- and releases it on exit. Output goes
# to runs/<session>/run.log and run.err. Re-running resumes: every stage
# skips itself when its artifacts already match, and neural training resumes
# from its checkpoint.

param(
    [Parameter(Mandatory = $true)] [string] $Date,
    [string] $Symbol = "BTCUSDT",
    [string] $Root = "C:/tickforge-runs",
    [string] $Batch = "docs/collection-batch-1.json",
    [string] $Session = "",
    [switch] $Inner
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if ($Session -eq "") { $Session = "$Symbol-$Date" }
$out = Join-Path $repo "runs\$Session"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out "run.log"
$err = Join-Path $out "run.err"

if (-not $Inner) {
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
              "-Date", $Date, "-Symbol", $Symbol, "-Root", $Root, "-Batch", $Batch, "-Session", $Session, "-Inner")
    $proc = Start-Process -FilePath "powershell" -ArgumentList $args -WorkingDirectory $repo -WindowStyle Hidden -PassThru
    Write-Host "detached PID $($proc.Id)   session $Session"
    Write-Host "log  $log"
    Write-Host "err  $err"
    exit 0
}

# --- inner: keep awake, run, release ---------------------------------------
$KEEP_AWAKE = [uint32] 2147483649   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$RELEASE = [uint32] 2147483648      # ES_CONTINUOUS
Add-Type -Name Power -Namespace Win32 -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$null = [Win32.Power]::SetThreadExecutionState($KEEP_AWAKE)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$python = Join-Path $repo ".venv\Scripts\python.exe"
$script = Join-Path $repo "scripts\run_session.py"
$argv = @("-u", $script, "--root", $Root, "--symbol", $Symbol, "--date", $Date, "--session", $Session)
if ($Batch -ne "") { $argv += @("--batch", $Batch) }
"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') start  PID $PID  $Session" | Out-File -Append -Encoding utf8 $log
try {
    $p = Start-Process -FilePath $python -ArgumentList $argv -WorkingDirectory $repo -NoNewWindow -PassThru `
        -RedirectStandardOutput "$log.stdout" -RedirectStandardError $err
    "python PID $($p.Id)" | Out-File -Append -Encoding utf8 $log
    $p.WaitForExit()   # Wait-Process leaves ExitCode empty; WaitForExit populates it
    "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') exit code $($p.ExitCode)" | Out-File -Append -Encoding utf8 $log
} finally {
    $null = [Win32.Power]::SetThreadExecutionState($RELEASE)
    "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') keep-awake released" | Out-File -Append -Encoding utf8 $log
}
