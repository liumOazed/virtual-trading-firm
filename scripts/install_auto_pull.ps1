# install_auto_pull.ps1
# ======================
# One-time setup on your Windows PC: registers a Task Scheduler job that runs
#   python scripts/state_bundle.py pull
# every day at 12:00 (noon, after GitHub's run has finished even when it starts
# late) and whenever you log in, so the CSVs in this folder always show the
# latest numbers from the GitHub Actions run. If the PC was off at noon, the job
# runs as soon as it's back on.
#
# Run once from the repo folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\install_auto_pull.ps1
#
# Remove it later with:
#   Unregister-ScheduledTask -TaskName "VTF daily state pull" -Confirm:$false

$ErrorActionPreference = "Stop"
$repo   = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python -ErrorAction Stop).Source
$log    = Join-Path $repo "state_pull.log"

if (-not (Test-Path (Join-Path $repo ".state_key"))) {
    throw ".state_key not found in $repo - run 'python scripts/state_bundle.py keygen' setup first."
}

$cmd = "cd /d `"$repo`" && `"$python`" scripts\state_bundle.py pull >> `"$log`" 2>&1"
$action   = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $cmd" -WorkingDirectory $repo
$triggers = @(
    (New-ScheduledTaskTrigger -Daily -At 12:00pm),
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME)
)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask -TaskName "VTF daily state pull" -Action $action `
    -Trigger $triggers -Settings $settings -Force | Out-Null

Write-Host "Installed. Running the first pull now..."
Start-ScheduledTask -TaskName "VTF daily state pull"
Start-Sleep -Seconds 20
Get-Content $log -Tail 3
