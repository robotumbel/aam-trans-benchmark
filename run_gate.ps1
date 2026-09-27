# run_gate.ps1 - gate v2 study (PROTOCOL_GATE_V2.md), run after run_all.ps1.
#   powershell -ExecutionPolicy Bypass -File run_gate.ps1
# Waits for run_all.ps1 to finish, then runs validation-only selection, then
# the confirmation grid on new seeds 11-20 with the selected candidate.
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Python314\python.exe"
Set-Location $PSScriptRoot

while (-not (Select-String -Path run_all.log -Pattern "\[done\] all stages" -Quiet)) {
    Start-Sleep -Seconds 300
}

Write-Output "[stage E] gate v2 selection on validation data"
& $py select_gate.py

$sel = (Get-Content runs\gate_select\SELECTED.txt -TotalCount 1).Trim()
Write-Output "[stage F] confirmation, seeds 11-20, selected = $sel"
& $py run.py --tag gate_confirm --methods trades --save-models --seeds (11..20) `
    --backbones transformer aam_noGate aam_trans $sel

Write-Output "[done] gate study"
