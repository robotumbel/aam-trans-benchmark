# run_gate3.ps1 - adaptive mechanism v3 study (PROTOCOL_GATE_V3.md).
#   powershell -ExecutionPolicy Bypass -File run_gate3.ps1
# Validation-only selection, then confirmation on new seeds 21-30.
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Python314\python.exe"
Set-Location $PSScriptRoot

Write-Output "[stage V3-select] validation-only selection"
& $py select_gate.py --tag gate3_select --candidates aam_trans aam_v3a aam_v3b aam_v3c

$sel = (Get-Content runs\gate3_select\SELECTED.txt -TotalCount 1).Trim()
Write-Output "[stage V3-confirm] seeds 21-30, selected = $sel"
& $py run.py --tag gate3_confirm --methods trades --save-models --seeds (21..30) `
    --backbones transformer aam_noGate aam_trans $sel

Write-Output "[done] v3 study"
