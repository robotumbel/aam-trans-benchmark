# run_v4.ps1 - v4 study (PROTOCOL_V4.md): validation selection, then
# confirmation on new seeds 41-50.
#   powershell -ExecutionPolicy Bypass -File run_v4.ps1
# Resumable: finished runs are skipped when the script is started again.
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Python314\python.exe"
Set-Location $PSScriptRoot

Write-Output "[stage V4-select] validation-only selection"
& $py select_gate.py --tag v4_select --candidates ft aam_noGate aam_trans aam_v4a aam_v4b aam_v4c aam_v4d

$combo = (& $py select_v4.py combo).Trim()
Write-Output "[stage V4-combo] $combo"
if ($combo -ne "none") {
    & $py select_gate.py --tag v4_select --candidates ft aam_noGate aam_trans aam_v4a aam_v4b aam_v4c aam_v4d $combo
}

$sel = (& $py select_v4.py final).Trim()
Write-Output "[stage V4-confirm] seeds 41-50, selected = $sel"
& $py run.py --tag v4_confirm --methods trades --save-models --seeds (41..50) `
    --datasets CICIoT2023 CICIoMT2024 TON_IoT `
    --backbones transformer ft aam_noGate aam_trans $sel

Write-Output "[done] v4 study"
