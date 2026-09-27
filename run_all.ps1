# run_all.ps1 - full benchmark, staged so the primary comparison lands first.
# Resumable: re-run the script and completed runs (JSON present) are skipped.
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Python314\python.exe"
Set-Location $PSScriptRoot
$seeds = 1..10

Write-Output "[stage A] TRADES: vanilla, AAM-TRANS, MLP, LSTM"
& $py run.py --tag main --methods trades --save-models --seeds $seeds `
    --backbones transformer aam_trans mlp lstm

Write-Output "[stage B] TRADES: single-switch ablations"
& $py run.py --tag main --methods trades --save-models --seeds $seeds `
    --backbones aam_noGate aam_noTok aam_noPE aam_noAux vanilla_gate

Write-Output "[stage C] PGD-AT: vanilla, AAM-TRANS"
& $py run.py --tag main --methods pgdat --save-models --seeds $seeds `
    --backbones transformer aam_trans

Write-Output "[stage D] multiclass TRADES: vanilla, AAM-TRANS"
& $py run.py --tag multiclass --modes multiclass --methods trades --quick `
    --save-models --seeds $seeds --backbones transformer aam_trans

Write-Output "[done] all stages"
