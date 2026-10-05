# run_v5.ps1 - v5 study (PROTOCOL_V5.md). Resumable: finished runs are skipped.
#   powershell -ExecutionPolicy Bypass -File run_v5.ps1
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Python314\python.exe"
Set-Location $PSScriptRoot
$two = "CICIoT2023", "TON_IoT"

Write-Output "[stage B1] factorial, group-token cells"
& $py run.py --tag factorial --methods trades --quick --datasets $two --backbones grp_none grp_learn

Write-Output "[stage A1] ft, TRADES, seeds 1-10, three datasets"
& $py run.py --tag main --methods trades --save-models --backbones ft

Write-Output "[stage B2] factorial, 40-token cells"
& $py run.py --tag factorial --methods trades --quick --datasets $two `
    --backbones ft_sin ft_learn ft_nobias shared_none shared_learn

Write-Output "[stage A2] ft, PGD-AT"
& $py run.py --tag main --methods pgdat --save-models --datasets $two --backbones ft

Write-Output "[stage A3] ft, multiclass"
& $py run.py --tag multiclass --modes multiclass --quick --save-models --backbones ft

Write-Output "[stage C] stronger attacks on round-four checkpoints"
& $py eval_strong.py

Write-Output "[stage A4] attacker-controllable features, ft"
& $py eval_realistic.py --backbones ft --methods trades

Write-Output "[done] v5 study"
