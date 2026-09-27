# run_cw.ps1 - multiclass mitigation: class-balanced CE (after run_gate.ps1).
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Python314\python.exe"
Set-Location $PSScriptRoot
while (-not (Select-String -Path run_gate.log -Pattern "\[done\] gate study" -Quiet)) {
    Start-Sleep -Seconds 300
}
Write-Output "[stage G] multiclass, class-balanced CE"
& $py run.py --tag multiclass_cw --modes multiclass --methods trades --quick `
    --class-weight --save-models --seeds (1..10) --backbones transformer aam_trans
Write-Output "[done] class-weight study"
