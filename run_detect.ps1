# run_detect.ps1 - gate-as-detector study (PROTOCOL_DETECT_V3.md).
#   powershell -ExecutionPolicy Bypass -File run_detect.ps1
# Run after run_gate3.ps1 has finished (both use the GPU).
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Python314\python.exe"
Set-Location $PSScriptRoot

Write-Output "[stage detect] seeds 31-40, 3 datasets"
& $py detect_v3.py --tag detect_v3
& $py analyze_detect.py detect_v3
Write-Output "[done] detect study"
