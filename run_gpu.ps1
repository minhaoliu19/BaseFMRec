param(
    [ValidateSet('amazon_beauty','ml-100k','steam','yelp')]
    [string]$Dataset = 'amazon_beauty',
    [int]$Epochs = 500,
    [int]$EvalInterval = 20
)
$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '0'
Push-Location $PSScriptRoot
try {
    & "$PSScriptRoot\.venv\Scripts\python.exe" -u "$PSScriptRoot\src\main.py" --dataset $Dataset --device cuda --epochs $Epochs --eval_interval $EvalInterval --log_file "runs/$Dataset/"
    if ($LASTEXITCODE -ne 0) { throw "FMRec exited with code $LASTEXITCODE" }
} finally { Pop-Location }
