# PowerShell wrapper for VIDEX sample video corpus evaluation
param(
    [string]$CorpusDir = "Sample_Videos",
    [string]$OutputDir = "outputs/evaluation",
    [int]$Stride = 5,
    [int]$MaxFrames = 0
)

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "VIDEX Sample Video Corpus Evaluation & Audit" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$CmdArgs = @(
    "run", "python", "scripts/evaluate_sample_corpus.py",
    "--corpus-dir", $CorpusDir,
    "--output-dir", $OutputDir,
    "--stride", $Stride
)

if ($MaxFrames -gt 0) {
    $CmdArgs += @("--max-frames", $MaxFrames)
}

& uv @CmdArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Sample corpus evaluation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
