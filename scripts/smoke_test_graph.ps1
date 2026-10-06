# PowerShell wrapper for VIDEX Phase 7 Evidence Graph Smoke Test
Write-Host "Running VIDEX Phase 7 Evidence Graph Smoke Test..." -ForegroundColor Cyan
uv run python scripts/smoke_test_graph.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Smoke test failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "Evidence Graph smoke test completed successfully." -ForegroundColor Green
