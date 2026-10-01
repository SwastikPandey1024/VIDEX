# PowerShell wrapper for VIDEX ingestion smoke test
$ErrorActionPreference = "Stop"
uv run python scripts/smoke_test_ingestion.py
exit $LASTEXITCODE
