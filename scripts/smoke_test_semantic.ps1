# PowerShell wrapper for VIDEX semantic router smoke test
$ErrorActionPreference = "Stop"
uv run python scripts/smoke_test_semantic.py
exit $LASTEXITCODE
