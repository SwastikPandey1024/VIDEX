# PowerShell wrapper for VIDEX events smoke test
$ErrorActionPreference = "Stop"
uv run python scripts/smoke_test_events.py
exit $LASTEXITCODE
