# PowerShell wrapper for VIDEX perception smoke test
$ErrorActionPreference = "Stop"
uv run python scripts/smoke_test_perception.py
exit $LASTEXITCODE
