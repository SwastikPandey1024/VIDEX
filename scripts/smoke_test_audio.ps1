# PowerShell wrapper for VIDEX audio smoke test
$ErrorActionPreference = "Stop"
uv run python scripts/smoke_test_audio.py
exit $LASTEXITCODE
