# PowerShell wrapper for VIDEX OCR smoke test
$ErrorActionPreference = "Stop"
uv run python scripts/smoke_test_ocr.py
exit $LASTEXITCODE
