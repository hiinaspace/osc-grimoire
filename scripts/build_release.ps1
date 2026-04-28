$ErrorActionPreference = "Stop"

uv sync --group build --group dev
uv run python scripts/download_release_models.py
uv run pytest -q
uv run pyinstaller --noconfirm osc-grimoire.spec

$zipPath = "dist/osc-grimoire-windows.zip"
if (Test-Path $zipPath) {
    Remove-Item $zipPath
}
Compress-Archive -Path "dist/osc-grimoire" -DestinationPath $zipPath
Write-Host "Wrote $zipPath"
