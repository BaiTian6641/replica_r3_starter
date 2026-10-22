#!/bin/sh
# Run from any directory. Uses only the project environment and cache.
set -eu
cd "$(dirname "$0")/.."
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo 'This setup targets Apple Silicon macOS.' >&2
    exit 1
fi
command -v uv >/dev/null 2>&1 || { echo 'Install uv before running setup.' >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
uv sync --locked --extra neural --extra test
run_tag="$(date -u +%Y%m%dT%H%M%SZ)-$$"
.venv/bin/python scripts/inventory.py --output "reports/hardware-macos-$run_tag.json"
.venv/bin/python scripts/check_core.py --output "reports/core-macos-$run_tag.json"
.venv/bin/python scripts/validate_data.py data/fixtures/train.jsonl data/fixtures/dev.jsonl --require-verified
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/smoke_training.py --device mps --optimizer muon8bit --output "reports/muon8bit-mps-$run_tag.json"
echo 'Setup complete. Activate with: source .venv/bin/activate'
