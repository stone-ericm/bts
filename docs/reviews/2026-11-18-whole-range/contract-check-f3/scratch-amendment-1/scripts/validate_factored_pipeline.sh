#!/bin/bash
# scripts/validate_factored_pipeline.sh
# Run on a Hetzner box (not Mac) to validate the factored pipeline matches the current one at seed=42.

set -euo pipefail

cd "$(dirname "$0")/.."

OUT=/tmp/factored_validation
rm -rf "$OUT"
mkdir -p "$OUT/run_a" "$OUT/run_b"

echo "=== Run A: current pipeline ==="
rm -rf experiments/results/phase1
UV_CACHE_DIR=/tmp/uv-cache BTS_LGBM_RANDOM_STATE=42 BTS_LGBM_DETERMINISTIC=1 \
    uv run bts experiment screen \
    --subset "$(cat scripts/audit_experiments.txt)" \
    --test-seasons 2024,2025
cp -r experiments/results/phase1 "$OUT/run_a/"

echo "=== Run B: factored pipeline ==="
rm -rf experiments/results/phase1
UV_CACHE_DIR=/tmp/uv-cache BTS_LGBM_RANDOM_STATE=42 BTS_LGBM_DETERMINISTIC=1 \
    uv run bts experiment screen \
    --subset "$(cat scripts/audit_experiments.txt)" \
    --test-seasons 2024,2025 \
    --use-factored
cp -r experiments/results/phase1 "$OUT/run_b/"

echo "=== Diff ==="
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/diff_audit_outputs.py "$OUT/run_a" "$OUT/run_b"
