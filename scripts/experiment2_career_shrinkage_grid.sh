#!/bin/bash
# Experiment 2: Career-rate shrinkage grid search
# Runs bts simulate backtest + bts validate scorecard for K ∈ {10, 20, 50}.
# Each K = one full 5-season backtest (~2-3 hours). Total: ~6-9 hours.
#
# LAUNCH FROM THE EXP2 WORKTREE (not the main repo):
#   cd /Users/stone/projects/bts/.worktrees/exp2-early-season-stab
#   nohup bash scripts/experiment2_career_shrinkage_grid.sh > data/experiments/exp2_master.log 2>&1 & disown
#
# Prerequisites:
# - Experiment 1 grid must be COMPLETE (they share CPU / memory resources).
# - The main worktree's baseline scorecard must exist:
#   ../../data/validation/baseline-pre-experiments-20260412.json
# - This worktree's data/processed/pa_*.parquet must be populated — if
#   empty, symlink or copy from ../../data/processed/ before running.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/experiments data/validation

BASELINE=../../data/validation/baseline-pre-experiments-20260412.json
if [ ! -f "$BASELINE" ]; then
    echo "BASELINE not found at $BASELINE" >&2
    exit 1
fi

# Unset Exp1 var so only Exp2 is active
unset BTS_SHRINKAGE_K

for K in 10 20 50; do
    OUT_DIR="data/experiments/career_shrink_k${K}"
    SCORE_FILE="data/validation/exp2_k${K}.json"
    RUN_LOG="data/experiments/exp2_k${K}.log"

    echo ""
    echo "=============================================="
    echo "=== Exp2 K=${K} backtest start: $(date) ==="
    echo "=============================================="
    mkdir -p "$OUT_DIR"

    BTS_CAREER_SHRINKAGE_K=${K} UV_CACHE_DIR="$TMPDIR/uv-cache-exp2" \
        uv run bts simulate backtest \
        --seasons 2021,2022,2023,2024,2025 \
        --output-dir "$OUT_DIR" \
        --retrain-every 7 \
        > "$RUN_LOG" 2>&1

    echo "=== Exp2 K=${K} backtest done: $(date) ==="
    echo "=== Exp2 K=${K} scorecard ==="
    UV_CACHE_DIR="$TMPDIR/uv-cache-exp2" uv run bts validate scorecard \
        --profiles-dir "$OUT_DIR" \
        --save "$SCORE_FILE" \
        --diff "$BASELINE" \
        >> "$RUN_LOG" 2>&1

    echo "=== Exp2 K=${K} complete: $(date) ==="
done

echo ""
echo "=============================================="
echo "=== ALL EXP2 RUNS COMPLETE: $(date) ==="
echo "=============================================="
echo "Results:"
echo "  data/validation/exp2_k10.json"
echo "  data/validation/exp2_k20.json"
echo "  data/validation/exp2_k50.json"
