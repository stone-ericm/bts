#!/bin/bash
# Run Mac's share of Phase 1 experiments sequentially.
#
# Updated 2026-04-29: defaults to scripts/audit_experiments.txt (auto-derived
# from the registry) so this script can't drift past unregistered experiments.
# Previously hard-coded a 7-experiment subset that included
# venn_abers_width (unregistered 2026-04-29) and quantile_gated_skip (never
# in registry) — both would crash. Pass --subset to override.
#
# Usage:
#   ./scripts/run_mac_experiments.sh                 # full registry
#   ./scripts/run_mac_experiments.sh kl_divergence   # comma-separated subset
#
# Baseline scorecard is cached after the first experiment that needs it.

set -e
export UV_CACHE_DIR=/tmp/uv-cache

if [ -n "$1" ]; then
    SUBSET_FLAG="--subset $1"
    LABEL="subset: $1"
else
    SUBSET_FLAG="--subset $(cat scripts/audit_experiments.txt)"
    LABEL="full registry from scripts/audit_experiments.txt"
fi

echo "=== Mac Phase 1: $LABEL ==="
echo "Started: $(date)"

uv run bts experiment screen $SUBSET_FLAG --test-seasons 2024,2025

echo "=== Mac Phase 1 complete: $(date) ==="
echo "Run 'uv run bts experiment summary' to see results."
