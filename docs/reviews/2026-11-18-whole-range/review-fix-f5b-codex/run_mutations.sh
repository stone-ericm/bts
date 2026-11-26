#!/bin/bash
set -u
out=.codex-review/wr-f5b-r1
scratch=$1
export TZ=America/New_York PYTHONDONTWRITEBYTECODE=1
for case in control early-writes-live date-only no-writer-leg-binding no-sealed-receipt-suppression \
            health-prefers-early morning-ignores-base-slot early-pick-no-1mib-cap \
            early-receipt-flag-before-gate no-installed-cron-flag gate-starts-1015 \
            no-morning-leg-binding no-health-leg-binding; do
    nice -n 19 timeout 95 .venv/bin/python "$out/mutation_checks.py" "$case" "$scratch" > "$out/mutation-$case.log" 2>&1
    status=$?
    printf '%s %s\n' "$case" "$status"
    if [[ "$case" == control && "$status" != 0 ]]; then
        exit "$status"
    fi
done
