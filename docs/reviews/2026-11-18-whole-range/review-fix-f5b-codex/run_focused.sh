#!/bin/bash
set -u
out=.codex-review/wr-f5b-r1
scratch=$1
export TZ=America/New_York PYTHONDONTWRITEBYTECODE=1
run() {
    label=$1
    source=$2
    limit=$3
    shift 3
    PYTHONPATH="$source" nice -n 19 timeout "$limit" .venv/bin/python "$@" > "$out/$label.log" 2>&1
    printf '%s %s\n' "$label" "$?"
}
run live-health-base "$scratch/base/src" 15 "$out/live_health_parity.py" base "$scratch"
run live-health-candidate src 15 "$out/live_health_parity.py" f5 "$scratch"
run live-health-compare src 15 "$out/live_health_parity.py" compare "$scratch"
run mutation-manual-resets-live-tiers src 95 "$out/additional_mutation.py" "$scratch"
