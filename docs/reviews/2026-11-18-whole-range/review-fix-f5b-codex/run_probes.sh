#!/bin/bash
set -u
out=.codex-review/wr-f5b-r1
scratch=$1
export TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
run() {
    label=$1
    source=$2
    shift 2
    PYTHONPATH="$source" nice -n 19 timeout 22 .venv/bin/python "$@" > "$out/$label.log" 2>&1
    printf '%s %s\n' "$label" "$?"
}
run early src "$out/early_probe.py"
run health src "$out/health_probe.py"
run timing-morning src "$out/timing_morning_probe.py"
run parity-base "$scratch/base/src" "$out/parity_probe.py" base "$scratch"
run parity-candidate src "$out/parity_probe.py" f5 "$scratch"
run parity-compare src "$out/parity_probe.py" compare "$scratch"
for args in 'r1 alert_probe main' 'r1 alert_probe cases' 'r1 mutation_witness main' \
            'r2 state_replay main' 'r2 state_replay cases' 'r2 surviving_mutations main' \
            'r2 morning_fifo main' 'r2 fallback_recovery main' 'r2 fallback_recovery cases' \
            'contract contract_rows main' 'contract contract_rows cases' 'contract health_cases main' \
            'contract standing main' 'contract pitch_correction main' 'contract operator_text main' \
            'contract round_one main' 'contract mutations main' 'contract morning_cases main' \
            'contract watchdog_sequences main' 'contract watchdog_sequences cases' 'contract lock_boundary main'; do
    set -- $args
    run "archive-$1-$2-$3" src "$out/replay_archives.py" "$2" "$1" "$3" "$scratch"
done
carbon=.codex-review/f5-records/docs/reviews/2026-11-18-whole-range/review-fix-f5-carbon
run archive-carbon src -m pytest -p no:cacheprovider -q "$carbon/probe_07_extra_regressions.py"
