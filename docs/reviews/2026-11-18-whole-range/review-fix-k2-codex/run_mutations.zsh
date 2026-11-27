#!/bin/zsh
set -u
review=.codex-review/wr-k2-r1
scratch=/tmp/wr-k2-cx-mutations

run_one() {
    local mutant=$1
    local selector=$2
    local logfile="$review/mutation-$mutant.log"
    local source="$scratch/$mutant/src"
    TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$source" K2_REVIEW_EXPECT_SRC="$source" \
        timeout 45 nice -n 19 .venv/bin/python "$review/offline.py" "$selector" -x > "$logfile" 2>&1
    local result=$?
    printf 'exit=%d\n' "$result" >> "$logfile"
    printf '%s ' "$mutant"
    rg ' (passed|failed|error).* in ' "$logfile" | tail -1
}

target=tests/test_wr_k2_season_boundary.py
run_one unconditional-floor "$target::test_selection_and_explicit_context"
run_one no-early-floor "$target"
run_one no-context-floor "$target::test_selection_and_explicit_context"
run_one warning-first "$target::test_real_morning_refusal_precedes_boundary"
run_one no-official-date "$target::test_fallback_warning_and_unclassifiable_rows"
run_one synchronous-send "$target::test_boundary_transport_can_outlive_wait"
run_one plain-dispatch "$target::test_warning_unconfirmed_send_retried"
