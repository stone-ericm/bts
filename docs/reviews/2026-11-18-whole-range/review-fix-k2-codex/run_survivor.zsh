#!/bin/zsh
set -u
review=.codex-review/wr-k2-r1
scratch=/tmp/wr-k2-cx-mutations/no-early-floor/src
tests=(${(f)"$(rg '^tests/.*\.py$' "$review/matched-files.txt")"})
special=(tests/test_audit_driver.py tests/test_shadow_scheduler.py tests/test_wr_h_longest.py)
regular=()
for file in $tests; do
    if (( ${special[(Ie)$file]} == 0 )); then
        regular+=("$file")
    fi
done

run_batch() {
    local label=$1
    shift
    local logfile="$review/survivor-$label.log"
    printf '%s\n' "$@" > "$logfile"
    TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$scratch" K2_REVIEW_EXPECT_SRC="$scratch" \
        timeout 110 nice -n 19 .venv/bin/python "$review/offline.py" "$@" >> "$logfile" 2>&1
    local result=$?
    printf 'exit=%d\n' "$result" >> "$logfile"
    printf '%s ' "$label"
    rg ' (passed|failed|error|skipped|xfailed).* in |^exit=' "$logfile" | tail -2
}

for file in $special; do
    run_batch "${${file:t}%.py}" "$file"
done
integer index=1
integer batch=1
while (( index <= ${#regular} )); do
    run_batch "group-$batch" "${regular[@][$index,$((index + 5))]}"
    index=$((index + 6))
    batch=$((batch + 1))
done
run_batch health tests/health/test_tail_policy_health.py
