#!/bin/zsh
set -u
review=.codex-review/wr-k2-r1
archived=/tmp/wr-k2-cx-base

run_probe() {
    local probe=$1
    local revision=$2
    local source=src
    local archived_walk=''
    if [[ "$revision" == base ]]; then
        source="$archived/src"
        archived_walk="$archived/scripts/audit/season_walk/walk.py"
    fi
    TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$source" K2_REVIEW_EXPECT_SRC="$source" \
        K2_REVIEW_ARCHIVED_WALK="$archived_walk" \
        timeout 40 nice -n 19 .venv/bin/python "$review/probe_$probe.py" \
        > "$review/$probe-$revision.out" 2> "$review/$probe-$revision.log"
    local result=$?
    printf 'exit=%d\n' "$result" >> "$review/$probe-$revision.log"
    printf '%s %s exit=%d\n' "$probe" "$revision" "$result"
}

for selector in "$@"; do
    run_probe "$selector" base
    run_probe "$selector" k2
done
