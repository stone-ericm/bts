#!/bin/zsh
set -eu

log_dir=.codex-review/wr-c2b
suite=tests/test_wr_c2_receipt_identity.py

run_mutant() {
  local variant=$1 selection=$2 expected=$3
  local running rc=0
  if running=$(pgrep -f yield_run.sh); then
    print -u2 "Shared wrapper is occupied by $running"
    return 2
  else
    [[ $? == 1 ]] || return 2
  fi
  ~/bts-runs/yield_run.sh "$log_dir/pytest-$variant.log" env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$log_dir/scratch/$variant/src:." .venv/bin/python -m pytest -p no:cacheprovider \
    -o pythonpath= "$suite" -k "$selection" -q --tb=short || rc=$?
  [[ $rc == 1 ]]
  local failed
  failed=$(rg -c "^FAILED $suite::" "$log_dir/pytest-$variant.log")
  [[ $failed == $expected ]]
  rg "^FAILED $suite::" "$log_dir/pytest-$variant.log"
}

run_mutant drop-keys extra-error 2
run_mutant allow-superset extra-unknown 2
run_mutant allow-duplicates 'duplicate-status or test_reader_rejects_duplicate_members_inside_nested_objects' 3
run_mutant drop-hash-check foreign-body 2
run_mutant hash-constant test_receipt_hash_matches_exact_transport_body 2
run_mutant hash-alert-message test_receipt_hash_matches_exact_transport_body 2
run_mutant compare-day-only foreign-body 2
