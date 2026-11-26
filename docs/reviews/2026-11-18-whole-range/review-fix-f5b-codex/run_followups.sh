#!/bin/bash
set -u
out=.codex-review/wr-f5b-r1
scratch=$1
export TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
run() {
    label=$1
    shift
    nice -n 19 timeout 40 .venv/bin/python "$@" > "$out/$label.log" 2>&1
    printf '%s %s\n' "$label" "$?"
}
run health-season-inventory "$out/receipt_inventory_probe.py"
run health-live-precedence-no-seal "$out/receipt_inventory_probe.py" precedence
run contract-edges "$out/contract_edges_probe.py" "$scratch"
run readers "$out/readers_probe.py"
run archive-operator-rerun "$out/replay_archives.py" operator_text contract main "$scratch"
PYTHONPATH="$scratch/base/src" run existing-audit-base -m pytest -p no:cacheprovider -q tests/test_audit_driver.py
