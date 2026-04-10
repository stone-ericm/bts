#!/bin/bash
# Phase 3 rollback: Fly → Pi5 authoritative.

set -euo pipefail
confirm() {
    read -p "$1 (yes/no): " answer
    [ "$answer" = "yes" ] || { echo "Aborted."; exit 1; }
}
log() { echo ""; echo "=== $* ==="; echo ""; }

log "Phase 3 Rollback — Fly → Pi5"
confirm "Proceed with rollback?"

log "Step 1: Restart Pi5 services"
ssh stonehengee@pi5.local 'systemctl --user start bts-scheduler bts-dashboard'
sleep 5
ssh stonehengee@pi5.local 'systemctl --user status bts-scheduler --no-pager' || true

log "Step 2: Pause Fly machine"
MACHINE_ID=$(flyctl machines list -a bts-mlb --json | python3 -c "import sys,json; print(json.loads(sys.stdin.read())[0]['id'])")
flyctl machines stop "$MACHINE_ID" -a bts-mlb

log "Rollback complete. Pi5 is authoritative again."
echo "Next: investigate, fix, re-run Phase 2 shadow validation."
