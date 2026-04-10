#!/bin/bash
# Phase 3 cutover: Pi5 → Fly authoritative.
# Interactive — prompts at each critical step.

set -euo pipefail

confirm() {
    read -p "$1 (yes/no): " answer
    if [ "$answer" != "yes" ]; then echo "Aborted."; exit 1; fi
}
log() { echo ""; echo "=== $* ==="; echo ""; }

log "Phase 3 Cutover — Pi5 → Fly"
echo "This will export state from Pi5, commit the snapshot,"
echo "disable shadow mode on Fly, and stop Pi5 BTS services."
echo ""
confirm "Ready to begin?"

log "Step 1: Export initial state from Pi5"
ssh stonehengee@pi5.local 'cd ~/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run bts state export --to data/state/initial-state.json' \
    || { echo "Export failed — check for unresolved picks."; exit 1; }

log "Step 2: Copy snapshot from Pi5"
scp stonehengee@pi5.local:~/projects/bts/data/state/initial-state.json data/state/initial-state.json

echo "Inspecting snapshot:"
python3 -c "
import json
d = json.loads(open('data/state/initial-state.json').read())
print(f'  cutoff: {d[\"cutoff_date\"]}')
print(f'  streak: {d[\"streak_at_cutoff\"]}')
print(f'  picks:  {len(d[\"historical_picks\"])}')
"
confirm "Snapshot look correct?"

log "Step 3: Commit snapshot to git"
git add data/state/initial-state.json
CUTOFF=$(python3 -c "import json; print(json.loads(open('data/state/initial-state.json').read())['cutoff_date'])")
git commit -m "migration: freeze state at cutoff ${CUTOFF}"
git push origin main

log "Step 4: Disable shadow mode on Fly"
confirm "Flip Fly to authoritative mode?"
flyctl ssh console -a bts-mlb -C "sh -c 'sed -i s/shadow_mode.*/shadow_mode\ =\ false/ /data/orchestrator.toml'"
MACHINE_ID=$(flyctl machines list -a bts-mlb --json | python3 -c "import sys,json; print(json.loads(sys.stdin.read())[0]['id'])")
flyctl machines restart "$MACHINE_ID" -a bts-mlb
echo "Waiting 60s for restart..."
sleep 60

log "Step 5: Stop Pi5 BTS services (rollback-ready)"
confirm "Stop Pi5 scheduler + dashboard?"
ssh stonehengee@pi5.local 'systemctl --user stop bts-scheduler bts-dashboard'
echo "Pi5 services stopped (still enabled for quick rollback)"

log "Step 6: Verify Fly is alive"
flyctl status -a bts-mlb
echo ""
echo "Check dashboard via Tailscale: curl http://bts-fly:3003/health"
echo ""
log "Phase 3 complete. 48-hour observation window starts now."
echo "If anything wobbles: scripts/cutover/phase3-rollback.sh"
echo "After 48h stable: scripts/cutover/phase4-decommission.sh"
