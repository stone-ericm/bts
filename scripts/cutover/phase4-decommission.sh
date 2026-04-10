#!/bin/bash
# Phase 4: permanently retire Pi5/Alienware from BTS.

set -euo pipefail
confirm() {
    read -p "$1 (yes/no): " answer
    [ "$answer" = "yes" ] || { echo "Aborted."; exit 1; }
}
log() { echo ""; echo "=== $* ==="; echo ""; }

log "Phase 4 Decommission"
confirm "Has Fly run stably for at least 48 hours?"
confirm "Permanently retire Pi5 BTS services?"

log "Step 1: Disable Pi5 BTS services"
ssh stonehengee@pi5.local '
    systemctl --user disable bts-scheduler bts-dashboard 2>/dev/null || true
    rm -f ~/.config/systemd/user/bts-scheduler.service
    rm -f ~/.config/systemd/user/bts-dashboard.service
    systemctl --user daemon-reload
'
echo "Pi5 BTS systemd units removed"

log "Step 2: Remove Mac BTS crons"
echo "Manual: run 'crontab -e' and remove bts data pull/build lines"
confirm "Done removing Mac crons?"

log "Step 3: Retire Alienware BTS"
echo "Manual: on Alienware, delete C:\\Users\\stone\\projects\\bts"
confirm "Done?"

log "Phase 4 complete."
echo "Cleanup PR remaining: delete src/bts/orchestrator.py, update docs."
