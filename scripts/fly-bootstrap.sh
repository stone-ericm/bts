#!/bin/bash
# Cold bootstrap: populate /data on first boot from R2 + MLB API.

set -euo pipefail
log() { echo "[bootstrap] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

cd /app
mkdir -p /data/processed /data/models /data/picks /data/raw /data/lineup_posting_times

log "Downloading parquets from R2"
UV_CACHE_DIR=/tmp/uv-cache uv run bts data sync-from-r2 \
    --processed-dir /data/processed \
    --models-dir /data/models \
    || { log "sync-from-r2 failed — cannot bootstrap"; exit 2; }

# Symlink data dirs so bts CLI finds them at default relative paths
for dir in processed models picks raw lineup_posting_times; do
    rm -rf "data/$dir"
    ln -sf "/data/$dir" "data/$dir"
done

# Bootstrap state from initial snapshot if present
if [ -f /app/data/state/initial-state.json ]; then
    log "Found initial-state.json — regenerating pick files"
    UV_CACHE_DIR=/tmp/uv-cache uv run bts state regenerate \
        --snapshot /app/data/state/initial-state.json \
        --out-picks-dir /data/picks \
        || log "WARNING: state regenerate failed (may be OK if no Bluesky data yet)"
fi

# Pull current-season raw feeds
log "Pulling current-season raw feeds"
UV_CACHE_DIR=/tmp/uv-cache uv run bts data pull \
    || log "WARNING: data pull failed (will retry on next cron)"

# Build current-season parquet
log "Building current-season parquet"
UV_CACHE_DIR=/tmp/uv-cache uv run bts data build \
    || log "WARNING: data build failed (will retry on next cron)"

# Write default orchestrator config if not present
if [ ! -f /data/orchestrator.toml ]; then
    log "Writing default orchestrator config"
    cat > /data/orchestrator.toml <<'EOF'
[orchestrator]
picks_dir = "/data/picks"
heartbeat_path = "/data/.heartbeat"

[bluesky]
dm_recipient = "did:plc:replace-me"

[scheduler]
lineup_check_offset_min = 60
fallback_deadline_min = 35
missed_pick_alert_min = 30
early_lock_gap = 0.03
cluster_min = 10
doubleheader_recheck_min = 15
results_poll_interval_min = 15
results_cap_hour_et = 5
default_init_hour_et = 10
early_game_buffer_min = 60

[[tiers]]
name = "local"
type = "local"
EOF
fi

log "Bootstrap complete"
