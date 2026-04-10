#!/bin/bash
# BTS Fly container entrypoint.
#
# Starts (in order):
# 1. Tailscale daemon (joins tailnet as tag:bts-prod)
# 2. Cold bootstrap if data volume is empty
# 3. Dashboard web server
# 4. Scheduler daemon
# 5. Cron loop (background)

set -euo pipefail

log() { echo "[entrypoint] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

log "Starting BTS Fly entrypoint"

# --- 1. Tailscale ---
if [ -n "${TS_AUTHKEY:-}" ]; then
    log "Starting Tailscale daemon"
    mkdir -p /var/lib/tailscale
    tailscaled --state=/var/lib/tailscale/tailscaled.state \
               --socket=/var/run/tailscale/tailscaled.sock &
    TAILSCALED_PID=$!

    for i in $(seq 1 30); do
        if tailscale status >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    log "Joining tailnet with ephemeral key"
    tailscale up \
        --authkey="${TS_AUTHKEY}" \
        --hostname="bts-fly" \
        --advertise-tags=tag:bts-prod \
        --accept-routes=false \
        --ssh=false
    log "Tailscale joined: $(tailscale ip -4)"
else
    log "WARNING: TS_AUTHKEY not set — Tailscale will not join the tailnet"
fi

# --- 2. Cold bootstrap if needed ---
cd /app

# --- 2a. Symlink /app/data dirs to the persistent volume ---
# Must happen on EVERY boot because the Docker image may not include
# these dirs, and deploys can overwrite symlinks with real directories.
log "Linking data dirs to volume"
mkdir -p /app/data
for dir in processed models picks raw lineup_posting_times; do
    rm -rf "/app/data/$dir"
    ln -sf "/data/$dir" "/app/data/$dir"
done

if [ ! -f /data/processed/pa_2026.parquet ]; then
    log "No parquets on volume — running cold bootstrap"
    ./scripts/fly-bootstrap.sh
else
    log "Parquets present on volume — skipping bootstrap"
fi

# --- 2b. Ensure orchestrator config exists (independent of bootstrap) ---
if [ ! -f /data/orchestrator.toml ]; then
    log "Writing default orchestrator config"
    cat > /data/orchestrator.toml <<'TOML'
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
shadow_mode = true

[[tiers]]
name = "local"
type = "local"
TOML
fi

# --- 3. Dashboard in background ---
# web.py uses BTS_HEARTBEAT_PATH env var and listens on 0.0.0.0:3003
log "Starting dashboard on 0.0.0.0:3003"
export BTS_HEARTBEAT_PATH=/data/.heartbeat
UV_CACHE_DIR=/tmp/uv-cache uv run python -m bts.web &
DASHBOARD_PID=$!
log "Dashboard PID: $DASHBOARD_PID"

# --- 4. Cron loop in background ---
log "Starting cron loop"
./scripts/fly-cron-loop.sh &
CRON_PID=$!
log "Cron PID: $CRON_PID"

# --- 5. Scheduler daemon (foreground) ---
log "Starting scheduler daemon"

cleanup() {
    log "Shutting down"
    kill "$DASHBOARD_PID" "$CRON_PID" 2>/dev/null || true
    if [ -n "${TAILSCALED_PID:-}" ]; then
        kill "$TAILSCALED_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGTERM SIGINT

while true; do
    UV_CACHE_DIR=/tmp/uv-cache uv run bts schedule --config /data/orchestrator.toml || true
    log "Scheduler exited, restarting in 60s"
    sleep 60
done
