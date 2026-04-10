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
if [ ! -f /data/processed/pa_2026.parquet ]; then
    log "No parquets on volume — running cold bootstrap"
    ./scripts/fly-bootstrap.sh
else
    log "Parquets present on volume — skipping bootstrap"
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
