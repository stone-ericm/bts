#!/bin/bash
# Lightweight cron loop for BTS periodic jobs.
#
# Schedule (ET):
# - 01:00  check-results
# - 02:00  reconcile
# - 03:00  data pull + build + sync-to-r2
# - */5    lineup time collection
# - */5    healthchecks.io ping

set -euo pipefail
log() { echo "[cron] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

cd /app
export UV_CACHE_DIR=/tmp/uv-cache

HC_PING_URL="${HEALTHCHECKS_PING_URL:-}"

while true; do
    HOUR=$(TZ=America/New_York date +%H)
    MIN=$(TZ=America/New_York date +%M)
    YESTERDAY=$(TZ=America/New_York date -d "yesterday" +%Y-%m-%d 2>/dev/null || TZ=America/New_York date -v-1d +%Y-%m-%d)

    # 01:00 ET check-results
    if [ "$HOUR" = "01" ] && [ "$MIN" = "00" ]; then
        log "Running check-results for $YESTERDAY"
        uv run bts check-results --date "$YESTERDAY" || log "check-results failed"
    fi

    # 02:00 ET reconcile
    if [ "$HOUR" = "02" ] && [ "$MIN" = "00" ]; then
        log "Running reconcile"
        uv run bts reconcile || log "reconcile failed"
    fi

    # 03:00 ET data pull + build + sync-to-r2
    if [ "$HOUR" = "03" ] && [ "$MIN" = "00" ]; then
        log "Running nightly data refresh"
        uv run bts data pull && uv run bts data build && uv run bts data sync-to-r2 \
            || log "data refresh failed"
    fi

    # Every 5 min: lineup time collection
    if [ $((10#$MIN % 5)) -eq 0 ]; then
        uv run bts data collect-lineup-times \
            --out-dir /data/lineup_posting_times \
            2>&1 | head -5 || log "collect-lineup-times failed"
    fi

    # Every 5 min: healthchecks.io ping
    if [ $((10#$MIN % 5)) -eq 0 ] && [ -n "$HC_PING_URL" ]; then
        curl -fsS --max-time 5 "$HC_PING_URL" >/dev/null \
            || log "healthchecks ping failed"
    fi

    sleep 60
done
