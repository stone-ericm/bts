#!/usr/bin/env bash
# BTS cron setup for Hetzner production server.
#
# All times are in system timezone (must be America/New_York / ET).
#
# Schedule (ET):
#   01:00 — check yesterday's results, update streak
#   02:00 — reconcile (re-check picks for scoring changes)
#   03:00 — nightly data refresh + sync to R2 + tomorrow's preview pick
#   */5  — lineup posting time collection
#   */5  — healthchecks.io ping
#
# IMPORTANT: cron's default shell is /bin/sh (= dash on Debian). dash has no
# `source` builtin — use `. ./.env` instead. Forgetting this kills every
# cron job before it touches the bts CLI.
#
# Usage: bash scripts/cron-setup-hetzner.sh [install|show|remove]

set -euo pipefail

BTS_DIR="$HOME/projects/bts"
LOG_DIR="$HOME/logs"
UV_BIN="$HOME/.local/bin/uv"
HC_PING_URL="${HEALTHCHECKS_PING_URL:-https://hc-ping.com/25ebdf3f-b784-4c6b-981c-7e5ea16218c8}"
MARKER="# BTS-HETZNER"

# Common prefix: cd, load .env via dot (POSIX), guard exports
PREFIX="cd $BTS_DIR && set -a && . ./.env && set +a &&"
YESTERDAY='$(date -d yesterday +\%Y-\%m-\%d)'

CRON_LINES="$MARKER
0 1 * * * $PREFIX $UV_BIN run bts check-results --date $YESTERDAY >> $LOG_DIR/cron.log 2>&1 $MARKER
0 2 * * * $PREFIX $UV_BIN run bts reconcile >> $LOG_DIR/cron.log 2>&1 $MARKER
0 3 * * * $PREFIX $UV_BIN run bts data pull && $UV_BIN run bts data build --seasons 2026 && $UV_BIN run bts data sync-to-r2 && $UV_BIN run bts preview >> $LOG_DIR/cron.log 2>&1 $MARKER
*/5 * * * * $PREFIX $UV_BIN run bts data collect-lineup-times --out-dir data/lineup_posting_times > /dev/null 2>&1 $MARKER
*/5 * * * * curl -fsS --max-time 5 $HC_PING_URL > /dev/null 2>&1 $MARKER"

case "${1:-show}" in
    install)
        if [ ! -f "$BTS_DIR/.env" ]; then
            echo "ERROR: $BTS_DIR/.env not found." >&2
            exit 1
        fi
        mkdir -p "$LOG_DIR"
        (crontab -l 2>/dev/null | grep -v "$MARKER"; echo "$CRON_LINES") | crontab -
        echo "Installed BTS Hetzner cron jobs. Verify with: crontab -l"
        ;;
    show)
        echo "Current BTS-HETZNER cron entries:"
        crontab -l 2>/dev/null | grep "$MARKER" || echo "(none)"
        echo ""
        echo "Would install:"
        echo "$CRON_LINES"
        ;;
    remove)
        crontab -l 2>/dev/null | grep -v "$MARKER" | crontab -
        echo "Removed BTS Hetzner cron jobs."
        ;;
    *)
        echo "Usage: $0 [install|show|remove]"
        exit 1
        ;;
esac
