#!/usr/bin/env bash
# BTS daily automation cron setup for Mac.
# All times are in the system timezone (ET assumed).
#
# Schedule:
#   11:00 AM ET — Early games (1-3pm starts)
#   4:00 PM ET  — Bulk games (6-8pm starts)
#   7:30 PM ET  — West coast (9-10pm starts), always posts
#   1:00 AM ET  — Check yesterday's results, update streak
#
# Usage: bash scripts/cron-setup.sh [install|show|remove]

set -euo pipefail

BTS_DIR="/Users/stone/projects/bts"
LOG_DIR="$BTS_DIR/data/picks"
UV_PREFIX="UV_CACHE_DIR=/tmp/uv-cache"
MARKER="# BTS-AUTOMATION"

CRON_LINES="$MARKER
0 11 * * * cd $BTS_DIR && $UV_PREFIX uv run bts run --date \$(date +\%Y-\%m-\%d) >> $LOG_DIR/automation.log 2>&1 $MARKER
0 16 * * * cd $BTS_DIR && $UV_PREFIX uv run bts run --date \$(date +\%Y-\%m-\%d) >> $LOG_DIR/automation.log 2>&1 $MARKER
30 19 * * * cd $BTS_DIR && $UV_PREFIX uv run bts run --date \$(date +\%Y-\%m-\%d) >> $LOG_DIR/automation.log 2>&1 $MARKER
0 1 * * * cd $BTS_DIR && $UV_PREFIX uv run bts check-results --date \$(date -v-1d +\%Y-\%m-\%d) >> $LOG_DIR/automation.log 2>&1 $MARKER"

case "${1:-show}" in
    install)
        mkdir -p "$LOG_DIR"
        # Remove old BTS entries, add new ones
        (crontab -l 2>/dev/null | grep -v "$MARKER"; echo "$CRON_LINES") | crontab -
        echo "Installed BTS cron jobs. Verify with: crontab -l"
        ;;
    show)
        echo "Current BTS cron entries:"
        crontab -l 2>/dev/null | grep "$MARKER" || echo "(none)"
        echo ""
        echo "Would install:"
        echo "$CRON_LINES"
        ;;
    remove)
        crontab -l 2>/dev/null | grep -v "$MARKER" | crontab -
        echo "Removed BTS cron jobs."
        ;;
    *)
        echo "Usage: $0 [install|show|remove]"
        exit 1
        ;;
esac
