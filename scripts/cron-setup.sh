#!/usr/bin/env bash
# BTS daily automation cron setup for Mac.
# All times are in the system timezone (ET assumed).
#
# Schedule (densest bucket strategy):
#   11:00 AM ET — Early game check (lock only if early is densest + P>80%)
#   4:00 PM ET  — Main run (pick from densest window, post to Bluesky)
#   1:00 AM ET  — Check yesterday's results, update streak
#
# Usage: bash scripts/cron-setup.sh [install|show|remove]

set -euo pipefail

BTS_DIR="/Users/stone/projects/bts"
LOG_DIR="$BTS_DIR/data/picks"
UV_PREFIX="UV_CACHE_DIR=/tmp/uv-cache"
MARKER="# BTS-AUTOMATION"

# Cross-platform yesterday date
if date -v-1d >/dev/null 2>&1; then
    YESTERDAY_CMD='$(date -v-1d +\%Y-\%m-\%d)'  # macOS
else
    YESTERDAY_CMD='$(date -d "yesterday" +\%Y-\%m-\%d)'  # Linux
fi

# Log rotation function — keeps automation.log from growing unbounded
rotate_log() {
    local LOG_FILE="$LOG_DIR/automation.log"
    if [ -f "$LOG_FILE" ] && [ $(wc -l < "$LOG_FILE") -gt 10000 ]; then
        tail -1000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
    fi
}

CRON_LINES="$MARKER
0 11 * * * cd $BTS_DIR && $UV_PREFIX uv run bts run --date \$(date +\%Y-\%m-\%d) >> $LOG_DIR/automation.log 2>&1 $MARKER
0 16 * * * cd $BTS_DIR && $UV_PREFIX uv run bts run --date \$(date +\%Y-\%m-\%d) >> $LOG_DIR/automation.log 2>&1 $MARKER
0 1 * * * cd $BTS_DIR && $UV_PREFIX uv run bts check-results --date $YESTERDAY_CMD >> $LOG_DIR/automation.log 2>&1 $MARKER"

case "${1:-show}" in
    install)
        mkdir -p "$LOG_DIR"
        # Rotate log before installing
        rotate_log
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
    rotate)
        rotate_log
        echo "Log rotation complete."
        ;;
    *)
        echo "Usage: $0 [install|show|remove|rotate]"
        exit 1
        ;;
esac
