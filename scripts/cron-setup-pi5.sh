#\!/usr/bin/env bash
# BTS orchestrator cron setup for Pi5.
# All times are in system timezone (ET assumed).
#
# Schedule:
#   11:00 AM ET — Early game check
#   4:00 PM ET  — Prime time run
#   7:30 PM ET  — West coast run
#   1:00 AM ET  — Check yesterday's results
#
# Usage: bash scripts/cron-setup-pi5.sh [install|show|remove]

set -euo pipefail

BTS_DIR="$HOME/projects/bts"
CONFIG="$HOME/.bts-orchestrator.toml"
LOG_DIR="$BTS_DIR/data/picks"
UV_PREFIX="UV_CACHE_DIR=/tmp/uv-cache"
ENV_SOURCE="set -a; . $BTS_DIR/.env; set +a"
PATH_PREFIX="PATH=\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
MARKER="# BTS-ORCHESTRATOR"

# Cross-platform yesterday date
if date -v-1d >/dev/null 2>&1; then
    YESTERDAY_CMD='$(date -v-1d +\%Y-\%m-\%d)'  # macOS
else
    YESTERDAY_CMD='$(date -d "yesterday" +\%Y-\%m-\%d)'  # Linux
fi

CRON_LINES="$MARKER
0 11 * * * $PATH_PREFIX; $ENV_SOURCE; cd $BTS_DIR && $UV_PREFIX uv run bts orchestrate --date \$(date +\%Y-\%m-\%d) --config $CONFIG >> $LOG_DIR/orchestrator.log 2>&1 $MARKER
0 16 * * * $PATH_PREFIX; $ENV_SOURCE; cd $BTS_DIR && $UV_PREFIX uv run bts orchestrate --date \$(date +\%Y-\%m-\%d) --config $CONFIG >> $LOG_DIR/orchestrator.log 2>&1 $MARKER
30 19 * * * $PATH_PREFIX; $ENV_SOURCE; cd $BTS_DIR && $UV_PREFIX uv run bts orchestrate --date \$(date +\%Y-\%m-\%d) --config $CONFIG >> $LOG_DIR/orchestrator.log 2>&1 $MARKER
0 1 * * * $PATH_PREFIX; $ENV_SOURCE; cd $BTS_DIR && $UV_PREFIX uv run bts check-results --date $YESTERDAY_CMD >> $LOG_DIR/orchestrator.log 2>&1 $MARKER"

case "${1:-show}" in
    install)
        if [ \! -f "$CONFIG" ]; then
            echo "ERROR: Config not found at $CONFIG"
            echo "Copy config/orchestrator.example.toml to $CONFIG first."
            exit 1
        fi
        mkdir -p "$LOG_DIR"
        (crontab -l 2>/dev/null | grep -v "$MARKER"; echo "$CRON_LINES") | crontab -
        echo "Installed BTS orchestrator cron jobs. Verify with: crontab -l"
        ;;
    show)
        echo "Current BTS orchestrator cron entries:"
        crontab -l 2>/dev/null | grep "$MARKER" || echo "(none)"
        echo ""
        echo "Would install:"
        echo "$CRON_LINES"
        ;;
    remove)
        crontab -l 2>/dev/null | grep -v "$MARKER" | crontab -
        echo "Removed BTS orchestrator cron jobs."
        ;;
    *)
        echo "Usage: $0 [install|show|remove]"
        exit 1
        ;;
esac
