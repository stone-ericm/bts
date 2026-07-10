#!/usr/bin/env bash
# BTS cron setup for Hetzner production server.
#
# All times are in system timezone (must be America/New_York / ET).
#
# Schedule (ET):
#   01:00 — check yesterday's results, update streak
#   02:00 — reconcile (re-check picks for scoring changes)
#   01:10,02:10,10:30,13:30 — fetch real MLB BTS account streak (contest state); alerts if stale/failed
#   23:30 — skip-policy shadow: record today's pick-vs-skip-band decision + reconcile (docs/audit/2026-06-20-skip-policy-shadow.md)
#   03:00 — nightly data refresh + sync to R2 + tomorrow's preview pick
#   */5  — lineup posting time collection
#   */5  — healthchecks.io ping
#   */5  — scheduler heartbeat staleness check (pings hc-ping /fail on stale)
#   */30 — capture BTS public static JSONs (pregame consensus + lookups; content-deduped, no auth)
#   */15 10-23 — DM if today's delivered pick was never entered in the MLB app (pre-first-pitch window)
#   20 */3 — restic 'ops' backup: data/picks + data/health_state to R2 (audit F5; needs RESTIC_PASSWORD in .env + scripts/install-restic-hetzner.sh)
#   50 4  — restic 'archive' backup: leaderboard / hetzner_results / external
#   35 5 Sun — restic prune (reclaims space from forgotten snapshots)
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
HC_PING_URL="${HEALTHCHECKS_PING_URL:?set HEALTHCHECKS_PING_URL (e.g. in .env, then: set -a && . ./.env && set +a) — refusing to bake a hardcoded ping URL into cron}"
MARKER="# BTS-HETZNER"

# Common prefix: cd, load .env via dot (POSIX), guard exports
PREFIX="cd $BTS_DIR && set -a && . ./.env && set +a &&"
YESTERDAY='$(date -d yesterday +\%Y-\%m-\%d)'
# bts data pull requires --start/--end. A 7-day backfill window catches any
# late MLB scoring corrections without re-downloading the full season.
DATA_PULL_START='$(date -d "7 days ago" +\%Y-\%m-\%d)'
DATA_PULL_END='$(date +\%Y-\%m-\%d)'

# 3am chain ordering: data pull -> build -> { preview ; sync-to-r2 }.
# preview is the user-facing deliverable (drives the morning dashboard); R2
# sync is nice-to-have backup. Wrapping the bts commands in a subshell so the
# log redirect captures ALL of their output (without it, sh's redirect only
# binds to the last command). preview and sync are grouped AFTER `&& build`, so
# NEITHER runs if pull/build fails -- a failed build must not publish a new-schema
# R2 manifest over stale parquets. The `;` between them inside the group keeps them
# decoupled: a preview error doesn't block the R2 backup, and a sync failure (R2
# outage) doesn't block preview.
CRON_LINES="$MARKER
0 1 * * * $PREFIX $UV_BIN run bts check-results --date $YESTERDAY >> $LOG_DIR/cron.log 2>&1 $MARKER
0 2 * * * $PREFIX $UV_BIN run bts reconcile >> $LOG_DIR/cron.log 2>&1 $MARKER
10 1 * * * $PREFIX $UV_BIN run bts fetch-contest-streak --picks-dir data/picks --expected-username stonehengee --dm-recipient stonehengee.bsky.social >> $LOG_DIR/cron.log 2>&1 $MARKER
10 2 * * * $PREFIX $UV_BIN run bts fetch-contest-streak --picks-dir data/picks --expected-username stonehengee --dm-recipient stonehengee.bsky.social >> $LOG_DIR/cron.log 2>&1 $MARKER
30 10 * * * $PREFIX $UV_BIN run bts fetch-contest-streak --picks-dir data/picks --expected-username stonehengee --dm-recipient stonehengee.bsky.social >> $LOG_DIR/cron.log 2>&1 $MARKER
30 13 * * * $PREFIX $UV_BIN run bts fetch-contest-streak --picks-dir data/picks --expected-username stonehengee --dm-recipient stonehengee.bsky.social >> $LOG_DIR/cron.log 2>&1 $MARKER
30 23 * * * $PREFIX $UV_BIN run bts skip-policy-shadow-update >> $LOG_DIR/cron.log 2>&1 $MARKER
45 7 * * * $PREFIX $UV_BIN run bts park-drag-refresh >> $LOG_DIR/park_drag.log 2>&1 $MARKER
0 3 * * * $PREFIX ($UV_BIN run bts data pull --start $DATA_PULL_START --end $DATA_PULL_END && $UV_BIN run bts data build --seasons 2026 && { $UV_BIN run bts preview ; $UV_BIN run bts data sync-to-r2 ; }) >> $LOG_DIR/cron.log 2>&1 $MARKER
*/5 * * * * $PREFIX $UV_BIN run bts data collect-lineup-times --out-dir data/lineup_posting_times > /dev/null 2>&1 $MARKER
*/5 * * * * curl -fsS --max-time 5 $HC_PING_URL > /dev/null 2>&1 $MARKER
*/5 * * * * $PREFIX $UV_BIN run python scripts/check_heartbeat.py --heartbeat-path data/.heartbeat --ping-url \"\$BTS_SCHEDULER_HEARTBEAT_PING_URL\" >> $LOG_DIR/heartbeat.log 2>&1 $MARKER
*/30 * * * * $PREFIX $UV_BIN run bts leaderboard capture-static >> $LOG_DIR/static_capture.log 2>&1 $MARKER
*/15 10-23 * * * $PREFIX $UV_BIN run bts check-pick-entered --picks-dir data/picks --expected-username stonehengee --dm-recipient stonehengee.bsky.social >> $LOG_DIR/cron.log 2>&1 $MARKER
20 */3 * * * $PREFIX $UV_BIN run bts backup run --set ops >> $LOG_DIR/backup.log 2>&1 $MARKER
50 4 * * * $PREFIX $UV_BIN run bts backup run --set archive >> $LOG_DIR/backup.log 2>&1 $MARKER
35 5 * * 0 $PREFIX $UV_BIN run bts backup prune >> $LOG_DIR/backup.log 2>&1 $MARKER"

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
