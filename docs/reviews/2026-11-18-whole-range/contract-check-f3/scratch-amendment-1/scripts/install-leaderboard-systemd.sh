#!/usr/bin/env bash
# Install bts-leaderboard.{service,timer} as systemd --user unit on bts-hetzner.
#
# Usage (from repo root on the SERVER):
#   bash scripts/install-leaderboard-systemd.sh install
#   bash scripts/install-leaderboard-systemd.sh status
#   bash scripts/install-leaderboard-systemd.sh remove
#
# Prerequisites (one-time, manual):
#   1. Cookies file exists at ~/.bts-leaderboard-cookies.json (chmod 600)
#      OR `pass` is installed and `pass show mlb-bts-session-cookies` works
#   2. The `bts` venv has playwright installed (only needed for refresh; the
#      scraper itself doesn't use playwright at runtime)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

action="${1:-help}"

case "$action" in
    install)
        mkdir -p "$SYSTEMD_USER_DIR"
        cp "$REPO_DIR/deploy/systemd/bts-leaderboard.service" "$SYSTEMD_USER_DIR/"
        cp "$REPO_DIR/deploy/systemd/bts-leaderboard.timer" "$SYSTEMD_USER_DIR/"
        systemctl --user daemon-reload
        systemctl --user enable --now bts-leaderboard.timer
        echo "installed bts-leaderboard.{service,timer}; enabled timer"
        echo
        systemctl --user list-timers bts-leaderboard.timer
        ;;
    status)
        systemctl --user status bts-leaderboard.timer --no-pager || true
        echo
        systemctl --user status bts-leaderboard.service --no-pager || true
        ;;
    remove)
        systemctl --user disable --now bts-leaderboard.timer 2>/dev/null || true
        rm -f "$SYSTEMD_USER_DIR/bts-leaderboard.service" "$SYSTEMD_USER_DIR/bts-leaderboard.timer"
        systemctl --user daemon-reload
        echo "removed bts-leaderboard.{service,timer}"
        ;;
    help|*)
        sed -n '2,15p' "$0"
        exit 1
        ;;
esac
