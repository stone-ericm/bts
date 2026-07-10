#!/usr/bin/env bash
# Install the canonical BTS systemd user units from repo templates (audit F12).
#
# EXPLICIT OPERATOR ACTION — deploys never call this; the unit_drift health
# check only reports divergence. Copies scripts/systemd/*.service into
# ~/.config/systemd/user/ and daemon-reloads. Does NOT enable or (re)start
# anything: enabling is a one-time bootstrap choice, and restarts belong to
# the deploy workflow's canary.
#
# Usage: bash scripts/install-systemd-hetzner.sh [--diff]
#   --diff   show differences only; change nothing

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_DIR/scripts/systemd"
DEST_DIR="$HOME/.config/systemd/user"

if [ "${1:-}" = "--diff" ]; then
    for unit in "$SRC_DIR"/*.service; do
        name="$(basename "$unit")"
        if [ -f "$DEST_DIR/$name" ]; then
            echo "--- $name"
            diff -u "$DEST_DIR/$name" "$unit" && echo "  (identical)"
        else
            echo "--- $name: not installed"
        fi
    done
    exit 0
fi

mkdir -p "$DEST_DIR"
changed=0
for unit in "$SRC_DIR"/*.service; do
    name="$(basename "$unit")"
    if [ -f "$DEST_DIR/$name" ] && cmp -s "$unit" "$DEST_DIR/$name"; then
        echo "$name: already current"
        continue
    fi
    install -m 0644 "$unit" "$DEST_DIR/$name"
    echo "$name: installed"
    changed=1
done

if [ "$changed" = 1 ]; then
    systemctl --user daemon-reload
    echo "daemon-reload done. Units NOT restarted — use the deploy workflow,"
    echo "or 'systemctl --user restart <unit>' deliberately."
fi
