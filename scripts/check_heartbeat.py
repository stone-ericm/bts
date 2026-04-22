#!/usr/bin/env python3
"""Heartbeat staleness checker — runs every 5 min via cron on bts-hetzner.

Usage:
    python3 scripts/check_heartbeat.py [--heartbeat-path PATH] [--ping-url URL]

Returns:
    Exit code 0 if fresh. Exit code 1 + POST to hc-ping /fail if stale.

Integration: invoke from cron like
    */5 * * * * cd /home/bts/projects/bts && /home/bts/.local/bin/uv run \\
        python scripts/check_heartbeat.py --heartbeat-path data/.heartbeat \\
        --ping-url "$BTS_SCHEDULER_HEARTBEAT_PING_URL" \\
        >> /home/bts/logs/heartbeat.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# State strings must match bts.heartbeat.HeartbeatState constants.
# If those change, update here in lockstep.
# State -> staleness thresholds (seconds)
RUNNING_MAX_AGE = 5 * 60          # running: fresh = timestamp age < 5 min
WAITING_MAX_AGE = 10 * 60         # waiting_for_games: 10 min
SLEEPING_OVERRUN = 10 * 60        # sleeping: if past sleeping_until, fresh = <10 min overshoot
IDLE_END_MAX_AGE = 90 * 60        # idle_end_of_day is a brief transitional state; stale if stuck >90 min


def is_stale(
    path: Path,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Return (is_stale, reason). `now` is optional for tests; defaults to datetime.now(UTC)."""
    if now is None:
        now = datetime.now(timezone.utc)

    if not path.exists():
        return True, f"heartbeat file not found: {path}"

    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return True, f"heartbeat unreadable: {e}"

    ts_str = raw.get("timestamp")
    state = raw.get("state", "unknown")
    try:
        ts = datetime.fromisoformat(ts_str)
    except (TypeError, ValueError):
        return True, f"heartbeat timestamp invalid: {ts_str}"
    age_s = (now - ts).total_seconds()

    if state == "running":
        if age_s > RUNNING_MAX_AGE:
            return True, f"running state but timestamp {age_s:.0f}s old (>{RUNNING_MAX_AGE}s)"
        return False, "fresh running"

    if state == "waiting_for_games":
        if age_s > WAITING_MAX_AGE:
            return True, f"waiting_for_games but timestamp {age_s:.0f}s old"
        return False, "fresh waiting"

    if state == "sleeping":
        wake_str = raw.get("sleeping_until")
        if not wake_str:
            return True, "sleeping state without sleeping_until"
        try:
            wake = datetime.fromisoformat(wake_str)
        except ValueError:
            return True, f"sleeping_until invalid: {wake_str}"
        overshoot = (now - wake).total_seconds()
        if overshoot > SLEEPING_OVERRUN:
            return True, f"sleeping past sleeping_until by {overshoot:.0f}s (>{SLEEPING_OVERRUN}s)"
        return False, "fresh sleeping"

    if state == "idle_end_of_day":
        if age_s > IDLE_END_MAX_AGE:
            return True, f"idle_end_of_day stuck {age_s:.0f}s (>{IDLE_END_MAX_AGE}s)"
        return False, "fresh idle_end_of_day"

    return True, f"unknown state: {state}"


def ping(url: str, suffix: str = "") -> None:
    full = url + suffix
    req = urllib.request.Request(full, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"ping failed: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heartbeat-path", type=Path, required=True)
    ap.add_argument("--ping-url", default=None,
                    help="Healthchecks.io base URL (without /fail suffix)")
    args = ap.parse_args()

    if args.ping_url is None:
        print("  (no --ping-url provided; alerts disabled)", file=sys.stderr)

    stale, reason = is_stale(args.heartbeat_path)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] stale={stale}  reason={reason}")

    if stale:
        if args.ping_url:
            ping(args.ping_url, "/fail")
        sys.exit(1)

    if args.ping_url:
        ping(args.ping_url)  # success ping keeps hc-ping "up"
    sys.exit(0)


if __name__ == "__main__":
    main()
