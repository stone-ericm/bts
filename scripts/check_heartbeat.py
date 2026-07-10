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
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
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

    if state == "stalled":
        stage = raw.get("stage", "?")
        stalled_for = raw.get("stalled_for_s", "?")
        return True, (
            f"cascade stalled in stage '{stage}' for {stalled_for}s "
            f"(process alive, progress stopped)"
        )

    return True, f"unknown state: {state}"


# --- restart-churn detection (audit F3) --------------------------------------
# A deterministic startup crash restarts every 30s (Restart=always) and can
# refresh the heartbeat each cycle, so freshness alone scores a crash-loop as
# healthy. NRestarts climbing inside a short window is the external tell.
CHURN_WINDOW_S = 20 * 60
CHURN_THRESHOLD = 3


def read_nrestarts(unit: str, run=subprocess.run) -> int | None:
    """NRestarts for a --user unit, or None if unreadable.

    Cron lacks the user-session env: default XDG_RUNTIME_DIR / DBUS address so
    `systemctl --user` can reach the user manager (lingering is enabled on the
    box). Any failure returns None — churn is a best-effort ADDITION; it must
    never break the primary liveness signal.
    """
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    env.setdefault("DBUS_SESSION_BUS_ADDRESS",
                   f"unix:path={env['XDG_RUNTIME_DIR']}/bus")
    try:
        r = run(
            ["systemctl", "--user", "show", unit, "-p", "NRestarts", "--value"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        if r.returncode != 0:
            return None
        return int(r.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def assess_churn(
    samples: list[dict],
    current_n: int | None,
    now: datetime,
    window_s: int = CHURN_WINDOW_S,
    threshold: int = CHURN_THRESHOLD,
) -> tuple[bool, str, list[dict]]:
    """Return (churn, reason, updated_samples).

    samples: [{"ts": iso, "n": int}] prior NRestarts readings. Churn fires when
    the counter climbed >= threshold above the window's minimum. Samples older
    than the window — or larger than current (counter reset on daemon-reload) —
    are pruned so a reset rebaselines instead of firing.
    """
    if current_n is None:
        return False, "nrestarts unavailable (churn check skipped)", samples

    cutoff = now - timedelta(seconds=window_s)
    kept = []
    for s in samples:
        try:
            ts = datetime.fromisoformat(s["ts"])
            n = int(s["n"])
        except (KeyError, TypeError, ValueError):
            continue
        if ts >= cutoff and n <= current_n:
            kept.append({"ts": s["ts"], "n": n})

    updated = kept + [{"ts": now.isoformat(), "n": current_n}]
    baseline = min((s["n"] for s in kept), default=current_n)
    delta = current_n - baseline
    if delta >= threshold:
        return True, (
            f"restart churn: NRestarts +{delta} within {window_s // 60} min "
            f"(now {current_n}) — daemon is crash-looping behind a fresh heartbeat"
        ), updated
    return False, f"nrestarts {current_n} (+{delta} in window)", updated


def load_churn_samples(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
        return data.get("samples", []) if isinstance(data, dict) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_churn_samples(path: Path, unit: str, samples: list[dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"unit": unit, "samples": samples}))
        os.replace(tmp, path)
    except OSError as e:
        print(f"churn state write failed: {e}", file=sys.stderr)


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
    ap.add_argument("--churn-unit", default="bts-scheduler",
                    help="systemd --user unit to watch for restart churn")
    ap.add_argument("--churn-state", type=Path, default=None,
                    help="sample-history JSON (default: <heartbeat dir>/health_state/scheduler_churn.json)")
    ap.add_argument("--no-churn", action="store_true",
                    help="disable the NRestarts churn check")
    args = ap.parse_args()

    if args.ping_url is None:
        print("  (no --ping-url provided; alerts disabled)", file=sys.stderr)

    stale, reason = is_stale(args.heartbeat_path)

    # Churn overrides a fresh heartbeat: a crash-loop refreshes the heartbeat
    # every 30s cycle, so freshness alone cannot clear the daemon (audit F3).
    if not stale and not args.no_churn:
        state_path = args.churn_state or (
            args.heartbeat_path.parent / "health_state" / "scheduler_churn.json"
        )
        samples = load_churn_samples(state_path)
        current = read_nrestarts(args.churn_unit)
        churn, churn_reason, samples = assess_churn(
            samples, current, datetime.now(timezone.utc)
        )
        save_churn_samples(state_path, args.churn_unit, samples)
        if churn:
            stale, reason = True, churn_reason
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
