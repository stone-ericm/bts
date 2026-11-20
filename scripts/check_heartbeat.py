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
from contextlib import contextmanager
import fcntl
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
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
HOLD_PATH = Path("/var/lib/bts-deploy/hold")
GUARD_TOOL = "/usr/local/bin/bts-deploy-hold"
PROC_LOCKS = Path("/proc/locks")
NOFOLLOW = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK


def _unique_fields(pairs):
    pairs = tuple(pairs)
    record = dict(pairs)
    if len(record) != len(pairs):
        raise ValueError("duplicate record key")
    return record


def _bounded_record(path):
    try:
        fd = os.open(path, NOFOLLOW)
    except FileNotFoundError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("record type")
        payload = os.read(fd, 8193)
    finally:
        os.close(fd)
    if len(payload) > 8192:
        raise ValueError("record size")
    record = json.loads(payload, object_pairs_hook=_unique_fields)
    if not isinstance(record, dict):
        raise ValueError("record shape")
    return record


@contextmanager
def _hold_snapshot(hold):
    fd = None
    try:
        try:
            parent = os.open(hold.parent, NOFOLLOW | os.O_DIRECTORY)
            try:
                expected_uid = os.geteuid() if hold.parent != HOLD_PATH.parent else 0
                info = os.fstat(parent)
                if info.st_uid != expected_uid or stat.S_IMODE(info.st_mode) != 0o755:
                    raise ValueError("guard directory")
                fd = os.open("state.lock", NOFOLLOW, dir_fd=parent)
            finally:
                os.close(parent)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != expected_uid
                    or stat.S_IMODE(info.st_mode) != 0o644 or info.st_nlink != 1):
                raise ValueError("state lock type")
            deadline = time.monotonic() + 2
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        except (OSError, ValueError):
            yield False
        else:
            yield True
    finally:
        if fd is not None:
            os.close(fd)


def _owner_alive(hold, unit):
    try:
        info = (hold.parent / "transaction.lock").lstat()
        expected_uid = 0 if hold.parent == HOLD_PATH.parent else os.geteuid()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != expected_uid
                or stat.S_IMODE(info.st_mode) != 0o644 or info.st_nlink != 1):
            return False
        result = subprocess.run(
            ["/usr/bin/systemctl", "show", unit, "-p", "ActiveState", "-p", "MainPID"],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=3,
        )
        if result.returncode != 0:
            return False
        fields = _unique_fields(line.split("=", 1) for line in result.stdout.decode().splitlines())
        pid = fields.get("MainPID", "")
        if fields.get("ActiveState") != "active" or not re.fullmatch(r"[1-9][0-9]*", pid):
            return False
        with PROC_LOCKS.open() as stream:
            locks = stream.read(1_048_577)
        if len(locks) > 1_048_576:
            return False
        for line in locks.splitlines():
            parts = line.split()
            if len(parts) != 8 or parts[1:5] != ["FLOCK", "ADVISORY", "WRITE", pid]:
                continue
            major, minor, inode = parts[5].split(":")
            if (int(major, 16), int(minor, 16), int(inode)) == (
                os.major(info.st_dev), os.minor(info.st_dev), info.st_ino,
            ):
                return True
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False
    return False


def _monitor_hold(hold, now):
    if hold == HOLD_PATH:
        result = subprocess.run(
            ["/usr/bin/python3", "-I", GUARD_TOOL, "status", "--json"],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=3,
        )
        if result.returncode != 1:
            raise ValueError("hold status")
        record = json.loads(result.stdout, object_pairs_hook=_unique_fields)
    else:
        record = _bounded_record(hold)
    if not isinstance(record, dict):
        raise ValueError("hold record")
    state, step, target = (record[key] for key in ("state", "step", "target"))
    run_id, armed, updated = (record[key] for key in ("deploy_id", "armed_at", "updated_at"))
    if (
        record.get("schema") != "bts_deploy_hold_v1"
        or state not in ("pending", "activating", "failed")
        or not re.fullmatch(r"[a-z-]{1,32}", step)
        or not re.fullmatch(r"[0-9a-f]{40}", target)
        or not re.fullmatch(r"[0-9]{1,20}-[A-Za-z0-9]{1,64}", run_id)
        or record.get("owner_unit") != f"bts-deploy-{run_id}.service"
        or type(armed) is not int or type(updated) is not int
        or not 0 <= armed <= updated <= int(now.timestamp())
    ):
        raise ValueError("hold values")
    age = int(now.timestamp()) - armed
    if state in ("pending", "activating") and age <= 1380 and _owner_alive(hold, record["owner_unit"]):
        return "suppress", f"deploy in progress (run {run_id}, armed {age}s ago); no ping"
    return "fail", f"Deploy hold: {state} at {step} (requested target {target[:7]}). See INCIDENT.md §4."


def _hold_decision(hold, heartbeat, now):
    try:
        hold.lstat()
    except FileNotFoundError:
        pass
    except OSError:
        return "fail", "Deploy hold present and unreadable. See INCIDENT.md §4."
    else:
        try:
            return _monitor_hold(hold, now)
        except (OSError, KeyError, TypeError, ValueError, subprocess.TimeoutExpired):
            return "fail", "Deploy hold present and unreadable. See INCIDENT.md §4."
    try:
        release = _bounded_record(hold.parent / "last-release")
        if release is None:
            return None, None
        epoch, kind = release["released_at"], release["kind"]
        if type(epoch) not in (int, float) or not 0 <= epoch <= now.timestamp() or kind not in ("hand", "deploy"):
            raise ValueError("release values")
        released_at = datetime.fromtimestamp(epoch, timezone.utc)
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        return "fail", "Deploy hold release record unreadable. See INCIDENT.md §4."
    try:
        record = _bounded_record(heartbeat)
        timestamp = datetime.fromisoformat(record["timestamp"])
        if timestamp.utcoffset() is not None and timestamp > released_at:
            return "legacy", _assess_heartbeat(record, datetime.now(timezone.utc))
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return "fail", (
        f"Deploy hold released at {released_at.isoformat()} ({kind}); "
        "no scheduler heartbeat since. See INCIDENT.md §4."
    )


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

    return _assess_heartbeat(raw, now)


def _assess_heartbeat(raw, now):
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
# Multiple horizons (Codex review #3): a fast 30s crash-loop trips the 20-min
# window; a slow ~10-min failure cycle only adds +2 per 20 min and needs the
# 60-min horizon; anything churning a few times over hours trips the 3-hour
# one. The daily lifecycle restart (+1/day) and deploys (manual restarts do
# not increment NRestarts) stay far below every threshold.
CHURN_WINDOWS = ((20 * 60, 3), (60 * 60, 3), (180 * 60, 4))


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
    windows: tuple = CHURN_WINDOWS,
) -> tuple[bool, str, list[dict]]:
    """Return (churn, reason, updated_samples).

    samples: [{"ts": iso, "n": int}] prior NRestarts readings. Churn fires when
    the counter climbed >= a window's threshold above that window's minimum.
    Samples older than the LONGEST window — or larger than current (counter
    reset on daemon-reload) — are pruned so a reset rebaselines instead of
    firing. Timezone-naive sample timestamps are treated as UTC; malformed
    samples are dropped (Codex review #10 — auxiliary state must never crash
    the liveness monitor).
    """
    if current_n is None:
        return False, "nrestarts unavailable (churn check skipped)", samples

    max_window_s = max(w for w, _ in windows)
    kept: list[tuple[datetime, int]] = []
    for s in samples:
        try:
            ts = datetime.fromisoformat(str(s["ts"]))
            n = int(s["n"])
        except (KeyError, TypeError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= now - timedelta(seconds=max_window_s) and n <= current_n:
            kept.append((ts, n))

    updated = ([{"ts": ts.isoformat(), "n": n} for ts, n in kept]
               + [{"ts": now.isoformat(), "n": current_n}])
    for window_s, threshold in windows:
        cutoff = now - timedelta(seconds=window_s)
        in_window = [n for ts, n in kept if ts >= cutoff]
        delta = current_n - min(in_window, default=current_n)
        if delta >= threshold:
            return True, (
                f"restart churn: NRestarts +{delta} within {window_s // 60} min "
                f"(now {current_n}) — daemon is crash-looping behind a fresh heartbeat"
            ), updated

    short_cutoff = now - timedelta(seconds=windows[0][0])
    short_delta = current_n - min(
        (n for ts, n in kept if ts >= short_cutoff), default=current_n)
    return False, f"nrestarts {current_n} (+{short_delta} in window)", updated


def load_churn_samples(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    samples = data.get("samples")
    return samples if isinstance(samples, list) else []


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


def ping_deploy_failure(url, reason):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    def deadline(signum, frame):
        raise TimeoutError()

    verified = False
    if re.fullmatch(r"https://hc-ping\.com/[A-Za-z0-9_-]+", url):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        request = urllib.request.Request(url + "/fail", data=reason.encode(), method="POST")
        previous = signal.signal(signal.SIGALRM, deadline)
        signal.alarm(10)
        try:
            with opener.open(request, timeout=10) as response:
                verified = 200 <= response.status <= 299
        except (OSError, ValueError):
            pass
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    print("Failure ping received HTTP 2xx: " + ("verified." if verified else "unverified."))


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
    test_root = os.environ.get("BTS_TEST_DEPLOY_GUARD_DIR")
    ap.add_argument("--deploy-hold", type=Path, default=Path(test_root) / "hold" if test_root else HOLD_PATH)
    args = ap.parse_args()

    with _hold_snapshot(args.deploy_hold) as verified:
        if not verified:
            try:
                args.deploy_hold.parent.lstat()
            except FileNotFoundError:
                return _legacy_check(args)
            except OSError:
                pass
            decision, reason = "fail", "Deploy hold lookup failed. See INCIDENT.md §4."
        else:
            decision, reason = _hold_decision(args.deploy_hold, args.heartbeat_path, datetime.now(timezone.utc))
        if decision == "suppress":
            print(reason)
            raise SystemExit(0)
        if decision == "fail":
            print(reason)
            if args.ping_url:
                ping_deploy_failure(args.ping_url, reason)
            raise SystemExit(1)
        return _legacy_check(args, reason if decision == "legacy" else None)


def _legacy_check(args, checked=None):
    if args.ping_url is None:
        print("  (no --ping-url provided; alerts disabled)", file=sys.stderr)

    stale, reason = checked if checked is not None else is_stale(args.heartbeat_path)

    # Churn overrides a fresh heartbeat: a crash-loop refreshes the heartbeat
    # every 30s cycle, so freshness alone cannot clear the daemon (audit F3).
    if not stale and not args.no_churn:
        try:
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
        except Exception as e:  # noqa: BLE001 — churn is auxiliary; the
            # liveness ping must go out even if churn state is malformed
            print(f"churn check errored (continuing with liveness only): {e}",
                  file=sys.stderr)
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
