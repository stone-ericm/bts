"""Small read-only product and Bash probes. All fixture files are temporary."""

from contextlib import redirect_stderr
from datetime import datetime, time, timedelta, timezone
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ET = ZoneInfo("America/New_York")


def json_file(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def heartbeat(root):
    path = root / ".heartbeat"
    spec = importlib.util.spec_from_file_location(
        "check_heartbeat", ROOT / "scripts/check_heartbeat.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = datetime(2027, 7, 15, 16, tzinfo=timezone.utc)
    print(f"stdlib heartbeat interpreter: {sys.version.split()[0]}")
    for value in (clock.isoformat(), None, ["bad"], "2027-07-15T16:00:00"):
        json_file(path, {"timestamp": value, "state": "deploy_failed"})
        try:
            outcome = module.is_stale(path, now=clock)
        except (TypeError, AttributeError) as error:
            outcome = type(error).__name__
        print(f"base deploy_failed timestamp {value!r}: {outcome}")
    for value in (["running"], {"timestamp": clock.isoformat(), "state": "running"}):
        json_file(path, value)
        try:
            outcome = module.is_stale(path, now=clock)
        except (TypeError, AttributeError) as error:
            outcome = type(error).__name__
        print(f"base other input {value!r}: {outcome}")
    path.unlink()
    os.mkfifo(path)
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_heartbeat.py"), "--heartbeat-path", str(path)],
            cwd=root, capture_output=True, timeout=0.4,
        )
    except subprocess.TimeoutExpired:
        print("base monitor FIFO: blocks without a writer")
    else:
        raise AssertionError("expected blocked base heartbeat read")


def pipe(root):
    for signature, body in [
        ("builtin", "printf down"),
        ("external", "/bin/echo down"),
        ("signal", 'kill -PIPE "$$"'),
    ]:
        reader, writer = os.pipe()
        os.close(reader)
        output = root / f"pipe-{signature}"
        with os.fdopen(writer, "wb") as closed:
            result = subprocess.run(
                ["/bin/bash", "-c", 'trap \'echo trapped >>"$1"; exit 1\' PIPE\n'
                 f'{body}\necho continued >>"$1"', "probe", str(output)],
                stdout=closed, stderr=subprocess.PIPE, timeout=2,
            )
        print(f"closed stdout {signature}: rc={result.returncode}, "
              f"events={output.read_text().splitlines() if output.exists() else []}, "
              f"stderr={result.stderr.decode().strip()!r}")


def contest(root):
    from bts import contest_state, entry_intent, scheduler, season_contract

    for mode in ("private", "dm", "public"):
        for left, right in ((False, False), (True, False), (False, True), (True, True)):
            config = {
                "scheduler": {"pick_delivery": mode, "entry_intent": "research" if mode == "private" else "enter",
                              "contest_state_required": left},
                "health_checks": {"contest_state_expected": right},
            }
            problem = entry_intent.live_gate_problem(config)
            reads = scheduler._pick_delivery_mode(config) in {"dm", "public"} and scheduler._contest_state_required(config)
            print(f"gate {mode} {int(left)}{int(right)}: "
                  f"{'refused' if problem else 'read' if reads else 'skip'}")

    picks = root / "picks"
    account = picks / "account_state"
    account.mkdir(parents=True)
    assert not (picks / "streak.json").exists()
    now = datetime(2027, 7, 15, 23, 59, 59, tzinfo=ET)
    directory = {
        "switch": account / "season_contract.json",
        "auto": account / "contest_streak.json",
        "manual": account / "contest_streak.manual.json",
    }
    switch = dict(schema_version=season_contract.SCHEMA, season=2027, status="automatic",
                  activation_id="c" * 32, switched_at=now.isoformat(), revoked_at=None, note=None)
    auto = dict(schema_version=season_contract.AUTO_SCHEMA, season=2027, activation_id="c" * 32,
                active_streak=12, best_streak=14, source="mlb_bts_profile",
                source_date="2027-07-14", recorded_at=now.isoformat())
    day = now.date()
    dated = dict(schema_version=contest_state.DATED_MANUAL_SCHEMA, effective_date=day.isoformat(),
                 season=day.year, expires_at=datetime.combine(day, time.max, ET).isoformat(),
                 active_streak=12, best_streak=14, saver_state="used",
                 source="manual_cli", recorded_at=now.isoformat())
    scenarios = (
        ("dated", None, None, dated, now),
        ("dated next ET day", None, None, dated, now + timedelta(seconds=1)),
        ("automatic", switch, auto, None, now),
        ("corrupt", switch | {"note": []}, auto, None, now),
        ("legacy", switch, auto, {"active_streak": 5, "override_expires_at": "2027-07-16T16:00:00Z"}, now),
        ("absent switch", None, auto, None, now),
        ("revoked switch", switch | {"status": "revoked", "revoked_at": now.isoformat()}, auto, None, now),
        ("absent observation", switch, None, None, now),
        ("schema", switch, auto | {"schema_version": "bts_contest_streak_auto_v1"}, None, now),
        ("observation season", switch, auto | {"season": 2026}, None, now),
        ("activation", switch, auto | {"activation_id": "b" * 32}, None, now),
        ("Dec 31", switch, auto, None, datetime(2027, 12, 31, 23, 59, 59, tzinfo=ET)),
        ("Jan 1", switch, auto, None, datetime(2028, 1, 1, 0, 0, 0, tzinfo=ET)),
    )
    for case, sw, automatic, manual, clock in scenarios:
        for key, payload in zip(directory, (sw, automatic, manual)):
            if payload is None:
                directory[key].unlink(missing_ok=True)
            else:
                json_file(directory[key], payload)
        before = {p.name: p.read_bytes() for p in account.iterdir()}
        try:
            observed = contest_state.load_decision_streak_state(picks, require_contest_state=True, now=clock)
            direct = observed.observation_kind
        except contest_state.ContestStateError as error:
            direct = str(error).splitlines()[0]
        journal = StringIO()
        with patch("bts.health.alert.dispatch_dm_for_health_alerts") as send, redirect_stderr(journal):
            scheduler._morning_state_check({}, picks, clock.date().isoformat(), clock,
                                           clock + timedelta(minutes=45))
        assert ("no pick can be made yet" in journal.getvalue()) == direct.startswith("Live decision refused:")
        assert not send.called and before == {p.name: p.read_bytes() for p in account.iterdir()}
        print(f"{case}: {direct}; morning: {journal.getvalue().strip()[:165]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        workspace = Path(temp)
        heartbeat(workspace)
        pipe(workspace)
        if "--stdlib-only" not in sys.argv:
            contest(workspace)
