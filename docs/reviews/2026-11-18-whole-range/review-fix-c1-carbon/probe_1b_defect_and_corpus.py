"""Item 1 probe (v2): reproduce W20 on base with check's drivers, verify gone on HEAD, and compare corpus vs base@end(D)."""

from __future__ import annotations

from contextlib import redirect_stdout
from datetime import date, datetime, time, timedelta, timezone
import io
import json
import logging
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_SRC = HERE / "src"
HEAD_SRC = REPO / "src"
CC_DIR = HERE / "contract-check-c1"
SLICES_DIR = REPO / "docs/reviews/2026-11-18-whole-range"

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
MATCH_ID = "a" * 32
OTHER_ID = "b" * 32


def deny_network(*_args, **_kwargs):
    raise AssertionError("Network access is forbidden")


def end_utc(day: date) -> datetime:
    return datetime.combine(day, time.max, ET).astimezone(UTC)


def at_et(day: date, hour: int = 0, minute: int = 0, second: int = 0, micro: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute, second, micro), ET)


def run_cc_driver(driver_kind: str, scratch: Path) -> None:
    """Run contract-check-c1 driver against whichever bts is first on sys.path (BASE_SRC)."""
    import bts
    assert Path(bts.__file__).resolve().is_relative_to(BASE_SRC.resolve()), bts.__file__

    sys.path.insert(0, str(CC_DIR))
    import probe as cc_probe
    # Fix probe.py's relative ROOT/SLICES paths and ensure BASE_SRC stays first on sys.path:
    sys.path.insert(0, str(BASE_SRC))
    cc_probe.ROOT = REPO
    cc_probe.SLICES = SLICES_DIR

    if driver_kind == "operator_scheduler":
        import operator_cases
        operator_cases.full_scheduler(scratch)
    elif driver_kind == "probe_messages":
        os.chdir(scratch)
        with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
            cc_probe.messages(scratch)
    elif driver_kind == "probe_edges":
        # Run edges except the 5x3s base FIFO hang loop (already covered in Item 5) to stay super fast,
        # or run full edges if fast enough (5 * 3s = 15s, well within 30s timeout!).
        os.chdir(scratch)
        with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
            cc_probe.edges(scratch)
    else:
        raise ValueError(driver_kind)


def verify_head_scheduler_w20(scratch: Path) -> dict:
    sys.path.insert(0, str(SLICES_DIR / "slice-1-carbon"))
    from test_slice1_probes_v3 import RehearsalWorld, _H, _setup_revoked_box, et_dt
    from bts.health import runner

    _H.REPO_ROOT = REPO
    schedules = _H.default_opening_days_schedules()
    day = "2027-03-24"
    game = schedules[day][0]
    game.start_et = et_dt(2027, 3, 24, 21, 5)
    game.away_confirm_et = et_dt(2027, 3, 24, 18, 45)
    game.home_confirm_et = et_dt(2027, 3, 24, 19, 45)
    game.final_et = et_dt(2027, 3, 25, 0, 15)
    game.away_hits = 0

    with RehearsalWorld(scratch / "head_sched", initial_et=et_dt(2027, 3, 24, 9), schedules=schedules) as world:
        _setup_revoked_box(world)
        pargs = ["--picks-dir", str(world.picks_dir)]
        assert world.invoke_cli(["saver-state", "--init", "not_earned", "--season", "2027", *pargs]).exit_code == 0
        (world.park_dir / "park_drag_manifest.json").write_text(json.dumps({
            "max_source_game_date": day, "generated_at": "2027-03-25T00:10:00", "export_rows": 2,
        }))
        world.clock.set_et(et_dt(2027, 3, 24, 9, 30))
        assert world.invoke_cli([
            "set-contest-streak", "--date", day, "--streak", "0",
            "--best-streak", "0", "--saver", "not_earned", *pargs,
        ]).exit_code == 0
        world.clock.set_et(et_dt(2027, 3, 24, 10))

        class RunnerClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return world.clock.now(tz)

        with patch.object(runner, "datetime", RunnerClock):
            world.run_scheduler_day(day, stop_at_et=et_dt(2027, 3, 25, 1))

        health_dms = [dm for dm in world.dms if dm["text"].startswith("BTS health ")]
        assert len(health_dms) == 1
        body = health_dms[0]["text"]
        assert "[contest_state]" not in body
        assert "[saver_state]" not in body
        assert "--date 2027-03-25" not in body
        return {"at_et": health_dms[0]["at_et"], "body": body}


def main() -> None:
    from probe_1_defect_and_corpus import build_corpus_specs, materialize_spec, worker_eval

    if len(sys.argv) > 1 and sys.argv[1] == "cc_driver":
        logging.disable(logging.CRITICAL)
        with TemporaryDirectory(dir=HERE) as tmp:
            run_cc_driver(sys.argv[2], Path(tmp))
        return

    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        logging.disable(logging.CRITICAL)
        with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
            worker_eval(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
        return

    signal.alarm(58)
    logging.disable(logging.CRITICAL)
    base_env = {**os.environ, "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(BASE_SRC)}
    head_env = {**os.environ, "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(HEAD_SRC)}

    for kind in ("operator_scheduler", "probe_messages", "probe_edges"):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "cc_driver", kind],
            env=base_env, capture_output=True, text=True, timeout=35, check=True,
        )
        print(f"[PASS] base_{kind}: exit=0, lines={len(proc.stdout.splitlines())}")

    with TemporaryDirectory(dir=HERE) as tmp:
        tmp_path = Path(tmp)
        head_sched = verify_head_scheduler_w20(tmp_path)
        print(f"[PASS] head_scheduler_eod_0015: at={head_sched['at_et']}, contest/saver alerts=0")

        all_specs = []
        for d in (date(2027, 6, 30), date(2027, 1, 15), date(2027, 12, 31)):
            all_specs.extend(build_corpus_specs(d))
        for spec in all_specs:
            materialize_spec(tmp_path / "corpus", spec)
        specs_file = tmp_path / "specs.json"
        specs_file.write_text(json.dumps(all_specs), encoding="utf-8")

        base_res = json.loads(subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "worker", "base_endpoint", str(specs_file), str(tmp_path / "corpus")],
            env=base_env, capture_output=True, text=True, timeout=25, check=True,
        ).stdout)
        head_res = json.loads(subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "worker", "head_historical", str(specs_file), str(tmp_path / "corpus")],
            env=head_env, capture_output=True, text=True, timeout=25, check=True,
        ).stdout)

        mismatches = [
            (k, base_res[k], head_res[k])
            for k in base_res
            if base_res[k] != head_res[k]
        ]
        assert not mismatches, f"Corpus mismatches found: {mismatches[:5]}"
        non_empty = sum(1 for v in head_res.values() if v)
        crits = sum(sum(1 for lvl, _, _ in v if lvl == "CRITICAL") for v in head_res.values())
        warns = sum(sum(1 for lvl, _, _ in v if lvl == "WARN") for v in head_res.values())
        infos = sum(sum(1 for lvl, _, _ in v if lvl == "INFO") for v in head_res.values())
        print(
            f"[PASS] corpus parity: {len(all_specs)} states across summer/winter/Dec31 x 3 wall times "
            f"(non-empty={non_empty}, CRITICAL={crits}, WARN={warns}, INFO={infos}, mismatches=0)"
        )
    signal.alarm(0)


if __name__ == "__main__":
    main()
