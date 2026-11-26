"""Item 1 probe (v3): inspect HEAD scheduler EOD DMs + run 450-state corpus parity check (BASE@end(D) vs HEAD historical)."""

from __future__ import annotations

from datetime import date
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from probe_1_defect_and_corpus import (
    BASE_SRC,
    HEAD_SRC,
    HERE,
    REPO,
    SLICES_DIR,
    build_corpus_specs,
    deny_network,
    materialize_spec,
    worker_eval,
)


def inspect_head_scheduler_w20(scratch: Path) -> dict:
    from datetime import datetime
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
                return world.clock.now_et().astimezone(tz) if tz is not None else world.clock.now(None)

        with patch.object(runner, "datetime", RunnerClock):
            world.run_scheduler_day(day, stop_at_et=et_dt(2027, 3, 25, 1))

        health_dms = [dm for dm in world.dms if dm["text"].startswith("BTS health ")]
        for dm in health_dms:
            assert "[contest_state]" not in dm["text"], dm["text"]
            assert "[saver_state]" not in dm["text"], dm["text"]
            assert "--date 2027-03-25" not in dm["text"], dm["text"]
        status_path = world.picks_dir.parent / "health_state" / "health_dm_delivery_status.json"
        status_data = json.loads(status_path.read_text()) if status_path.exists() else None
        return {"health_dm_count": len(health_dms), "health_dms": health_dms, "status": status_data}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        logging.disable(logging.CRITICAL)
        with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
            worker_eval(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
        return

    signal.alarm(50)
    base_env = {**os.environ, "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(BASE_SRC)}
    head_env = {**os.environ, "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(HEAD_SRC)}

    with TemporaryDirectory(dir=HERE) as tmp:
        tmp_path = Path(tmp)
        sched_info = inspect_head_scheduler_w20(tmp_path)
        print(f"[PASS] head_scheduler_eod_0015: health_dm_count={sched_info['health_dm_count']}, sent_day={sched_info['status'] and sched_info['status'].get('sent_day')}")
        for dm in sched_info["health_dms"]:
            print("  DM:", dm["text"])

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
