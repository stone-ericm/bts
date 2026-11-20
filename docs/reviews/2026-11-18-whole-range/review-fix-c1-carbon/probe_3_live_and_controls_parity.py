"""Item 3 probe: byte-identical verification of same-day/no-today/future runs and all live/CLI/morning paths (BASE vs HEAD)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
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
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_SRC = HERE / "src"
HEAD_SRC = REPO / "src"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
D = date(2027, 3, 24)
DEC31 = date(2027, 12, 31)
MATCH_ID = "a" * 32


def deny_network(*_args, **_kwargs):
    raise AssertionError("Network access is forbidden")


def at_et(day: date, hr: int = 0, mn: int = 0) -> datetime:
    return datetime.combine(day, time(hr, mn), ET)


def seed_fixtures(base_dir: Path) -> list[str]:
    names = []

    def mk(name: str) -> Path:
        names.append(name)
        p = base_dir / name / "oom" / "picks" if name == "oom_legacy" else base_dir / name / "picks"
        (p / "account_state").mkdir(parents=True, exist_ok=True)
        (p / "streak.json").write_text(json.dumps({"season": D.year, "streak": 4, "saver_available": True}))
        return p

    # 1. Empty
    mk("empty")

    # 2. Dated valid + dated saver contradiction
    p = mk("dated_contradict")
    (p / "account_state/contest_streak.manual.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_manual_v3", "season": D.year,
        "effective_date": D.isoformat(), "active_streak": 12, "best_streak": 12,
        "saver_state": "not_earned", "recorded_at": at_et(D, 9).isoformat(),
        "expires_at": datetime.combine(D, time.max, ET).isoformat(), "source": "cli",
    }))
    (p / "account_state/contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": 3, "best_streak": 3,
        "source_date": D.isoformat(), "recorded_at": at_et(D, 8).isoformat(),
    }))

    # 3. Revoked switch + flag saver contradiction (fix set B dated_template) + lag + legacy expired
    p = mk("revoked_flag_contradict")
    (p / "account_state/contest_streak.manual.json").write_text(json.dumps({"streak": 5}))
    (p / "account_state/contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2", "season": D.year, "activation_id": MATCH_ID,
        "active_streak": 12, "best_streak": 14, "source_date": (D - timedelta(days=3)).isoformat(),
        "recorded_at": at_et(D, 7).isoformat(), "source": "mlb_bts_profile",
    }))
    (p / "account_state/season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1", "season": D.year, "activation_id": MATCH_ID,
        "status": "revoked", "switched_at": at_et(D, 6).isoformat(), "revoked_at": at_et(D, 7).isoformat(),
    }))
    (p / "account_state/saver_state.json").write_text(json.dumps({"season": D.year, "state": "not_earned"}))
    (p / f"{(D - timedelta(days=1)).isoformat()}.json").write_text('{"result":"hit"}')
    (p / f"{D.isoformat()}.json").write_text('{"result":"hit"}')

    # 4. Accepted automatic (fresh vs stale 20h)
    p = mk("accepted_auto_stale")
    (p / "account_state/contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2", "season": D.year, "activation_id": MATCH_ID,
        "active_streak": 8, "best_streak": 10, "source_date": (D - timedelta(days=1)).isoformat(),
        "recorded_at": at_et(D - timedelta(days=1), 10).isoformat(), "source": "mlb_bts_profile",
    }))
    (p / "account_state/season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1", "season": D.year, "activation_id": MATCH_ID,
        "status": "automatic", "switched_at": at_et(D - timedelta(days=1), 8).isoformat(), "revoked_at": None,
    }))
    (p / "account_state/saver_state.json").write_text(json.dumps({"season": D.year, "state": "not_earned"}))
    (p / "account_state/contest_ledger.jsonl").write_text(
        json.dumps({"recorded_at": "2027-03-23T14:00:00Z", "active_streak": 8, "best_streak": 10, "season": 2027}) + "\n"
        + "{bad_line\n"
    )

    # 5. Corrupt switch + selected legacy manual + OOM path control
    p = mk("oom_legacy")
    (p / "account_state/contest_streak.manual.json").write_text(json.dumps({
        "streak": 2, "source_date": D.isoformat(),
        "override_expires_at": at_et(D, 12).isoformat(),
    }))
    (p / "account_state/season_contract.json").write_text("{corrupt")

    # 6. Corrupt auto
    p = mk("corrupt_auto")
    (p / "account_state/contest_streak.json").write_text(json.dumps({"active_streak": 99}))

    return names


def collect_snapshot(fixtures_dir: Path, names: list[str]) -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod
    from bts import contest_state as cs, saver_state as ss, scheduler, season_contract as sc
    from bts.health import alert, contest_state as hcs, runner

    wall = at_et(D, 23, 50)
    results: dict = {}

    def call_capture(fn):
        try:
            val = fn()
            return {"ok": repr(val)}
        except Exception as exc:
            return {"err": [type(exc).__name__, str(exc)]}

    for name in names:
        picks_src = (
            fixtures_dir / name / "oom" / "picks"
            if name == "oom_legacy" else fixtures_dir / name / "picks"
        )
        entry: dict = {}

        # A. Live readers & helpers
        entry["load_contest_reqF"] = call_capture(lambda: cs.load_contest_streak_state(picks_src, now=wall, require_dated_auto=False))
        entry["load_contest_reqT"] = call_capture(lambda: cs.load_contest_streak_state(picks_src, now=wall, require_dated_auto=True))
        entry["load_contest_omit"] = call_capture(lambda: cs.load_contest_streak_state(picks_src, now=wall, require_dated_auto=True, omit_selected_dated=True))
        entry["load_decision_reqF"] = call_capture(lambda: cs.load_decision_streak_state(picks_src, require_contest_state=False, now=wall))
        entry["load_decision_reqT"] = call_capture(lambda: cs.load_decision_streak_state(picks_src, require_contest_state=True, now=wall))
        entry["load_decision_omit"] = call_capture(lambda: cs.load_decision_streak_state(picks_src, require_contest_state=True, now=wall, omit_selected_dated=True))
        entry["load_saver_2027"] = call_capture(lambda: ss.load_saver_state(picks_src, season=2027))
        entry["load_saver_2028"] = call_capture(lambda: ss.load_saver_state(picks_src, season=2028))
        entry["switch_state"] = call_capture(lambda: sc.switch_state(picks_src, wall))
        sel = cs.load_contest_streak_state(picks_src, now=wall, require_dated_auto=False) if name != "corrupt_auto" else None
        entry["live_auto_refusal"] = call_capture(lambda: sc.live_automatic_refusal(picks_src, wall, sel))
        entry["dated_template"] = sc.dated_template(picks_src, wall)
        entry["direct_health_check_T"] = call_capture(lambda: hcs.check(picks_src, expected=True, now=wall.astimezone(UTC)))
        entry["direct_health_check_F"] = call_capture(lambda: hcs.check(picks_src, expected=False, now=wall.astimezone(UTC)))

        # B. CLI `bts season-contract status`
        with patch.object(cli_mod, "_now_et", lambda: wall):
            cli_res = CliRunner().invoke(cli_mod.cli, ["season-contract", "status", "--picks-dir", str(picks_src)])
        entry["cli_status"] = {"code": cli_res.exit_code, "out": cli_res.output}

        # C. Morning state check
        morning_dms: list[str] = []
        with TemporaryDirectory(dir=fixtures_dir) as mtmp:
            mpicks = Path(mtmp) / "picks"
            import shutil
            shutil.copytree(picks_src, mpicks)
            with patch.object(alert, "send_dm", side_effect=lambda _r, b: (morning_dms.append(b), "m-id")[1]):
                scheduler._morning_state_check(
                    {"bluesky": {"dm_recipient": "op.invalid"}}, mpicks, D.isoformat(),
                    at_et(D, 10, 0), at_et(D, 12, 0),
                )
        entry["morning_dms"] = morning_dms

        # D. `run_all_checks` for omitted `today`, same-day `today=D`, and future `today=D+1`
        for mode_label, kwargs in [
            ("omitted", {}),
            ("same_day", {"today": D}),
            ("future_day", {"today": D + timedelta(days=1)}),
        ]:
            class FixedClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return wall.astimezone(tz) if tz is not None else wall.replace(tzinfo=None)

            class FixedDate(date):
                @classmethod
                def today(cls):
                    return D

            dms: list[str] = []
            hdir = fixtures_dir / f"run_{name}_{mode_label}"
            hdir.mkdir(parents=True, exist_ok=True)
            orig_safe = runner._safe_run
            with (
                patch.object(runner, "datetime", FixedClock, create=True),
                patch.object(runner, "date", FixedDate),
                patch.object(hcs, "datetime", FixedClock),
                patch.object(alert, "datetime", FixedClock),
                patch.object(alert, "send_dm", side_effect=lambda _r, b: (dms.append(b), "r-id")[1]),
                patch.object(
                    runner, "_safe_run",
                    lambda n, fn: orig_safe(n, fn) if n in {"contest_state", "streak_validation"} else [],
                ),
            ):
                alerts = runner.run_all_checks(
                    picks_src, fixtures_dir / "models", "op.invalid",
                    contest_state_expected=True,
                    thresholds_overrides={
                        "warn_attention_state": hdir / "warn.json",
                        "health_dm_delivery_status": hdir / "dm.json",
                    },
                    **kwargs,
                )
            entry[f"runner_{mode_label}"] = {
                "alerts": [(a.level, a.source, a.incident_key, a.message) for a in alerts],
                "dms": dms,
                "warn_file": (hdir / "warn.json").read_text() if (hdir / "warn.json").exists() else None,
                "dm_file": (hdir / "dm.json").read_text() if (hdir / "dm.json").exists() else None,
            }

        results[name] = entry
    return results


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        logging.disable(logging.CRITICAL)
        fdir = Path(sys.argv[2])
        names = json.loads(sys.argv[3])
        with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
            snap = collect_snapshot(fdir, names)
        print(json.dumps(snap, sort_keys=True))
        return

    signal.alarm(50)
    with TemporaryDirectory(dir=HERE) as tmp:
        tmp_path = Path(tmp)
        base_dir = tmp_path / "base_fix"
        head_dir = tmp_path / "head_fix"
        names = seed_fixtures(base_dir)
        seed_fixtures(head_dir)

        def run_side(src_path: Path, fdir: Path) -> str:
            env = {**os.environ, "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(src_path)}
            raw = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "worker", str(fdir), json.dumps(names)],
                env=env, capture_output=True, text=True, timeout=30, check=True,
            ).stdout
            # Normalize temp root prefix so path strings inside same-day messages compare cleanly
            return raw.replace(str(fdir), "<FIXTURES>")

        base_json = run_side(BASE_SRC, base_dir)
        head_json = run_side(HEAD_SRC, head_dir)
        base_sha = hashlib.sha256(base_json.encode("utf-8")).hexdigest()
        head_sha = hashlib.sha256(head_json.encode("utf-8")).hexdigest()
        if base_json != head_json:
            b_obj = json.loads(base_json)
            h_obj = json.loads(head_json)
            for k in b_obj:
                for subk in b_obj[k]:
                    if b_obj[k][subk] != h_obj[k][subk]:
                        raise AssertionError(f"Diff at {k}.{subk}:\nBASE={b_obj[k][subk]}\nHEAD={h_obj[k][subk]}")
        assert base_sha == head_sha
        # Also confirm signature parity of public functions
        import inspect
        import bts.contest_state as cs
        import bts.saver_state as ss
        import bts.season_contract as sc
        assert str(inspect.signature(cs.load_decision_streak_state)) == "(picks_dir: 'Path', *, require_contest_state: 'bool' = False, now: 'datetime | None' = None, switch=None, omit_selected_dated: 'bool' = False) -> 'DecisionStreakState'"
        assert str(inspect.signature(cs.load_contest_streak_state)) == "(picks_dir: 'Path', *, now: 'datetime | None' = None, require_dated_auto: 'bool' = False, omit_selected_dated: 'bool' = False) -> 'ContestStreakState | None'"
        assert str(inspect.signature(ss.load_saver_state)) == "(picks_dir: 'Path', *, season: 'int') -> 'SaverState'"
        print(f"[PASS] Item 3: BASE vs HEAD byte-identical across {len(names)} fixtures x live readers, CLI status, morning check, and omitted/same/future runner modes (sha256={head_sha[:12]})")
    signal.alarm(0)


if __name__ == "__main__":
    main()
