"""Item 1 probe: reproduce W20 on base vs HEAD, then compare historical HEAD vs base@end(D) over a corpus."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, time, timedelta, timezone
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


def build_corpus_specs(day: date) -> list[dict]:
    """Generate a broad corpus of genuine end-of-D states (healthy, missing, corrupt, refused, lag, saver, legacy)."""
    specs: list[dict] = []

    def add(name: str, *, expected: bool, manual=None, auto=None, switch=None, saver=None, picks=()):
        specs.append({
            "name": f"{day}_{'req' if expected else 'opt'}_{name}",
            "day": day.isoformat(),
            "expected": expected,
            "manual": manual,
            "auto": auto,
            "switch": switch,
            "saver": saver,
            "picks": list(picks),
        })

    valid_dated = lambda streak=0, best=0, sv="not_earned", eff=day: {
        "kind": "json",
        "body": {
            "schema_version": "bts_contest_streak_manual_v3",
            "season": eff.year,
            "effective_date": eff.isoformat(),
            "active_streak": streak,
            "best_streak": best,
            "saver_state": sv,
            "recorded_at": at_et(eff, 9).isoformat(),
            "expires_at": datetime.combine(eff, time.max, ET).isoformat(),
            "source": "corpus",
        },
    }
    valid_auto = lambda streak=12, best=12, src=(day - timedelta(days=2)).isoformat(), **extra: {
        "kind": "json",
        "body": {
            "schema_version": "bts_contest_streak_auto_v2",
            "season": day.year,
            "activation_id": MATCH_ID,
            "active_streak": streak,
            "best_streak": best,
            "source_date": src,
            "recorded_at": at_et(day, 20).isoformat(),
            **extra,
        },
    }
    valid_switch = lambda status="automatic", season=day.year, act=MATCH_ID, **extra: {
        "kind": "json",
        "body": {
            "schema_version": "bts_season_contract_v1",
            "season": season,
            "activation_id": act,
            "status": status,
            "switched_at": at_et(day, 8).isoformat(),
            "revoked_at": at_et(day, 8).isoformat() if status == "revoked" else None,
            **extra,
        },
    }
    valid_saver = lambda state="active", season=day.year: {
        "kind": "json",
        "body": {"season": season, "state": state},
    }

    for expected in (False, True):
        # 1. Empty / missing states with various switches
        add("empty_no_switch", expected=expected)
        add("empty_switch_auto", expected=expected, switch=valid_switch("automatic"))
        add("empty_switch_revoked", expected=expected, switch=valid_switch("revoked"))
        add("empty_switch_other_next", expected=expected, switch=valid_switch("automatic", season=day.year + 1))
        add("empty_switch_other_prev", expected=expected, switch=valid_switch("automatic", season=day.year - 1))
        add("empty_switch_corrupt_json", expected=expected, switch={"kind": "raw", "raw": "{"})
        add("empty_switch_corrupt_season", expected=expected, switch=valid_switch("automatic", season="bad"))

        # 2. Dated manual observations (healthy, saver contradiction, boundary streaks, other-day dated)
        add("dated_ok_s0", expected=expected, manual=valid_dated(0, 0, "not_earned"),
            switch=valid_switch("revoked"), saver=valid_saver("not_earned"))
        add("dated_ok_s9", expected=expected, manual=valid_dated(9, 9, "not_earned"))
        add("dated_contradict_s10", expected=expected, manual=valid_dated(10, 10, "not_earned"))
        add("dated_contradict_s12", expected=expected, manual=valid_dated(12, 14, "not_earned"),
            auto=valid_auto(), switch=valid_switch(), saver=valid_saver("active"))
        add("dated_contradict_s15", expected=expected, manual=valid_dated(15, 15, "not_earned"))
        add("dated_ok_s16_not_earned", expected=expected, manual=valid_dated(16, 16, "not_earned"))
        add("dated_ok_s12_active", expected=expected, manual=valid_dated(12, 12, "active"))
        add("dated_ok_s12_used", expected=expected, manual=valid_dated(12, 12, "used"))
        add("dated_with_lag_picks", expected=expected, manual=valid_dated(4, 4, "not_earned"),
            picks=[(day + timedelta(days=1)).isoformat(), (day + timedelta(days=2)).isoformat()])
        add("dated_prev_day_no_auto", expected=expected,
            manual=valid_dated(5, 5, "not_earned", eff=day - timedelta(days=1)),
            switch=valid_switch("automatic"))
        add("dated_next_day_with_auto", expected=expected,
            manual=valid_dated(5, 5, "not_earned", eff=day + timedelta(days=1)),
            auto=valid_auto(), switch=valid_switch("revoked"), saver=valid_saver("not_earned"))

        # 3. Corrupt / invalid manual files
        add("manual_bad_json", expected=expected, manual={"kind": "raw", "raw": "{"}, auto=valid_auto())
        add("manual_bad_utf8", expected=expected, manual={"kind": "bytes", "hex": "fffe"}, auto=valid_auto())
        add("manual_non_dict", expected=expected, manual={"kind": "json", "body": [1, 2]}, auto=valid_auto())
        add("manual_legacy_bad_streak_bool", expected=expected, manual={"kind": "json", "body": {"streak": True}})
        add("manual_legacy_bad_streak_neg", expected=expected, manual={"kind": "json", "body": {"streak": -1}})
        add("manual_legacy_bad_saver", expected=expected,
            manual={"kind": "json", "body": {"streak": 3, "saver_available": "yes"}})
        for field, bad_val in [
            ("effective_date", "bad"), ("season", day.year + 1), ("expires_at", "2027-01-01T00:00:00Z"),
            ("active_streak", 58), ("best_streak", -1), ("saver_state", "bogus"),
            ("recorded_at", "naive"), ("source", 123), ("reason", 456),
        ]:
            m = valid_dated()
            m["body"] = dict(m["body"], **{field: bad_val})
            add(f"dated_invalid_{field}", expected=expected, manual=m, auto=valid_auto())

        # 4. Legacy manual overrides (no expiry, expired, exact endpoint, unexpired)
        end_iso = end_utc(day).isoformat()
        before_iso = (end_utc(day) - timedelta(microseconds=1)).isoformat()
        after_iso = (end_utc(day) + timedelta(microseconds=1)).isoformat()
        add("legacy_no_exp_alone_s4", expected=expected,
            manual={"kind": "json", "body": {"streak": 4, "source_date": day.isoformat()}},
            switch=valid_switch(), saver=valid_saver("active"))
        add("legacy_no_exp_alone_s12_uninit", expected=expected,
            manual={"kind": "json", "body": {"streak": 12, "source_date": day.isoformat()}},
            switch=valid_switch("revoked"))
        add("legacy_expired_before_with_auto", expected=expected,
            manual={"kind": "json", "body": {"streak": 4, "source_date": day.isoformat(), "override_expires_at": before_iso}},
            auto=valid_auto(), switch=valid_switch(), saver=valid_saver())
        add("legacy_expired_exact_with_auto", expected=expected,
            manual={"kind": "json", "body": {"streak": 4, "source_date": day.isoformat(), "override_expires_at": end_iso}},
            auto=valid_auto(), switch=valid_switch(), saver=valid_saver())
        add("legacy_unexpired_masks_corrupt_auto", expected=expected,
            manual={"kind": "json", "body": {"streak": 12, "source_date": day.isoformat(), "override_expires_at": after_iso}},
            auto={"kind": "raw", "raw": "{"}, switch=valid_switch(), saver=valid_saver("not_earned"))
        add("legacy_unexpired_switch_corrupt_priority", expected=expected,
            manual={"kind": "json", "body": {"streak": 5, "source_date": day.isoformat(), "override_expires_at": after_iso}},
            switch={"kind": "raw", "raw": "{"})

        # 5. Automatic file validation & coverage / future grace
        add("auto_bad_json", expected=expected, auto={"kind": "raw", "raw": "{"})
        add("auto_non_dict", expected=expected, auto={"kind": "json", "body": "str"})
        add("auto_bad_streak_over", expected=expected, auto=valid_auto(streak=58, best=58))
        add("auto_bad_best_under", expected=expected, auto=valid_auto(streak=12, best=5))
        add("auto_bad_best_over", expected=expected, auto=valid_auto(streak=12, best=58))
        add("auto_bad_saver_bool", expected=expected, auto=valid_auto(saver_available="nope"))
        add("auto_bad_season_type", expected=expected, auto=valid_auto(season=True), switch=valid_switch())
        add("auto_bad_act_id", expected=expected, auto=valid_auto(activation_id="XYZ"), switch=valid_switch())
        add("auto_undated_completely", expected=expected,
            auto=valid_auto(source_date=None, recorded_at=None), switch=valid_switch(), saver=valid_saver())
        add("auto_missing_source_date_only", expected=expected,
            auto=valid_auto(source_date=None), switch=valid_switch(), saver=valid_saver())
        add("auto_grace_d_plus_2", expected=expected,
            auto=valid_auto(src=(day + timedelta(days=2)).isoformat()), switch=valid_switch(), saver=valid_saver())
        add("auto_beyond_grace_d_plus_3", expected=expected,
            auto=valid_auto(src=(day + timedelta(days=3)).isoformat()), switch=valid_switch(), saver=valid_saver())
        add("auto_lag_1_info", expected=expected,
            auto=valid_auto(src=(day - timedelta(days=1)).isoformat()),
            switch=valid_switch(), saver=valid_saver(), picks=[day.isoformat()])
        add("auto_lag_2_warn_plus_saver_plus_refusal", expected=expected,
            manual={"kind": "json", "body": {"streak": 2}},
            auto=valid_auto(streak=12, best=12, src=(day - timedelta(days=3)).isoformat()),
            switch=valid_switch("revoked"), saver=valid_saver("not_earned"),
            picks=[(day - timedelta(days=1)).isoformat(), day.isoformat()])

        # 6. Switch & auto refusal kinds
        for field, bad_val in [
            ("schema_version", "bad"), ("season", True), ("status", "paused"),
            ("activation_id", "short"), ("switched_at", None), ("revoked_at", "bad"), ("note", 99),
        ]:
            add(f"switch_corrupt_{field}", expected=expected,
                auto=valid_auto(), switch=valid_switch(**{field: bad_val}), saver=valid_saver())
        add("auto_v1_schema", expected=expected,
            auto={
                "kind": "json",
                "body": {
                    "schema_version": "bts_contest_streak_auto_v1",
                    "active_streak": 6, "best_streak": 6,
                    "source_date": day.isoformat(), "recorded_at": at_et(day, 20).isoformat(),
                },
            },
            switch=valid_switch(), saver=valid_saver())
        add("auto_null_schema", expected=expected,
            auto=valid_auto(schema_version=None), switch=valid_switch(), saver=valid_saver())
        add("auto_int_schema", expected=expected,
            auto=valid_auto(schema_version=42), switch=valid_switch(), saver=valid_saver())
        add("auto_wrong_season_prev", expected=expected,
            auto=valid_auto(season=day.year - 1), switch=valid_switch(), saver=valid_saver())
        add("auto_wrong_season_next", expected=expected,
            auto=valid_auto(season=day.year + 1), switch=valid_switch(), saver=valid_saver())
        add("auto_mismatch_activation", expected=expected,
            auto=valid_auto(activation_id=OTHER_ID), switch=valid_switch(), saver=valid_saver())

        # 7. Flag-based saver cases (in zone vs out of zone, missing, corrupt, other season, contradiction, crash)
        for streak in (4, 12, 16):
            add(f"saver_missing_s{streak}", expected=expected,
                auto=valid_auto(streak=streak, best=max(streak, 12)), switch=valid_switch())
            add(f"saver_bad_json_s{streak}", expected=expected,
                auto=valid_auto(streak=streak, best=max(streak, 12)), switch=valid_switch(),
                saver={"kind": "raw", "raw": "{"})
            add(f"saver_bad_state_same_yr_s{streak}", expected=expected,
                auto=valid_auto(streak=streak, best=max(streak, 12)), switch=valid_switch(),
                saver=valid_saver("bogus", season=day.year))
            add(f"saver_other_yr_next_s{streak}", expected=expected,
                auto=valid_auto(streak=streak, best=max(streak, 12)), switch=valid_switch(),
                saver=valid_saver("active", season=day.year + 1))
            add(f"saver_other_yr_prev_s{streak}", expected=expected,
                auto=valid_auto(streak=streak, best=max(streak, 12)), switch=valid_switch(),
                saver=valid_saver("not_earned", season=day.year - 1))
            add(f"saver_not_earned_s{streak}", expected=expected,
                auto=valid_auto(streak=streak, best=max(streak, 12)), switch=valid_switch(),
                saver=valid_saver("not_earned", season=day.year))
            add(f"saver_used_s{streak}", expected=expected,
                auto=valid_auto(streak=streak, best=max(streak, 12)), switch=valid_switch(),
                saver=valid_saver("used", season=day.year))
        add("saver_overlong_int_crash", expected=expected,
            auto=valid_auto(), switch=valid_switch(),
            saver={"kind": "raw", "raw": '{"season":' + ("9" * 5000) + "}"})

    return specs


def materialize_spec(root: Path, spec: dict) -> Path:
    picks = root / spec["name"] / "picks"
    acct = picks / "account_state"
    acct.mkdir(parents=True, exist_ok=True)

    def write_entry(path: Path, entry: dict | None):
        if entry is None:
            return
        if entry["kind"] == "json":
            path.write_text(json.dumps(entry["body"]), encoding="utf-8")
        elif entry["kind"] == "raw":
            path.write_text(entry["raw"], encoding="utf-8")
        elif entry["kind"] == "bytes":
            path.write_bytes(bytes.fromhex(entry["hex"]))

    write_entry(acct / "contest_streak.manual.json", spec["manual"])
    write_entry(acct / "contest_streak.json", spec["auto"])
    write_entry(acct / "season_contract.json", spec["switch"])
    write_entry(acct / "saver_state.json", spec["saver"])
    for pdate in spec["picks"]:
        (picks / f"{pdate}.json").write_text('{"result":"hit"}', encoding="utf-8")
    return picks


def worker_eval(mode: str, specs_file: Path, root: Path) -> None:
    from bts.health import alert, contest_state, runner

    specs = json.loads(specs_file.read_text(encoding="utf-8"))
    out = {}
    for spec in specs:
        picks = root / spec["name"] / "picks"
        day = date.fromisoformat(spec["day"])
        expected = spec["expected"]
        if mode == "base_endpoint":
            res = runner._safe_run(
                "contest_state",
                lambda: contest_state.check(picks, expected=expected, now=end_utc(day)),
            )
            out[spec["name"]] = [(a.level, a.source, a.incident_key) for a in res]
        elif mode == "head_historical":
            res = []
            for delta_days, hr, mn in [(1, 0, 15), (1, 2, 0), (4, 14, 0)]:
                wall = at_et(day + timedelta(days=delta_days), hr, mn)

                class Clock(datetime):
                    @classmethod
                    def now(cls, tz=None):
                        return wall.astimezone(tz) if tz is not None else wall.replace(tzinfo=None)

                orig_safe = runner._safe_run
                with (
                    patch.object(runner, "datetime", Clock),
                    patch.object(contest_state, "datetime", Clock),
                    patch.object(alert, "send_dm", return_value="ack"),
                    patch.object(
                        runner, "_safe_run",
                        lambda n, fn: orig_safe(n, fn) if n == "contest_state" else [],
                    ),
                ):
                    alerts = runner.run_all_checks(
                        picks, root / "models", None, today=day, contest_state_expected=expected,
                        thresholds_overrides={
                            "warn_attention_state": picks / ".h/warn.json",
                            "health_dm_delivery_status": picks / ".h/dm.json",
                        },
                    )
                relevant = [
                    (a.level, a.source, a.incident_key, a.message)
                    for a in alerts
                    if a.source in {"contest_state", "saver_state"}
                    or a.incident_key == "health_runner:contest_state"
                ]
                res.append(relevant)
            assert res[0] == res[1] == res[2], (spec["name"], res)
            out[spec["name"]] = [(lvl, src, key) for (lvl, src, key, _msg) in res[0]]
    print(json.dumps(out))


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
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        logging.disable(logging.CRITICAL)
        with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
            worker_eval(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
        return

    signal.alarm(55)
    logging.disable(logging.CRITICAL)
    base_env = {**os.environ, "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(BASE_SRC)}
    head_env = {**os.environ, "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(HEAD_SRC)}

    # 1. Run check's drivers on BASE
    for label, cmd in [
        ("base_operator_scheduler", [sys.executable, str(CC_DIR / "operator_cases.py"), "scheduler"]),
        ("base_probe_messages", [sys.executable, str(CC_DIR / "probe.py"), "messages"]),
        ("base_probe_edges", [sys.executable, str(CC_DIR / "probe.py"), "edges"]),
    ]:
        proc = subprocess.run(cmd, env=base_env, capture_output=True, text=True, timeout=30, check=True)
        print(f"[PASS] {label}: exit=0, lines={len(proc.stdout.splitlines())}")

    # 2. Verify W20 gone on HEAD (real scheduler EOD at 00:15 ET + corpus across summer, winter, Dec 31)
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
