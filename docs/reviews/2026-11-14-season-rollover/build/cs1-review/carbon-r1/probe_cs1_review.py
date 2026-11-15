"""Adversarial probe script for CS1 review (defensive reads & model-state isolation).

Compares `main`'s `src/bts/contest_state.py` (loaded from git without touching any
tracked file) against `HEAD`'s `src/bts/contest_state.py` across all 7 review items.
No network, no box, no contest account.
"""
from __future__ import annotations

import os
import time

os.environ["TZ"] = "America/New_York"
time.tzset()

import datetime as dt
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

import bts.cli as cli_mod
import bts.contest_fetch as contest_fetch
import bts.contest_state as head_cs
import bts.daily_decision as daily_decision
import bts.picks as picks_mod
import bts.posting as posting_mod
import bts.strategy as strategy_mod
import bts.web as web_mod
from bts.health import contest_state as health_cs
from bts.health import streak_validation as health_sv
from bts.saver_state import transition_saver_state

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
NOW = dt.datetime(2027, 3, 24, 16, 0, tzinfo=dt.timezone.utc)
ET = ZoneInfo("America/New_York")


def load_main_contest_state():
    src = subprocess.check_output(
        ["git", "show", "main:src/bts/contest_state.py"],
        cwd=REPO_ROOT,
        text=True,
    )
    tmp = Path(tempfile.mkdtemp(prefix="cs1-main-mod-"))
    mod_file = tmp / "contest_state_main.py"
    mod_file.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("bts_contest_state_main", mod_file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bts_contest_state_main"] = mod
    spec.loader.exec_module(mod)
    return mod


main_cs = load_main_contest_state()


def outcome(fn):
    try:
        val = fn()
        if hasattr(val, "__dataclass_fields__"):
            d = asdict(val)
            return {"ok": True, "type": type(val).__name__, "value": d}
        return {"ok": True, "type": type(val).__name__, "value": val}
    except BaseException as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


def write_corrupt(path: Path, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            path.rmdir()
        else:
            path.chmod(0o644)
            path.unlink()
    if kind == "utf8":
        path.write_bytes(b"\xff")
    elif kind == "non_object":
        path.write_bytes(b"[]")
    elif kind == "json":
        path.write_bytes(b"{broken")
    elif kind == "dir_oserror":
        path.mkdir()
    elif kind == "chmod000":
        path.write_text('{"streak": 4, "result": "miss", "active_streak": 4}', encoding="utf-8")
        path.chmod(0o000)
    elif kind == "dangling_symlink":
        path.symlink_to(path.parent / "does_not_exist_target.json")
    elif kind == "symlink_loop":
        path.symlink_to(path)
    else:
        raise ValueError(kind)


def cleanup_perms(root: Path) -> None:
    for p in [root, *root.rglob("*")]:
        try:
            if not p.is_symlink() and p.exists():
                p.chmod(0o755 if p.is_dir() else 0o644)
        except OSError:
            pass


def make_auto(picks_dir: Path, active: int = 13, best: int | None = 20,
              source_date: str | None = "2027-03-24",
              recorded_at: str | None = "2027-03-24T16:00:00Z", **extra) -> Path:
    sd = picks_dir / "account_state"
    sd.mkdir(parents=True, exist_ok=True)
    body: dict = {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": active,
        "source": "mlb_bts_profile",
    }
    if best is not None:
        body["best_streak"] = best
    if source_date is not None:
        body["source_date"] = source_date
    if recorded_at is not None:
        body["recorded_at"] = recorded_at
    body.update(extra)
    p = sd / "contest_streak.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


def make_manual(picks_dir: Path, active: int = 7, best: int | None = 12,
                source_date: str | None = "2027-03-24",
                expires_at: str | None = "2027-03-25T16:00:00Z", **extra) -> Path:
    sd = picks_dir / "account_state"
    sd.mkdir(parents=True, exist_ok=True)
    body: dict = {
        "schema_version": "bts_contest_streak_manual_v2",
        "active_streak": active,
        "source": "manual_cli",
    }
    if best is not None:
        body["best_streak"] = best
    if source_date is not None:
        body["source_date"] = source_date
    if expires_at is not None:
        body["override_expires_at"] = expires_at
    body.update(extra)
    p = sd / "contest_streak.manual.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


def probe_rows_1_to_8() -> dict:
    res: dict = {}
    corrupt_kinds = ["utf8", "non_object", "json", "dir_oserror", "chmod000"]

    # Row 1: date-named pick scans
    r1 = {}
    for kind in corrupt_kinds + ["dangling_symlink"]:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            make_auto(p, source_date="2027-03-22")
            write_corrupt(p / "2027-03-23.json", kind)
            (p / "2027-03-24.json").write_text('{"result": "miss"}', encoding="utf-8")
            try:
                r1[kind] = {
                    "main_resolved": outcome(lambda: main_cs._resolved_pick_dates(p)),
                    "head_resolved": outcome(lambda: head_cs._resolved_pick_dates(p)),
                    "main_miss": outcome(lambda: main_cs._has_unconfirmed_miss(p, dt.date(2027, 3, 22))),
                    "head_miss": outcome(lambda: head_cs._has_unconfirmed_miss(p, dt.date(2027, 3, 22))),
                    "main_decision": outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                    "head_decision": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                }
            finally:
                cleanup_perms(p)
    # Near Row 1: JSON object with unhashable "result": []
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "2027-03-24.json").write_text('{"result": []}', encoding="utf-8")
        r1["near_unhashable_result_list"] = {
            "main_resolved": outcome(lambda: main_cs._resolved_pick_dates(p)),
            "head_resolved": outcome(lambda: head_cs._resolved_pick_dates(p)),
            "main_miss": outcome(lambda: main_cs._has_unconfirmed_miss(p, dt.date(2027, 3, 22))),
            "head_miss": outcome(lambda: head_cs._has_unconfirmed_miss(p, dt.date(2027, 3, 22))),
        }
    res["row1"] = r1

    # Row 2: contest files not UTF-8 (and other corruptions) + manual override precedence
    r2 = {}
    for fname in ("contest_streak.json", "contest_streak.manual.json"):
        r2[fname] = {}
        for kind in corrupt_kinds:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td)
                write_corrupt(p / "account_state" / fname, kind)
                try:
                    r2[fname][kind] = {
                        "main": outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                        "head": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                    }
                finally:
                    cleanup_perms(p)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        write_corrupt(p / "account_state" / "contest_streak.json", "utf8")
        make_manual(p, active=4, best=9)
        r2["unexpired_manual_over_utf8_auto"] = {
            "main": outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
            "head": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
        }
    res["row2"] = r2

    # Row 3: stored automatic bounds + manual sequencing + near-row non-int best & boundaries
    r3 = {"auto": {}, "manual_unexpired": {}, "near": {}}
    for label, active, best in [
        ("best_below_streak_13_10", 13, 10),
        ("negative_best_0_neg1", 0, -1),
        ("streak_above_57_58_58", 58, 58),
        ("streak_above_57_no_best_58_None", 58, None),
        ("best_above_57_13_58", 13, 58),
        ("boundary_57_57", 57, 57),
        ("boundary_0_0", 0, 0),
    ]:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            make_auto(p, active=active, best=best)
            r3["auto"][label] = {
                "main_req": outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                "head_req": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                "head_opt": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=False, now=NOW)),
            }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            make_manual(p, active=active, best=best, expires_at="2027-03-25T16:00:00Z")
            r3["manual_unexpired"][label] = {
                "main_req": outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                "head_req": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
            }
    for bad_best_label, bad_best in [("str_10", "10"), ("bool_true", True), ("float_99_5", 99.5)]:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            make_auto(p, active=13, best=20, best_streak=bad_best)
            r3["near"][bad_best_label] = {
                "main": outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                "head": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
            }
    res["row3"] = r3

    # Row 4: undated vs dated automatic observations
    r4 = {}
    r4_cases = {
        "bare_no_dates": {"active_streak": 14, "best_streak": 18},
        "null_dates": {"active_streak": 14, "best_streak": 18, "source_date": None, "recorded_at": None},
        "empty_strings": {"active_streak": 14, "best_streak": 18, "source_date": "", "as_of": "", "recorded_at": ""},
        "malformed_all": {"active_streak": 14, "best_streak": 18, "source_date": "bad", "as_of": "bad", "recorded_at": "bad"},
        "dated_by_source_date": {"active_streak": 14, "best_streak": 18, "source_date": "2027-03-24"},
        "dated_by_as_of": {"active_streak": 14, "best_streak": 18, "as_of": "2027-03-24"},
        "dated_by_recorded_at_only": {"active_streak": 14, "best_streak": 18, "source_date": None, "recorded_at": "2027-03-24T16:00:00Z"},
        "malformed_source_date_valid_as_of": {"active_streak": 14, "best_streak": 18, "source_date": "not-a-date", "as_of": "2027-03-24"},
        "future_source_date": {"active_streak": 14, "best_streak": 18, "source_date": "2099-01-01"},
        "future_recorded_at_only": {"active_streak": 14, "best_streak": 18, "recorded_at": "2099-01-01T00:00:00Z"},
        "compact_iso_recorded_at_20270324": {"active_streak": 14, "best_streak": 18, "recorded_at": "20270324"},
    }
    for name, payload in r4_cases.items():
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            sd = p / "account_state"
            sd.mkdir()
            (sd / "contest_streak.json").write_text(json.dumps(payload), encoding="utf-8")
            r4[name] = {
                "main_req": outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                "head_req": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                "head_opt": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=False, now=NOW)),
                "health_expected": [asdict(a) for a in health_cs.check(p, expected=True, now=NOW)],
            }
    res["row4"] = r4

    # Rows 5, 6, 7, 8: model state isolation, fallback refusal, missing contest precedence, {} defaults
    r5, r6, r7 = {}, {}, {}
    for kind in corrupt_kinds:
        # Row 5: auto and manual
        for obs in ("auto", "manual"):
            with tempfile.TemporaryDirectory() as td:
                p = Path(td)
                if obs == "auto":
                    make_auto(p, active=13, best=20)
                else:
                    make_manual(p, active=7, best=12)
                write_corrupt(p / "streak.json", kind)
                records: list[logging.LogRecord] = []
                handler = logging.Handler()
                handler.emit = records.append  # type: ignore[method-assign]
                head_cs.logger.addHandler(handler)
                head_cs.logger.setLevel(logging.WARNING)
                try:
                    r5[f"{obs}_{kind}"] = {
                        "main": outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                        "head": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                        "warnings": [r.getMessage() for r in records],
                    }
                finally:
                    head_cs.logger.removeHandler(handler)
                    cleanup_perms(p)

        # Row 6: no contest, require_contest_state=False
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            write_corrupt(p / "streak.json", kind)
            try:
                r6[kind] = {
                    "main": outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=False, now=NOW)),
                    "head": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=False, now=NOW)),
                }
            finally:
                cleanup_perms(p)

        # Row 7: no contest, require_contest_state=True
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            write_corrupt(p / "streak.json", kind)
            try:
                r7[kind] = {
                    "main": outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                    "head": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)),
                }
            finally:
                cleanup_perms(p)

    # Near Row 6: symlink loop on streak.json (Path.exists() swallows ELOOP -> False)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        write_corrupt(p / "streak.json", "symlink_loop")
        r6["near_symlink_loop"] = {
            "main": outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=False, now=NOW)),
            "head": outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=False, now=NOW)),
        }

    res["row5"] = r5
    res["row6"] = r6
    res["row7"] = r7

    # Row 8: JSON objects with absent keys and wrong value types
    r8 = {}
    for name, body in [
        ("empty_obj", {}),
        ("only_streak", {"streak": 6}),
        ("only_saver_false", {"saver_available": False}),
        ("null_values", {"streak": None, "saver_available": None}),
        ("string_values", {"streak": "4", "saver_available": "false"}),
        ("bool_streak", {"streak": True, "saver_available": 0}),
        ("negative_streak", {"streak": -3, "saver_available": True}),
    ]:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "streak.json").write_text(json.dumps(body), encoding="utf-8")
            r8[name] = {
                "scorers": {
                    "load_streak": picks_mod.load_streak(p),
                    "load_saver_available": picks_mod.load_saver_available(p),
                    "load_applied_dates": picks_mod.load_applied_dates(p),
                },
                "main_model_only": outcome(lambda: main_cs.load_decision_streak_state(p, now=NOW)),
                "head_model_only": outcome(lambda: head_cs.load_decision_streak_state(p, now=NOW)),
            }
            make_auto(p, active=11, best=15)
            r8[name]["main_with_contest"] = outcome(lambda: main_cs.load_decision_streak_state(p, now=NOW))
            r8[name]["head_with_contest"] = outcome(lambda: head_cs.load_decision_streak_state(p, now=NOW))
    res["row8_and_model_matrix"] = r8
    return res


def probe_saver_mismatch_and_callers() -> dict:
    out: dict = {}

    # Saver recorded in one place and not the other
    saver_cases = {}
    for streak_json_saver in (True, False, None):
        for flag_state in (None, "not_earned", "active", "used"):
            for has_contest in (False, True):
                with tempfile.TemporaryDirectory() as td:
                    p = Path(td)
                    if streak_json_saver is not None:
                        (p / "streak.json").write_text(
                            json.dumps({"streak": 12, "saver_available": streak_json_saver}),
                            encoding="utf-8",
                        )
                    if flag_state is not None:
                        transition_saver_state(
                            p, expected_prior="uninitialized", new_state=flag_state,
                            season=2027, source="probe",
                        )
                    if has_contest:
                        make_auto(p, active=12, best=14, saver_available=True)
                    m = main_cs.load_decision_streak_state(p, now=NOW)
                    h = head_cs.load_decision_streak_state(p, now=NOW)
                    key = f"sjson={streak_json_saver}_flag={flag_state}_contest={has_contest}"
                    saver_cases[key] = {
                        "equal_excl_error": (
                            (m.streak, m.saver_available, m.model_streak, m.model_saver_available, m.contest_saver_available)
                            == (h.streak, h.saver_available, h.model_streak, h.model_saver_available, h.contest_saver_available)
                        ),
                        "decision_saver": h.saver_available,
                        "model_saver": h.model_saver_available,
                    }
    out["saver_matrix_all_match_main"] = all(v["equal_excl_error"] for v in saver_cases.values())
    out["saver_matrix_count"] = len(saver_cases)

    # Caller checks on Row 3, Row 4, Row 5, Row 6
    caller_checks: dict = {}

    # 1. Dashboard on Row 5 (corrupt streak.json + valid contest): works cleanly
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        make_auto(p, active=9, best=15)
        write_corrupt(p / "streak.json", "utf8")
        with patch.object(web_mod, "PICKS_DIR", p):
            ds, err = web_mod.load_dashboard_streak_context()
            sub = web_mod._streak_subtitle(ds, err)
            caller_checks["dashboard_row5"] = {"streak": ds.streak if ds else None, "err": err, "subtitle": sub}

    # 2. Dashboard on Row 6 (corrupt streak.json + no contest): render_page fallback web.load_streak() crashes!
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        write_corrupt(p / "streak.json", "utf8")
        with patch.object(web_mod, "PICKS_DIR", p):
            ds, err = web_mod.load_dashboard_streak_context()
            fallback_streak = outcome(web_mod.load_streak)
            caller_checks["dashboard_row6"] = {
                "context_ds": ds,
                "context_err": err,
                "render_page_load_streak_fallback": fallback_streak,
            }

    # 3. Dashboard on Row 3 (auto best < active, healthy streak.json=4):
    #    - render_page falls back to model streak 4 under "Streak State Error"
    #    - saver_transition_response ("Never raises") raises ContestStateError!
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "streak.json").write_text('{"streak": 4, "saver_available": true}', encoding="utf-8")
        make_auto(p, active=13, best=10)
        transition_saver_state(p, expected_prior="uninitialized", new_state="active", season=2027, source="probe")
        with patch.object(web_mod, "PICKS_DIR", p):
            ds, err = web_mod.load_dashboard_streak_context()
            hero_streak = ds.streak if ds is not None else web_mod.load_streak()
            saver_ctx = outcome(lambda: web_mod.saver_dashboard_context(p, now=NOW))
            saver_post = outcome(lambda: web_mod.saver_transition_response(
                p, expected_prior="active", new_state="used", same_origin=True, now=NOW,
            ))
            caller_checks["dashboard_row3"] = {
                "context_err": err,
                "displayed_hero_streak": hero_streak,
                "subtitle": web_mod._streak_subtitle(ds, err),
                "saver_dashboard_context": saver_ctx,
                "saver_transition_response": saver_post,
            }

    # 4. Health check streak_validation on 0xff and [] (untouched module)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        write_corrupt(p / "streak.json", "utf8")
        sv_utf8 = outcome(lambda: health_sv.check(p))
        write_corrupt(p / "streak.json", "non_object")
        sv_list = outcome(lambda: health_sv.check(p))
        caller_checks["health_streak_validation_untouched"] = {"utf8": sv_utf8, "non_object": sv_list}

    # 5. DM formatting + decision.json when model_streak=None, model_saver_available=None (Row 5)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        make_auto(p, active=9, best=15)
        write_corrupt(p / "streak.json", "utf8")
        st = head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)
        dm_text = posting_mod.format_post("Steven Kwan", "CLE", "Gerrit Cole", 0.812, st.streak)
        rec = daily_decision.write_decision(
            "2027-03-24", p,
            action="single", source="mdp",
            primary={"batter_id": 1, "batter_name": "Steven Kwan", "team": "CLE", "game_pk": 100, "p_game_hit": 0.812},
            streak=st.streak, saver_available=st.saver_available,
            state_source=st.source, state_status=st.status,
            allow_double=st.allow_double,
            contest_source_date=(st.contest_source_date.isoformat() if st.contest_source_date else None),
            delivery_status="delivered", scoreable=True,
            best_streak=st.best_streak, best_status=st.best_status,
            now=NOW,
        )
        caller_checks["dm_and_decision_row5"] = {"dm_text": dm_text, "decision_json": rec}

    # 6. Model-only fallback with {"streak": null} -> streak=None breaks downstream (same on main and HEAD)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "streak.json").write_text('{"streak": null, "saver_available": null}', encoding="utf-8")
        st = head_cs.load_decision_streak_state(p, require_contest_state=False, now=NOW)
        ctx = strategy_mod.DecisionContext(0.82, None, False, "2027-03-24", True, strategy_mod._load_mdp())
        caller_checks["null_streak_in_streak_json_model_fallback"] = {
            "state": asdict(st),
            "policy_decision": outcome(lambda: strategy_mod.resolve_policy_decision(ctx, st.streak, st.saver_available)),
            "dm_text": posting_mod.format_post("Steven Kwan", "CLE", "Gerrit Cole", 0.812, st.streak),
        }

    # 7. Real fetch writer: validate_fetch + build_observation when seasonBestStreak > 57
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        obs_58 = contest_fetch.build_observation(
            {"activeStreak": 0, "seasonBestStreak": 58},
            source_date=dt.date(2027, 3, 24),
            user_id=1, username="fixture", recorded_at=NOW,
        )
        (p / "account_state").mkdir()
        (p / "account_state" / "contest_streak.json").write_text(json.dumps(obs_58), encoding="utf-8")
        main_58 = outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW))
        head_58 = outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW))
        # Recovery via set-contest-streak CLI
        cli_res = CliRunner().invoke(cli_mod.cli, [
            "set-contest-streak", "--streak", "0", "--best-streak", "57",
            "--source-date", "2027-03-24", "--override-expires-at", "2027-03-25T23:00:00Z",
            "--picks-dir", str(p),
        ])
        recovered_58 = outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=True, now=NOW))
        caller_checks["fetch_writer_best_58"] = {
            "build_observation_succeeded": True,
            "main_loader": main_58,
            "head_loader": head_58,
            "cli_recovery_exit": cli_res.exit_code,
            "head_after_manual_override": recovered_58,
        }

    # 8. fetch-contest-streak CLI when prior contest_streak.json is 0xff or [] (untouched cli.py:2032)
    for bad_kind in ("utf8", "non_object"):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            write_corrupt(p / "account_state" / "contest_streak.json", bad_kind)
            with patch("bts.leaderboard.auth.load_session_cookies", return_value={}), \
                 patch("bts.leaderboard.auth.extract_uid", return_value="fixture"), \
                 patch("bts.leaderboard.auth.fetch_login_session",
                       return_value=type("S", (), {"xsid": "x", "user_id": 1, "username": "fixture"})()), \
                 patch("bts.contest_fetch.fetch_profile",
                       return_value={"activeStreak": 2, "seasonBestStreak": 5, "predictions": []}), \
                 patch("bts.cli._fetch_rounds", return_value={}):
                r = CliRunner().invoke(cli_mod.cli, ["fetch-contest-streak", "--picks-dir", str(p)])
                caller_checks[f"fetch_cli_over_corrupt_prior_{bad_kind}"] = {
                    "exit_code": r.exit_code,
                    "exception": type(r.exception).__name__ if r.exception else None,
                    "message": str(r.exception) if r.exception else None,
                }

    out["callers"] = caller_checks
    return out


def probe_precedence() -> dict:
    matrix: dict = {}
    manual_states = [
        ("unexpired", {"active_streak": 7, "best_streak": 12, "source_date": "2027-03-24",
                       "override_expires_at": "2027-03-25T16:00:00Z"}),
        ("expired", {"active_streak": 7, "best_streak": 12, "source_date": "2027-03-24",
                     "override_expires_at": "2027-03-23T16:00:00Z"}),
        ("legacy_no_expiry", {"active_streak": 7, "best_streak": 12, "source_date": "2027-03-24"}),
        ("corrupt_utf8", b"\xff"),
        ("corrupt_json", b"{broken"),
    ]
    auto_states = [
        ("corrupt_utf8", b"\xff"),
        ("corrupt_json", b"{broken"),
        ("row3_invalid_13_10", {"active_streak": 13, "best_streak": 10, "source_date": "2027-03-24",
                                "recorded_at": "2027-03-24T16:00:00Z"}),
        ("row4_undated", {"active_streak": 14, "best_streak": 18}),
    ]
    for m_name, m_val in manual_states:
        for a_name, a_val in auto_states:
            for req in (False, True):
                with tempfile.TemporaryDirectory() as td:
                    p = Path(td)
                    sd = p / "account_state"
                    sd.mkdir()
                    mp = sd / "contest_streak.manual.json"
                    ap = sd / "contest_streak.json"
                    if isinstance(m_val, bytes):
                        mp.write_bytes(m_val)
                    else:
                        mp.write_text(json.dumps(m_val), encoding="utf-8")
                    if isinstance(a_val, bytes):
                        ap.write_bytes(a_val)
                    else:
                        ap.write_text(json.dumps(a_val), encoding="utf-8")
                    m_out = outcome(lambda: main_cs.load_decision_streak_state(p, require_contest_state=req, now=NOW))
                    h_out = outcome(lambda: head_cs.load_decision_streak_state(p, require_contest_state=req, now=NOW))
                    same = (
                        m_out["ok"] == h_out["ok"]
                        and (
                            (m_out["ok"] and m_out["value"]["streak"] == h_out["value"]["streak"]
                             and m_out["value"]["status"] == h_out["value"]["status"])
                            or (not m_out["ok"] and m_out["error"] == h_out["error"])
                        )
                    )
                    matrix[f"manual={m_name}__auto={a_name}__req={req}"] = {
                        "same_as_main": same,
                        "main": m_out,
                        "head": h_out,
                    }
    return matrix


def setup_false_green_scratch() -> str:
    """Create isolated scratch package with main's src/bts/contest_state.py for yield_run pytest."""
    scratch = OUT_DIR / ".scratch-main-pkg"
    pkg = scratch / "bts"
    if pkg.exists():
        for child in pkg.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
    pkg.mkdir(parents=True, exist_ok=True)
    for item in (REPO_ROOT / "src" / "bts").iterdir():
        if item.name == "contest_state.py":
            main_src = subprocess.check_output(
                ["git", "show", "main:src/bts/contest_state.py"],
                cwd=REPO_ROOT,
                text=True,
            )
            (pkg / "contest_state.py").write_text(main_src, encoding="utf-8")
        elif item.name == "__pycache__":
            continue
        else:
            (pkg / item.name).symlink_to(item)
    return str(scratch)


def main() -> None:
    with patch("socket.socket.connect", side_effect=AssertionError("NETWORK FORBIDDEN")):
        results = {
            "rows": probe_rows_1_to_8(),
            "callers_and_model": probe_saver_mismatch_and_callers(),
            "precedence": probe_precedence(),
            "scratch_main_pkg": setup_false_green_scratch(),
        }
    out_file = OUT_DIR / "probe_results.json"
    out_file.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    assert results["callers_and_model"]["saver_matrix_all_match_main"] is True
    print("Probe completed and saved to", out_file)


if __name__ == "__main__":
    main()
