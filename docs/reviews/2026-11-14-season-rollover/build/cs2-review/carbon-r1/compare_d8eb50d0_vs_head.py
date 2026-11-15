#!/usr/bin/env python3
"""Differential adversarial probe harness: d8eb50d0 (base) vs 3b4196be (CS2 HEAD).

Runs identical probe scenarios in subprocesses against:
  - BASE: d8eb50d0's `src` extracted into a fresh TemporaryDirectory via `git archive`
  - HEAD: `src` in this checkout (3b4196be)
and writes full structured JSON evidence to .codex-review/cs2-review/differential-probes.log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
HEAD_SRC = ROOT / "src"
LOG_PATH = REVIEW_DIR / "differential-probes.log"

WORKER_CODE = r'''
import datetime as dt
import io
import json
import os
import sys
import tempfile
from contextlib import nullcontext, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

import bts
import bts.cli as cli_mod
from bts.cli import cli
import bts.contest_state as cs_mod
from bts.contest_state import ContestStateError, load_decision_streak_state
from bts.health.streak_validation import check as check_streak_health
import bts.picks as p
import bts.scheduler as sched_mod
from bts.state.export import export_initial_state
import bts.strategy as strat_mod
import bts.web as web_mod

ET = ZoneInfo("America/New_York")
MODE = sys.argv[1]  # "BASE" or "HEAD"
LEGACY = (
    b'{"streak": 13, "saver_available": false, '
    b'"updated": "2026-09-13T05:01:02.123456+00:00", '
    b'"applied_dates": ["2026-09-11", "2026-09-12"]}'
)
DEEP = b"[" * 200_000 + b"]" * 200_000


def call_capture(fn, *args, **kwargs):
    try:
        val = fn(*args, **kwargs)
        return {"ok": True, "value": val, "type": type(val).__name__}
    except BaseException as exc:
        return {"ok": False, "exc": type(exc).__name__, "msg": str(exc)[:160]}


def make_pick(batter_id=700363, game_pk=778899, day="2027-03-24", name="Steven Kwan"):
    return p.Pick(
        batter_name=name,
        batter_id=batter_id,
        team="CLE",
        lineup_position=1,
        pitcher_name="Gerrit Cole",
        pitcher_id=543037,
        p_game_hit=0.82,
        flags=[],
        projected_lineup=False,
        game_pk=game_pk,
        game_time=f"{day}T23:10:00Z",
    )


def plant_scoreable_daily(picks_dir: Path, day: str, *, double: bool = False,
                          result=None, slot_results=None, bluesky_uri="at://did:plc:test/post/1"):
    primary = make_pick(700363, 778899, day, "Steven Kwan")
    dd = make_pick(666126, 778900, day, "Bobby Witt Jr.") if double else None
    daily = p.DailyPick(
        date=day,
        run_time=f"{day}T18:00:00+00:00",
        pick=primary,
        double_down=dd,
        runner_up=None,
        result=result,
        slot_results=slot_results,
        bluesky_posted=True,
        bluesky_uri=bluesky_uri,
        notification_sent=True,
        delivery_attempted=True,
    )
    p.save_pick(daily, picks_dir)
    return daily


def readers_snapshot(picks_dir: Path, date_str: str = "2027-03-24",
                     now_dt: dt.datetime | None = None):
    if now_dt is None:
        now_dt = dt.datetime(2027, 3, 24, 20, 0, tzinfo=ET)
    year = dt.date.fromisoformat(date_str).year
    snap = {}
    raw_path = picks_dir / "streak.json"
    snap["file_exists"] = raw_path.exists()
    if raw_path.exists() and raw_path.is_file():
        try:
            snap["file_bytes"] = raw_path.read_text(encoding="utf-8")
        except Exception as exc:
            snap["file_bytes"] = f"<unreadable:{type(exc).__name__}>"
    # Dateless vs season-scoped picks readers
    snap["load_streak_dateless"] = call_capture(p.load_streak, picks_dir)
    if MODE == "HEAD":
        snap["load_streak_season"] = call_capture(p.load_streak, picks_dir, season=year)
        snap["load_saver_season"] = call_capture(p.load_saver_available, picks_dir, season=year)
        snap["load_applied_season"] = call_capture(p.load_applied_dates, picks_dir, season=year)
    # Row 10 contest_state model-only fallback
    snap["row10_model_only"] = call_capture(
        lambda: (
            lambda st: {
                "streak": st.streak,
                "model_streak": st.model_streak,
                "model_saver": st.model_saver_available,
                "model_error": st.model_state_error,
            }
        )(load_decision_streak_state(picks_dir, require_contest_state=False, now=now_dt))
    )
    return snap


def probe_rows_1_to_12():
    out = {}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # Row 1: covered legacy + tags + corrupt + DEEP recursion limit + non-int caller season
        d1 = root / "r1"
        d1.mkdir()
        (d1 / "streak.json").write_bytes(LEGACY)
        out["row1_legacy_dateless"] = {
            "streak": call_capture(p.load_streak, d1),
            "saver": call_capture(p.load_saver_available, d1),
            "applied": call_capture(p.load_applied_dates, d1),
        }
        if MODE == "HEAD":
            out["row1_legacy_s2027"] = {
                "streak": call_capture(p.load_streak, d1, season=2027),
                "saver": call_capture(p.load_saver_available, d1, season=2027),
                "applied": call_capture(p.load_applied_dates, d1, season=2027),
                "bytes_unchanged": (d1 / "streak.json").read_bytes() == LEGACY,
            }
            # DEEP nested array (non-object JSON -> does not match 2027)
            d1_deep = root / "r1_deep"
            d1_deep.mkdir()
            (d1_deep / "streak.json").write_bytes(DEEP)
            out["row1_deep_nested_s2027"] = {
                "streak": call_capture(p.load_streak, d1_deep, season=2027),
                "saver": call_capture(p.load_saver_available, d1_deep, season=2027),
                "applied": call_capture(p.load_applied_dates, d1_deep, season=2027),
            }
            # Non-int caller season arg: float 2027.0 and bool True
            d1_arg = root / "r1_arg"
            d1_arg.mkdir()
            (d1_arg / "streak.json").write_text(json.dumps({"season": 2027, "streak": 13, "saver_available": False}))
            d1_bool = root / "r1_bool"
            d1_bool.mkdir()
            (d1_bool / "streak.json").write_text(json.dumps({"season": 1, "streak": 9, "saver_available": False}))
            out["row1_caller_non_int_season"] = {
                "float_2027_0_against_int_2027": call_capture(p.load_streak, d1_arg, season=2027.0),
                "bool_true_against_int_1": call_capture(p.load_streak, d1_bool, season=True),
            }

        # Row 2: save_streak mismatch reset, dateless tag preservation, DEEP regression, non-int season write
        d2 = root / "r2"
        d2.mkdir()
        (d2 / "streak.json").write_bytes(LEGACY)
        if MODE == "HEAD":
            p.save_streak(4, d2, applied_date="2027-03-24", season=2027)
            out["row2_dated_save_on_legacy"] = json.loads((d2 / "streak.json").read_text())
            p.save_streak(9, d2)
            out["row2_dateless_save_preserves_tag"] = json.loads((d2 / "streak.json").read_text())
            # Tagless dateless save keeps tagless
            d2_tl = root / "r2_tl"
            d2_tl.mkdir()
            (d2_tl / "streak.json").write_bytes(LEGACY)
            p.save_streak(9, d2_tl)
            out["row2_dateless_save_keeps_tagless"] = "season" in json.loads((d2_tl / "streak.json").read_text())
            # Non-int season passed to save_streak writes poison tag that fails next read!
            d2_poison = root / "r2_poison"
            d2_poison.mkdir()
            p.save_streak(7, d2_poison, season=2027.0)
            out["row2_save_float_season_roundtrip"] = {
                "written_season": json.loads((d2_poison / "streak.json").read_text())["season"],
                "read_back_2027": p.load_streak(d2_poison, season=2027),
                "read_back_2027_0": p.load_streak(d2_poison, season=2027.0),
            }
        else:
            p.save_streak(4, d2, applied_date="2027-03-24")
            out["row2_dated_save_on_legacy"] = json.loads((d2 / "streak.json").read_text())

        # Row 2 / Row 5 DEEP nested JSON repair vs crash!
        d2_deep = root / "r2_deep"
        d2_deep.mkdir()
        (d2_deep / "streak.json").write_bytes(DEEP)
        if MODE == "HEAD":
            out["row2_save_explicit_saver_on_deep_s2027"] = call_capture(
                p.save_streak, 4, d2_deep, True, season=2027
            )
        else:
            out["row2_save_explicit_saver_on_deep_base"] = call_capture(
                p.save_streak, 4, d2_deep, True
            )

        # Row 3: update_streak on LEGACY + invalid dates + datetime.date object
        d3 = root / "r3"
        d3.mkdir()
        (d3 / "streak.json").write_bytes(LEGACY)
        out["row3_hit_on_legacy"] = {
            "ret": p.update_streak([True], d3, date="2027-03-24"),
            "disk": json.loads((d3 / "streak.json").read_text()),
        }
        d3_miss = root / "r3_miss"
        d3_miss.mkdir()
        (d3_miss / "streak.json").write_text(json.dumps({"streak": 12, "saver_available": True}))
        out["row3_miss_on_tagless_12"] = {
            "ret": p.update_streak([False], d3_miss, date="2027-03-24"),
            "disk": json.loads((d3_miss / "streak.json").read_text()),
        }
        d3_bad = root / "r3_bad"
        d3_bad.mkdir()
        out["row3_bad_dates"] = {
            "malformed_str": call_capture(p.update_streak, [True], d3_bad, date="not-a-date"),
            "leap_invalid": call_capture(p.update_streak, [True], d3_bad, date="2027-02-29"),
            "compact_iso": call_capture(p.update_streak, [True], d3_bad, date="20270324"),
            "unpadded": call_capture(p.update_streak, [True], d3_bad, date="2027-3-24"),
            "with_time": call_capture(p.update_streak, [True], d3_bad, date="2027-03-24T12:00:00"),
            "trailing_newline": call_capture(p.update_streak, [True], d3_bad, date="2027-03-24\n"),
            "arabic_digits": call_capture(p.update_streak, [True], d3_bad, date="٢٠٢٧-٠٣-٢٤"),
            "date_object": call_capture(p.update_streak, [True], d3_bad, date=dt.date(2027, 3, 24)),
            "dir_entries_after": [x.name for x in d3_bad.iterdir()],
        }

        # Row 4: dateless update_streak
        d4 = root / "r4"
        d4.mkdir()
        (d4 / "streak.json").write_bytes(LEGACY)
        out["row4_dateless_on_legacy"] = {
            "ret": p.update_streak([True], d4),
            "disk": json.loads((d4 / "streak.json").read_text()),
        }

        # Row 5: reconcile_results replay + pp1 B1 + DEEP regression + Jan 1 empty-season stamp
        d5 = root / "r5"
        d5.mkdir()
        (d5 / "streak.json").write_bytes(LEGACY)
        plant_scoreable_daily(d5, "2027-03-24", result="hit", slot_results={"pick": "hit"})
        with patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=True):
            p.reconcile_results(d5, clock=lambda: dt.datetime(2027, 3, 25, 2, 0, tzinfo=ET))
        out["row5_reconcile_2027"] = json.loads((d5 / "streak.json").read_text())

        d5_deep = root / "r5_deep"
        d5_deep.mkdir()
        (d5_deep / "streak.json").write_bytes(DEEP)
        plant_scoreable_daily(d5_deep, "2027-03-24", result="hit", slot_results={"pick": "hit"})
        with patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=True):
            out["row5_reconcile_on_deep"] = call_capture(
                p.reconcile_results, d5_deep, clock=lambda: dt.datetime(2027, 3, 25, 2, 0, tzinfo=ET)
            )

        d5_jan1 = root / "r5_jan1"
        d5_jan1.mkdir()
        (d5_jan1 / "streak.json").write_bytes(LEGACY)
        p.reconcile_results(d5_jan1, clock=lambda: dt.datetime(2027, 1, 1, 2, 0, tzinfo=ET))
        out["row5_jan1_idle_reconcile"] = json.loads((d5_jan1 / "streak.json").read_text())

        # Row 6: cli void display + scheduler midgame/final peer & void display + compact ISO edge
        d6 = root / "r6"
        d6.mkdir()
        (d6 / "streak.json").write_bytes(LEGACY)
        plant_scoreable_daily(d6, "2027-03-24")
        with patch.object(p, "resolve_daily_slot_results", return_value={"pick": "void"}):
            res6 = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(d6)])
        out["row6_cli_void"] = {"exit": res6.exit_code, "output": res6.output.strip()}

        # Compact ISO "20270324" divergence between void load_streak (fromisoformat) and hit update_streak (regex)
        d6_compact = root / "r6_compact"
        d6_compact.mkdir()
        (d6_compact / "streak.json").write_bytes(LEGACY)
        plant_scoreable_daily(d6_compact, "20270324")
        with patch.object(p, "resolve_daily_slot_results", return_value={"pick": "void"}), \
             patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 24, 23, 0, tzinfo=ET)):
            res6_c_void = CliRunner().invoke(
                cli, ["check-results", "--date", "20270324", "--allow-stale-scoring", "--picks-dir", str(d6_compact)]
            )
        d6_compact_hit = root / "r6_compact_hit"
        d6_compact_hit.mkdir()
        (d6_compact_hit / "streak.json").write_bytes(LEGACY)
        plant_scoreable_daily(d6_compact_hit, "20270324")
        with patch.object(p, "resolve_daily_slot_results", return_value={"pick": "hit"}), \
             patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 24, 23, 0, tzinfo=ET)):
            res6_c_hit = CliRunner().invoke(
                cli, ["check-results", "--date", "20270324", "--allow-stale-scoring", "--picks-dir", str(d6_compact_hit)]
            )
        out["row6_compact_iso_divergence"] = {
            "void_exit": res6_c_void.exit_code,
            "void_out": res6_c_void.output.strip(),
            "hit_exit": res6_c_hit.exit_code,
            "hit_exc": type(res6_c_hit.exception).__name__ if res6_c_hit.exception else None,
        }

        # Row 7: cli reconcile at UTC/ET New Year boundary AND ticking clock across midnight ET!
        from bts.reconcile_receipt import ReconcileReceipt
        d7 = root / "r7"
        d7.mkdir()
        (d7 / "streak.json").write_text(json.dumps({"season": 2026, "streak": 13, "saver_available": False}))
        # Unresolved 2027-01-01 not needed if we plant unresolved 2026-12-30 so 2026 replay returns None
        plant_scoreable_daily(d7, "2026-12-30", result=None)
        # 1) 2027-01-01T01:00Z = 2026-12-31 20:00 ET (still 2026 in ET!)
        t_ny_eve_utc = dt.datetime(2027, 1, 1, 1, 0, tzinfo=dt.timezone.utc)
        with patch("bts.reconcile_receipt.ReconcileReceipt",
                   side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_ny_eve_utc, lookback_days=kw["lookback_days"])), \
             patch.object(p, "resolve_daily_slot_results", return_value=None):
            res7_2026 = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(d7)])
        # 2) 2027-01-01T05:05Z = 2027-01-01 00:05 ET, with an unresolved 2027-01-01 in applied_dates so replay=None
        d7b = root / "r7b"
        d7b.mkdir()
        (d7b / "streak.json").write_text(json.dumps({
            "season": 2026, "streak": 13, "saver_available": False, "applied_dates": ["2027-01-01"]
        }))
        plant_scoreable_daily(d7b, "2027-01-01", result=None)
        t_ny_day_utc = dt.datetime(2027, 1, 1, 5, 5, tzinfo=dt.timezone.utc)
        with patch("bts.reconcile_receipt.ReconcileReceipt",
                   side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_ny_day_utc, lookback_days=kw["lookback_days"])), \
             patch.object(p, "resolve_daily_slot_results", return_value=None):
            res7_2027 = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(d7b)])
        # 3) Ticking clock crossing 00:00:00 ET between reconcile_results replay and cli summary line!
        d7_tick = root / "r7_tick"
        d7_tick.mkdir()
        plant_scoreable_daily(d7_tick, "2026-12-30", result="hit", slot_results={"pick": "hit"})
        ticks = iter([
            dt.datetime(2026, 12, 31, 23, 59, 58, tzinfo=ET),  # ReconcileReceipt.__init__ started_at
            dt.datetime(2026, 12, 31, 23, 59, 58, tzinfo=ET),  # reconcile_results start today = now_et().date()
            dt.datetime(2026, 12, 31, 23, 59, 59, tzinfo=ET),  # Phase 1 cutoff check
            dt.datetime(2026, 12, 31, 23, 59, 59, tzinfo=ET),  # Phase 2 cutoff check
            dt.datetime(2026, 12, 31, 23, 59, 59, tzinfo=ET),  # replay_day = now_et().date() -> 2026!
            dt.datetime(2027, 1, 1, 0, 0, 0, tzinfo=ET),       # receipt.finish() finished_at
            dt.datetime(2027, 1, 1, 0, 0, 1, tzinfo=ET),       # cli.py summary now_et().year -> 2027!
        ])
        last_tick = dt.datetime(2027, 1, 1, 0, 0, 1, tzinfo=ET)
        def ticking_clock():
            return next(ticks, last_tick)
        with patch("bts.reconcile_receipt.ReconcileReceipt",
                   side_effect=lambda **kw: ReconcileReceipt(clock=ticking_clock, lookback_days=kw["lookback_days"])), \
             patch.object(p, "resolve_daily_slot_results", return_value={"pick": "hit"}):
            res7_tick = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(d7_tick)])
        out["row7_boundary"] = {
            "utc_jan1_0100_et_dec31": res7_2026.output.strip(),
            "utc_jan1_0505_et_jan1_replay_none": res7_2027.output.strip(),
            "ticking_across_midnight_et_output": res7_tick.output.strip(),
            "ticking_across_midnight_et_saved": json.loads((d7_tick / "streak.json").read_text()),
        }

        # Row 8: shadow selection
        d8 = root / "r8"
        d8.mkdir()
        (d8 / "streak.json").write_bytes(LEGACY)
        seen_shadow = []
        with patch.object(sched_mod, "predict_local_shadow", return_value=object()), \
             patch.object(sched_mod, "heartbeat_watchdog", return_value=nullcontext()), \
             patch.object(sched_mod, "_analytics_job_status", return_value={}), \
             patch.object(sched_mod, "_update_analytics_job_status", return_value=None), \
             patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(sched_mod, "select_pick",
                          side_effect=lambda *a, **kw: (seen_shadow.append(kw["streak"]), SimpleNamespace(pick_result=None))[1]):
            sched_mod._run_shadow_prediction({"orchestrator": {"picks_dir": str(d8)}}, "2027-03-24", "Prod")
        out["row8_shadow_streak"] = seen_shadow

        # Row 9: strategy.select_pick fallback + unpadded date "2027-3-24" change
        from tests.test_strategy import _predictions
        d9 = root / "r9"
        d9.mkdir()
        (d9 / "streak.json").write_bytes(LEGACY)
        preds = _predictions([{"p_game_hit": 0.85}])
        seen_saver = []
        orig_resolve = strat_mod.resolve_policy_decision
        def wrap_resolve(*args, **kwargs):
            seen_saver.append(args[2])
            return orig_resolve(*args, **kwargs)
        with patch.object(strat_mod, "get_game_statuses", return_value={778899: "P"}), \
             patch.object(strat_mod, "_load_mdp", return_value=None), \
             patch.object(strat_mod, "resolve_policy_decision", side_effect=wrap_resolve):
            strat_mod.select_pick(preds, "2027-03-24", d9, streak=12)
            unpadded_res = call_capture(strat_mod.select_pick, preds, "2027-3-24", d9, streak=12)
            explicit_saver_unpadded = call_capture(
                strat_mod.select_pick, preds, "2027-3-24", d9, streak=12, saver_available=True
            )
        out["row9_saver_and_unpadded"] = {
            "saver_for_2027_03_24": seen_saver[0],
            "unpadded_2027_3_24_saver_none": {k: v for k, v in unpadded_res.items() if k != "value"},
            "unpadded_2027_3_24_saver_explicit": {k: v for k, v in explicit_saver_unpadded.items() if k != "value"},
        }

        # Row 10: contest_state model diagnostics + corrupt preservation
        d10 = root / "r10"
        d10.mkdir()
        (d10 / "streak.json").write_bytes(LEGACY)
        st10_no_obs = load_decision_streak_state(
            d10, require_contest_state=False, now=dt.datetime(2027, 3, 24, 15, 0, tzinfo=ET)
        )
        sd10 = d10 / "account_state"
        sd10.mkdir()
        (sd10 / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "active_streak": 7, "best_streak": 7,
            "source": "mlb_bts_profile", "source_date": "2027-03-24",
        }))
        st10_obs = load_decision_streak_state(
            d10, require_contest_state=True, now=dt.datetime(2027, 3, 24, 15, 0, tzinfo=ET)
        )
        (d10 / "streak.json").write_bytes(b"{")
        st10_corrupt = load_decision_streak_state(
            d10, require_contest_state=True, now=dt.datetime(2027, 3, 24, 15, 0, tzinfo=ET)
        )
        out["row10"] = {
            "model_only_2027_on_legacy": (st10_no_obs.streak, st10_no_obs.model_streak, st10_no_obs.model_saver_available),
            "with_obs_2027_on_legacy": (st10_obs.streak, st10_obs.model_streak, st10_obs.model_saver_available),
            "corrupt_still_errors": st10_corrupt.model_state_error is not None,
        }

        # Row 12: untouched readers on Opening Night with 2026 streak.json + regenerate tagless write
        d12 = root / "r12"
        d12.mkdir()
        (d12 / "streak.json").write_text(json.dumps({
            "season": 2026, "streak": 13, "saver_available": False,
            "updated": "2026-09-27T05:00:00+00:00", "applied_dates": ["2026-09-27"],
        }))
        with patch.object(web_mod, "PICKS_DIR", d12):
            web_fallback_streak = web_mod.load_streak()
        # Check health validation on invalid season type e.g. "season": "2027"
        d12_bad_tag = root / "r12_bad_tag"
        d12_bad_tag.mkdir()
        (d12_bad_tag / "streak.json").write_text(json.dumps({
            "season": "2027", "streak": 5, "saver_available": True,
            "updated": "2027-03-24T05:00:00+00:00",
        }))
        out["row12"] = {
            "web_fallback_on_2026_file": web_fallback_streak,
            "health_check_on_2026_file": [a.message for a in check_streak_health(d12)],
            "health_check_on_string_season": [a.message for a in check_streak_health(d12_bad_tag)],
            "export_on_2026_file": export_initial_state(d12, root / "exp.json"),
        }
    return out


def probe_boundary_sequences():
    """Item 2: Full sequences across 2026 -> 2027 on one directory using real scorers."""
    out = {}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # Sequence A: 2026 end -> idle winter -> Opening Night 2027 hit -> racing scorer -> Day 2 double crash+restart -> 02:00 reconcile
        seq_a = root / "seq_a"
        seq_a.mkdir()
        steps = []
        # Seed 2026 prior state at streak 12, saver consumed False, then score final 2026 day (2026-09-27) via CliRunner check-results
        if MODE == "HEAD":
            p.save_streak(12, seq_a, saver_available=False, applied_date="2026-09-26", season=2026)
        else:
            p.save_streak(12, seq_a, saver_available=False, applied_date="2026-09-26")
        plant_scoreable_daily(seq_a, "2026-09-27")
        with patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=True), \
             patch.object(cli_mod, "_now_et", return_value=dt.datetime(2026, 9, 28, 1, 0, tzinfo=ET)):
            r_2026 = CliRunner().invoke(cli, ["check-results", "--date", "2026-09-27", "--picks-dir", str(seq_a)])
        steps.append(("1_after_2026_last_day", r_2026.output.strip(), readers_snapshot(seq_a, "2026-09-27", dt.datetime(2026, 9, 28, 12, 0, tzinfo=ET))))

        # Step 2: Idle winter -> Opening Night 2027 pre-game readers (before any 2027 write)
        steps.append(("2_opening_night_pregame_idle_winter", None, readers_snapshot(seq_a, "2027-03-24", dt.datetime(2027, 3, 24, 18, 0, tzinfo=ET))))

        # Step 3: Opening Night 2027 HIT via scheduler.run_result_polling (final)
        plant_scoreable_daily(seq_a, "2027-03-24")
        replies = []
        with patch.object(sched_mod, "_now_et", return_value=dt.datetime(2027, 3, 24, 22, 30, tzinfo=ET)), \
             patch.object(sched_mod, "poll_game_result", return_value="final"), \
             patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=True), \
             patch("bts.posting.format_result_reply", side_effect=lambda res, st: (replies.append((res, st)), f"{res}:{st}")[1]), \
             patch("bts.posting.reply_to_bluesky", return_value="at://reply/1"), \
             redirect_stderr(io.StringIO()) as err3:
            st_poll = sched_mod.run_result_polling(778899, "2027-03-24", seq_a)
        steps.append(("3_opening_night_hit_via_polling", {"status": st_poll, "replies": replies, "stderr": err3.getvalue().strip()}, readers_snapshot(seq_a, "2027-03-24")))

        # Step 4: Racing / second scorer (01:00 check-results) on same date 2027-03-24
        with patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=True), \
             patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
            r_dup = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(seq_a)])
        steps.append(("4_second_scorer_same_date", r_dup.output.strip(), readers_snapshot(seq_a, "2027-03-24")))

        # Step 5: Day two (2027-03-25) DOUBLE hit with crash between update_streak and save_pick, then restart!
        plant_scoreable_daily(seq_a, "2027-03-25", double=True)
        real_save_pick = p.save_pick
        class CrashAfterStreak(BaseException):
            pass
        def crash_once(daily, pd):
            raise CrashAfterStreak("killed between streak.json and pick save")
        with patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=True), \
             patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 26, 1, 0, tzinfo=ET)), \
             patch.object(p, "save_pick", side_effect=crash_once):
            try:
                CliRunner().invoke(cli, ["check-results", "--date", "2027-03-25", "--picks-dir", str(seq_a)], catch_exceptions=False)
            except CrashAfterStreak:
                pass
        steps.append(("5a_day2_after_crash_before_pick_save", {"pick_result": p.load_pick("2027-03-25", seq_a).result}, readers_snapshot(seq_a, "2027-03-25")))

        with patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=True), \
             patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 26, 1, 5, tzinfo=ET)):
            r_day2_retry = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-25", "--picks-dir", str(seq_a)])
        steps.append(("5b_day2_after_restart_check_results", r_day2_retry.output.strip(), readers_snapshot(seq_a, "2027-03-25")))

        # Step 6: 02:00 reconcile via CliRunner
        from bts.reconcile_receipt import ReconcileReceipt
        t_rec = dt.datetime(2027, 3, 26, 2, 0, tzinfo=ET)
        with patch("bts.reconcile_receipt.ReconcileReceipt",
                   side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_rec, lookback_days=kw["lookback_days"])), \
             patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=True):
            r_rec = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(seq_a)])
        steps.append(("6_after_0200_reconcile", r_rec.output.strip(), readers_snapshot(seq_a, "2027-03-26", t_rec)))
        out["seq_a_hit_double_crash_reconcile"] = steps

        # Sequence B: 2026 ends at streak=12, saver_available=True -> Opening Night 2027 MISS!
        seq_b = root / "seq_b"
        seq_b.mkdir()
        if MODE == "HEAD":
            p.save_streak(12, seq_b, saver_available=True, applied_date="2026-09-27", season=2026)
        else:
            p.save_streak(12, seq_b, saver_available=True, applied_date="2026-09-27")
        plant_scoreable_daily(seq_b, "2027-03-24")
        with patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=False), \
             patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
            r_miss = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(seq_b)])
        out["seq_b_opening_night_miss_from_2026_streak_12"] = {
            "cli_output": r_miss.output.strip(),
            "disk": json.loads((seq_b / "streak.json").read_text()),
        }

        # Sequence C: 2026 ends at streak=13, saver=False -> Opening Night 2027 VOID -> Day 2 HIT
        seq_c = root / "seq_c"
        seq_c.mkdir()
        (seq_c / "streak.json").write_bytes(LEGACY)
        plant_scoreable_daily(seq_c, "2027-03-24")
        with patch.object(p, "get_game_statuses_detailed", return_value={778899: {"abstract": "F", "detailed": "Postponed"}}), \
             patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
            r_void = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(seq_c)])
        void_bytes_unchanged = (seq_c / "streak.json").read_bytes() == LEGACY
        plant_scoreable_daily(seq_c, "2027-03-25")
        with patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=True), \
             patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 26, 1, 0, tzinfo=ET)):
            r_d2 = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-25", "--picks-dir", str(seq_c)])
        out["seq_c_void_then_hit"] = {
            "void_output": r_void.output.strip(),
            "void_left_legacy_bytes_untouched": void_bytes_unchanged,
            "day2_hit_output": r_d2.output.strip(),
            "disk_after_day2": json.loads((seq_c / "streak.json").read_text()),
        }
    return out


def probe_corrupt_during_scoring():
    """Item 4: Corrupt streak.json during check-results & run_result_polling, plus health + reconcile after."""
    out = {}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for label, payload in [("invalid_json", b"{"), ("invalid_utf8", b"\xff"), ("json_array", b"[]"), ("deep_array", DEEP)]:
            case_res = {}
            # 1) check-results on a HIT when real mid-season history had 12 hits and spent saver!
            d_hit = root / f"{label}_hit"
            d_hit.mkdir()
            # Plant 12 resolved hits EXCEPT 2027-03-10 is unresolved (simulating incomplete early history)
            plant_scoreable_daily(d_hit, "2027-03-10", result=None)
            plant_scoreable_daily(d_hit, "2027-03-24", result=None)
            (d_hit / "streak.json").write_bytes(payload)
            with patch.object(p, "get_game_statuses_detailed", return_value={}), \
                 patch.object(p, "check_hit", return_value=True), \
                 patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
                res = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(d_hit)])
            health_after = call_capture(lambda: [a.message for a in check_streak_health(d_hit)])
            cs_after = call_capture(
                lambda: load_decision_streak_state(
                    d_hit, require_contest_state=False, now=dt.datetime(2027, 3, 25, 9, 0, tzinfo=ET)
                ).model_state_error
            )
            # Run 02:00 reconcile when 2027-03-10 is unresolved (outside 8-day lookback!)
            with patch.object(p, "get_game_statuses_detailed", return_value={}), \
                 patch.object(p, "check_hit", return_value=True):
                rec_incomplete = call_capture(
                    p.reconcile_results, d_hit, clock=lambda: dt.datetime(2027, 3, 25, 2, 0, tzinfo=ET)
                )
            after_bytes = None
            try:
                parsed_after = json.loads((d_hit / "streak.json").read_text())
                after_bytes = parsed_after if isinstance(parsed_after, dict) else f"<non_dict:{type(parsed_after).__name__}>"
            except BaseException as exc:
                after_bytes = f"<still_corrupt:{type(exc).__name__}>"
            case_res["check_results_hit"] = {
                "exit_code": res.exit_code,
                "exc": type(res.exception).__name__ if res.exception else None,
                "pick_result_saved": p.load_pick("2027-03-24", d_hit).result,
                "health_alerts_after": health_after,
                "contest_state_model_error_after": cs_after,
                "reconcile_with_incomplete_history": rec_incomplete,
                "streak_json_after_reconcile": after_bytes,
            }

            # 2) check-results on VOID with corrupt streak.json
            d_void = root / f"{label}_void"
            d_void.mkdir()
            plant_scoreable_daily(d_void, "2027-03-24", result=None)
            (d_void / "streak.json").write_bytes(payload)
            with patch.object(p, "get_game_statuses_detailed", return_value={778899: {"abstract": "F", "detailed": "Postponed"}}), \
                 patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
                res_v = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(d_void)])
            case_res["check_results_void"] = {
                "exit_code": res_v.exit_code,
                "exc": type(res_v.exception).__name__ if res_v.exception else None,
                "bytes_still_corrupt": (d_void / "streak.json").read_bytes() == payload,
                "health_alerts_after": call_capture(lambda: [a.message for a in check_streak_health(d_void)]),
            }

            # 3) scheduler.run_result_polling on HIT (with Bluesky reply capture)
            d_poll = root / f"{label}_poll"
            d_poll.mkdir()
            plant_scoreable_daily(d_poll, "2027-03-24", result=None)
            (d_poll / "streak.json").write_bytes(payload)
            replies = []
            with patch.object(sched_mod, "_now_et", return_value=dt.datetime(2027, 3, 24, 22, 30, tzinfo=ET)), \
                 patch.object(sched_mod, "poll_game_result", return_value="final"), \
                 patch.object(p, "get_game_statuses_detailed", return_value={}), \
                 patch.object(p, "check_hit", return_value=True), \
                 patch("bts.posting.format_result_reply", side_effect=lambda r, s: (replies.append((r, s)), f"{r}:{s}")[1]), \
                 patch("bts.posting.reply_to_bluesky", return_value="at://reply/1"), \
                 redirect_stderr(io.StringIO()):
                poll_res = call_capture(sched_mod.run_result_polling, 778899, "2027-03-24", d_poll)
            case_res["polling_final_hit"] = {
                "result": poll_res,
                "bluesky_replies": replies,
            }
            out[label] = case_res

        # Directory in place of streak.json
        d_dir = root / "is_a_dir"
        d_dir.mkdir()
        (d_dir / "streak.json").mkdir()
        plant_scoreable_daily(d_dir, "2027-03-24", result=None)
        with patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=True), \
             patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
            res_dir = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(d_dir)])
        out["streak_json_is_directory"] = {
            "exit_code": res_dir.exit_code,
            "exc": type(res_dir.exception).__name__ if res_dir.exception else None,
        }
    return out


def probe_mid_season_and_test_timebomb():
    out = {}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Item 3: Saver boundaries 9, 10, 12, 15, 16 on matching season vs base
        saver_cases = {}
        for st in (9, 10, 12, 15, 16):
            dm = root / f"saver_{st}"
            dm.mkdir()
            if MODE == "HEAD":
                p.save_streak(st, dm, saver_available=True, season=2027)
            else:
                p.save_streak(st, dm, saver_available=True)
            after_st = p.update_streak([False], dm, date="2027-05-10")
            saver_cases[str(st)] = (after_st, p.load_saver_available(dm))
        out["saver_boundaries"] = saver_cases

        # Item 3: pp1 B1 on BOTH 2027-tagged and tagless streak.json
        for tag_label, tag_val in [("tagless", None), ("tagged_2027", 2027)]:
            db1 = root / f"b1_{tag_label}"
            db1.mkdir()
            plant_scoreable_daily(db1, "2027-05-09", result="hit", slot_results={"pick": "hit"})
            plant_scoreable_daily(db1, "2027-05-10", result=None)
            body = {"streak": 2, "saver_available": True, "applied_dates": ["2027-05-10"]}
            if tag_val is not None:
                body["season"] = tag_val
            (db1 / "streak.json").write_text(json.dumps(body))
            before_b1 = (db1 / "streak.json").read_bytes()
            with patch.object(p, "get_game_statuses_detailed", return_value={}), \
                 patch.object(p, "check_hit", return_value=None):
                p.reconcile_results(db1, clock=lambda: dt.datetime(2027, 5, 10, 23, 0, tzinfo=ET))
            out[f"pp1_b1_{tag_label}_bytes_unchanged"] = (db1 / "streak.json").read_bytes() == before_b1

        # Item 3: dateless update_streak and save_streak between dated updates on 2027-tagged file
        d_mid = root / "mid_dateless"
        d_mid.mkdir()
        if MODE == "HEAD":
            p.save_streak(5, d_mid, saver_available=True, season=2027)
        else:
            p.save_streak(5, d_mid, saver_available=True)
        s1 = p.update_streak([True], d_mid, date="2027-05-01")
        s2 = p.update_streak([True], d_mid)  # dateless between dated
        s3 = p.update_streak([True], d_mid, date="2027-05-01")  # same date again -> idempotent!
        s4 = p.update_streak([True, True], d_mid, date="2027-05-02")
        out["mid_season_dateless_between_dated"] = {
            "steps": [s1, s2, s3, s4],
            "disk": json.loads((d_mid / "streak.json").read_text()),
        }

        # Item 5: Contest dates on 2026-12-31 and 2027-01-01 in sequence
        d_nye = root / "nye"
        d_nye.mkdir()
        if MODE == "HEAD":
            p.save_streak(11, d_nye, saver_available=False, season=2026)
        else:
            p.save_streak(11, d_nye, saver_available=False)
        nye_hit = p.update_streak([True], d_nye, date="2026-12-31")
        nye_disk = json.loads((d_nye / "streak.json").read_text())
        nyd_hit = p.update_streak([True], d_nye, date="2027-01-01")
        nyd_disk = json.loads((d_nye / "streak.json").read_text())
        out["contest_date_1231_then_0101"] = {
            "after_2026_12_31": {"streak": nye_hit, "disk": nye_disk},
            "after_2027_01_01": {"streak": nyd_hit, "disk": nyd_disk},
        }

        # Item 4 extra: chmod 000 corrupt streak.json during check-results hit
        if os.geteuid() != 0:
            d_perm = root / "perm000"
            d_perm.mkdir()
            plant_scoreable_daily(d_perm, "2027-03-24", result=None)
            sp = d_perm / "streak.json"
            sp.write_text(json.dumps({"season": 2027, "streak": 12, "saver_available": False}))
            sp.chmod(0o000)
            try:
                with patch.object(p, "get_game_statuses_detailed", return_value={}), \
                     patch.object(p, "check_hit", return_value=True), \
                     patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
                    res_perm = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(d_perm)])
                out["chmod000_during_check_results_hit"] = {
                    "exit_code": res_perm.exit_code,
                    "exc": type(res_perm.exception).__name__ if res_perm.exception else None,
                    "after_disk": json.loads((d_perm / "streak.json").read_text()) if res_perm.exit_code == 0 else None,
                }
            finally:
                if sp.exists():
                    sp.chmod(0o644)

        # Item 7: Demonstrate tests/test_contest_state.py time-bomb when wall clock crosses 2027-01-01!
        if MODE == "HEAD":
            import tests.test_contest_state as tcs
            real_dt = cs_mod.datetime
            class FakeDateTime2027(dt.datetime):
                @classmethod
                def now(cls, tz=None):
                    return dt.datetime(2027, 1, 2, 15, 0, tzinfo=tz or dt.timezone.utc)
            results_2027 = {}
            for fn_name in [
                "test_model_state_used_when_no_contest_state",
                "test_fresh_contest_state_drives_live_decision",
                "test_lagged_contest_uses_contest_value_keeps_doubles",
                "test_lagged_contest_model_does_not_inflate",
                "test_model_never_inflates_decision_streak",
            ]:
                sub = root / fn_name
                sub.mkdir()
                with patch.object(cs_mod, "datetime", FakeDateTime2027):
                    results_2027[fn_name] = call_capture(getattr(tcs, fn_name), sub)
            out["test_contest_state_in_year_2027"] = results_2027
    return out


print(json.dumps({
    "mode": MODE,
    "bts_file": bts.__file__,
    "rows_1_to_12": probe_rows_1_to_12(),
    "sequences": probe_boundary_sequences(),
    "corrupt_scoring": probe_corrupt_during_scoring(),
    "mid_season_and_tests": probe_mid_season_and_test_timebomb(),
}, indent=2, default=str))
'''


def run_worker(mode: str, src_dir: Path) -> dict:
    env = os.environ.copy()
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = f"{src_dir}:{ROOT}"
    proc = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-c", WORKER_CODE, mode],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(f"Worker {mode} failed (rc={proc.returncode}):\n{proc.stderr}\n")
        sys.exit(proc.returncode)
    return json.loads(proc.stdout)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cs2_base_d8eb50d0_") as tmp_base:
        archive = subprocess.run(
            ["git", "archive", "d8eb50d0", "src"],
            cwd=str(ROOT),
            capture_output=True,
            check=True,
        )
        subprocess.run(["tar", "-x", "-C", tmp_base], input=archive.stdout, check=True)
        base = run_worker("BASE", Path(tmp_base) / "src")
    head = run_worker("HEAD", HEAD_SRC)
    combined = {"base_d8eb50d0": base, "head_3b4196be": head}
    LOG_PATH.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {LOG_PATH} ({LOG_PATH.stat().st_size} bytes)")
    print("BASE bts.__file__:", base["bts_file"])
    print("HEAD bts.__file__:", head["bts_file"])
    print("\n--- KEY DIFFERENTIAL HIGHLIGHTS ---")
    print("Row 1 DEEP nested s=2027 (HEAD):", head["rows_1_to_12"]["row1_deep_nested_s2027"])
    print("Row 1 caller non-int season (HEAD):", head["rows_1_to_12"]["row1_caller_non_int_season"])
    print("Row 2 float season roundtrip (HEAD):", head["rows_1_to_12"]["row2_save_float_season_roundtrip"])
    print("Row 2 DEEP save explicit saver BASE vs HEAD:",
          base["rows_1_to_12"]["row2_save_explicit_saver_on_deep_base"],
          head["rows_1_to_12"]["row2_save_explicit_saver_on_deep_s2027"])
    print("Row 5 DEEP reconcile BASE vs HEAD:",
          base["rows_1_to_12"]["row5_reconcile_on_deep"],
          head["rows_1_to_12"]["row5_reconcile_on_deep"])
    print("Row 3 bad dates (HEAD):", head["rows_1_to_12"]["row3_bad_dates"])
    print("Row 6 compact ISO divergence (HEAD):", head["rows_1_to_12"]["row6_compact_iso_divergence"])
    print("Row 7 boundary + midnight tick (HEAD):", head["rows_1_to_12"]["row7_boundary"])
    print("Row 9 saver + unpadded date BASE vs HEAD:",
          base["rows_1_to_12"]["row9_saver_and_unpadded"],
          head["rows_1_to_12"]["row9_saver_and_unpadded"])
    print("Seq B (Opening Night miss from 2026 streak 12) BASE vs HEAD:",
          base["sequences"]["seq_b_opening_night_miss_from_2026_streak_12"],
          head["sequences"]["seq_b_opening_night_miss_from_2026_streak_12"])
    print("Corrupt invalid_json hit BASE vs HEAD:",
          base["corrupt_scoring"]["invalid_json"]["check_results_hit"],
          head["corrupt_scoring"]["invalid_json"]["check_results_hit"])
    print("test_contest_state in 2027 (HEAD):", head["mid_season_and_tests"]["test_contest_state_in_year_2027"])


if __name__ == "__main__":
    main()
