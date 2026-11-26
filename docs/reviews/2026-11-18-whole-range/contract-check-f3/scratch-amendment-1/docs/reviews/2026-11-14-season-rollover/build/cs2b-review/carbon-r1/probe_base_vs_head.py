#!/usr/bin/env python3
"""Independent adversarial comparison probe: base (d8eb50d0) vs HEAD (dd915ac9).

Covers:
1. Suite 1: Contract Rows A through K (exact cases + near-row boundary probes)
2. Suite 2: Row L exhaustive 59-input x 11-path base-vs-head comparison matrix
            + direct season_of_date torture tests (types, 50M-char strings, etc.)
3. Suite 3: Corrupt state 10-payload x 12-path base-vs-head comparison matrix
"""
from __future__ import annotations

from contextlib import nullcontext
import datetime as dt
from enum import IntEnum
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "base_src" / "src"
HEAD_SRC = REPO_ROOT / "src"

ET = ZoneInfo("America/New_York")
FIXED_NOW_ET = dt.datetime(2027, 3, 24, 23, 0, tzinfo=ET)
FIXED_NOW_UTC = FIXED_NOW_ET.astimezone(dt.timezone.utc)

LEGACY_UNTAGGED = (
    b'{"streak": 13, "saver_available": false, '
    b'"updated": "2026-09-13T05:01:02.123456+00:00", '
    b'"applied_dates": ["2026-09-11", "2026-09-12"]}'
)
TAGGED_2026 = (
    b'{"streak": 13, "saver_available": false, '
    b'"updated": "2026-09-13T05:01:02.123456+00:00", '
    b'"applied_dates": ["2026-09-11", "2026-09-12"], "season": 2026}'
)
DEEP_JSON = b"[" * 200_000 + b"]" * 200_000
HUGE_INT_JSON = b'{"season": 2027, "streak": ' + (b"9" * 5000) + b"}"


class DateStr(str):
    """Plain str subclass."""
    tag = "custom"


class RaisingSliceStr(str):
    """str subclass whose Python slicing/__getitem__ raises RuntimeError."""
    def __getitem__(self, item):
        raise RuntimeError("slicing forbidden on RaisingSliceStr")


class SeasonEnum(IntEnum):
    S2027 = 2027


def build_date_cases() -> list[tuple[str, object]]:
    return [
        ("padded", "2027-03-24"),
        ("unpadded_month", "2027-3-24"),
        ("unpadded_day", "2027-03-4"),
        ("unpadded_both", "2027-3-4"),
        ("compact", "20270324"),
        ("week_hyphen", "2027-W12-3"),
        ("week_compact", "2027W123"),
        ("week_cross_fwd_2027", "2026-W53-5"),     # 2027-01-01 -> calendar year 2027
        ("week_cross_bwd_2024", "2025-W01-1"),     # 2024-12-30 -> calendar year 2024
        ("ordinal_hyphen", "2027-083"),
        ("ordinal_compact", "2027083"),
        ("year_month_only", "2027-03"),
        ("leading_space", " 2027-03-24"),
        ("trailing_space", "2027-03-24 "),
        ("trailing_newline", "2027-03-24\n"),
        ("surrounding_tab", "\t2027-03-24\t"),
        ("datetime_T", "2027-03-24T10:00"),
        ("datetime_space", "2027-03-24 00:00:00"),
        ("date_Z", "2027-03-24Z"),
        ("datetime_offset", "2027-03-24T00:00:00-04:00"),
        ("slash_sep", "2027/03/24"),
        ("dot_sep", "2027.03.24"),
        ("underscore_sep", "2027_03_24"),
        ("nonleap_feb29", "2027-02-29"),
        ("feb30", "2027-02-30"),
        ("apr31", "2027-04-31"),
        ("month13", "2027-13-01"),
        ("week54", "2027-W54-1"),
        ("leap_2028_feb29", "2028-02-29"),
        ("year_0000", "0000-03-24"),
        ("year_0001", "0001-03-24"),
        ("year_3digit_999", "999-03-24"),
        ("year_9999", "9999-03-24"),
        ("year_10000", "10000-03-24"),
        ("negative_year", "-2027-03-24"),
        ("plus_year", "+2027-03-24"),
        ("non_ascii_all_arabic", "٢٠٢٧-٠٣-٢٤"),
        ("non_ascii_all_fullwidth", "２０２７-０３-２４"),
        ("non_ascii_year_only", "٢٠٢٧-03-24"),
        ("non_ascii_day_2nd_digit", "2027-03-2٤"),
        ("embedded_nul", "2027-03-24\x00"),
        ("lone_surrogate", "2027-03-24\ud800"),
        ("long_200", "2" * 200),
        ("long_10000", "2" * 10_000),
        ("empty", ""),
        ("junk", "junk"),
        ("none", None),
        ("date_obj", dt.date(2027, 3, 24)),
        ("datetime_obj", dt.datetime(2027, 3, 24, 12, 0)),
        ("path_obj", Path("2027-03-24")),
        ("bytes_obj", b"2027-03-24"),
        ("bytearray_obj", bytearray(b"2027-03-24")),
        ("int_obj", 20270324),
        ("bool_true", True),
        ("bool_false", False),
        ("float_obj", 2027.0),
        ("str_sub_valid", DateStr("2027-03-24")),
        ("str_sub_junk", DateStr("junk")),
        ("raising_slice_padded", RaisingSliceStr("2027-03-24")),
        ("raising_slice_unpadded", RaisingSliceStr("2027-3-24")),
        ("raising_slice_junk", RaisingSliceStr("junk")),
    ]


def _configure_worker_imports(mode: str) -> Path:
    expected = BASE_SRC if mode == "base" else HEAD_SRC
    unwanted = HEAD_SRC if mode == "base" else BASE_SRC
    cleaned = []
    for entry in sys.path:
        try:
            rp = Path(entry).resolve()
        except Exception:
            cleaned.append(entry)
            continue
        if rp == unwanted.resolve():
            continue
        cleaned.append(entry)
    sys.path[:] = [str(expected.resolve()), str(REPO_ROOT.resolve())] + cleaned
    import bts.picks as p
    actual_file = Path(p.__file__).resolve()
    assert actual_file == (expected / "bts" / "picks.py").resolve(), (
        f"Worker {mode} imported wrong picks.py: {actual_file}"
    )
    return actual_file


def _plant_pick_for_any_date(picks_dir: Path, day_val: object, *, result=None, bluesky_uri="at://did:plc:test/1"):
    """Plant a delivered scoreable pick at {day_val}.json if the OS allows that filename."""
    try:
        target = picks_dir / f"{day_val}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        day_str = day_val if isinstance(day_val, str) else str(day_val)
        payload = {
            "date": day_str,
            "run_time": "2027-03-24T18:00:00+00:00",
            "pick": {
                "batter_name": "Steven Kwan",
                "batter_id": 700363,
                "team": "CLE",
                "lineup_position": 1,
                "pitcher_name": "Gerrit Cole",
                "pitcher_id": 543037,
                "p_game_hit": 0.85,
                "flags": [],
                "projected_lineup": False,
                "game_pk": 778899,
                "game_time": "2027-03-24T23:10:00Z",
                "pitcher_team": "NYY",
            },
            "double_down": None,
            "runner_up": None,
            "bluesky_posted": True,
            "bluesky_uri": bluesky_uri,
            "notification_sent": True,
            "notification_channel": "bluesky_dm",
            "notification_id": "dm-1",
            "delivery_attempted": True,
            "delivered_at": "2027-03-24T18:01:00+00:00",
            "result": result,
            "slot_results": {"pick": result} if result else None,
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target
    except (ValueError, OSError):
        return None


def _read_bytes_safe(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_bytes().hex()
    except PermissionError:
        os.chmod(path, 0o600)
        try:
            return "CHMOD000:" + path.read_bytes().hex()
        finally:
            os.chmod(path, 0o000)
    except OSError as e:
        return f"OSERR:{type(e).__name__}"


def _read_pick_result_safe(picks_dir: Path, day_val: object) -> object:
    try:
        target = picks_dir / f"{day_val}.json"
        if not target.exists():
            return "NO_FILE"
        return json.loads(target.read_text(encoding="utf-8")).get("result")
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def _make_predictions_df():
    import pandas as pd
    return pd.DataFrame([{
        "batter_name": "Steven Kwan",
        "batter_id": 700363,
        "team": "CLE",
        "lineup": 1,
        "pitcher_name": "Gerrit Cole",
        "pitcher_id": 543037,
        "pitcher_team": "NYY",
        "game_pk": 778899,
        "game_time": "2027-03-24T23:10:00Z",
        "p_hit_pa": 0.32,
        "p_game_hit": 0.85,
        "flags": "",
    }])


def run_date_matrix_worker(mode: str, work_root: Path) -> dict:
    import bts.cli as cli_mod
    from bts.cli import cli
    import bts.picks as p
    import bts.scheduler as sched_mod
    import bts.strategy as strat_mod
    from click.testing import CliRunner

    preds_df = _make_predictions_df()
    paths = [
        "shadow_real",
        "poll_midgame_peer",
        "poll_midgame_score",
        "poll_final_peer",
        "poll_final_void",
        "poll_final_score",
        "cli_void_default",
        "cli_void_allow_stale",
        "cli_score_allow_stale",
        "select_pick",
        "update_streak",
    ]

    class _FakeDT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return FIXED_NOW_UTC.astimezone(tz) if tz else FIXED_NOW_UTC.replace(tzinfo=None)

    results = {}
    for case_id, day_val in build_date_cases():
        results[case_id] = {}
        for path_name in paths:
            d = work_root / f"m_{mode}_{case_id}_{path_name}"
            d.mkdir(parents=True, exist_ok=True)
            streak_file = d / "streak.json"
            streak_file.write_bytes(TAGGED_2026)
            before_hex = _read_bytes_safe(streak_file)
            _plant_pick_for_any_date(d, day_val, result=None)

            observed_calls = []
            orig_ls = p.load_streak
            orig_lsa = p.load_saver_available
            orig_lad = p.load_applied_dates
            orig_ss = p.save_streak

            def wrap_ls(pd, *a, **kw):
                val = orig_ls(pd, *a, **kw)
                observed_calls.append(("load_streak", kw.get("season", "NO_ARG"), val))
                return val

            def wrap_lsa(pd, *a, **kw):
                val = orig_lsa(pd, *a, **kw)
                observed_calls.append(("load_saver_available", kw.get("season", "NO_ARG"), val))
                return val

            def wrap_lad(pd, *a, **kw):
                val = orig_lad(pd, *a, **kw)
                observed_calls.append(("load_applied_dates", kw.get("season", "NO_ARG"), list(val)))
                return val

            def wrap_ss(st, pd, *a, **kw):
                observed_calls.append(("save_streak", kw.get("season", "NO_ARG"), st))
                return orig_ss(st, pd, *a, **kw)

            out_buf = io.StringIO()
            err_buf = io.StringIO()
            outcome = None
            exc_info = None

            with patch.object(p, "datetime", _FakeDT), \
                 patch.object(strat_mod, "datetime", _FakeDT), \
                 patch.object(p, "load_streak", side_effect=wrap_ls), \
                 patch.object(p, "load_saver_available", side_effect=wrap_lsa), \
                 patch.object(p, "load_applied_dates", side_effect=wrap_lad), \
                 patch.object(p, "save_streak", side_effect=wrap_ss), \
                 patch.object(strat_mod, "load_saver_available", side_effect=wrap_lsa):
                try:
                    if path_name == "shadow_real":
                        seen_sel = []
                        orig_select = strat_mod.select_pick

                        def spy_select(*sa, **skw):
                            seen_sel.append({"streak": skw.get("streak")})
                            res = orig_select(*sa, **skw)
                            seen_sel[-1]["saver"] = res.saver_available
                            seen_sel[-1]["action"] = res.action
                            return res

                        with patch.object(sched_mod, "_now_et", return_value=FIXED_NOW_ET), \
                             patch.object(sched_mod, "predict_local_shadow", return_value=preds_df), \
                             patch.object(sched_mod, "heartbeat_watchdog", return_value=nullcontext()), \
                             patch.object(sched_mod, "_analytics_job_status", return_value={}), \
                             patch.object(sched_mod, "_update_analytics_job_status", return_value=None), \
                             patch.object(p, "get_game_statuses_detailed",
                                          return_value={778899: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}), \
                             patch.object(strat_mod, "_load_mdp", return_value=None), \
                             patch.object(sched_mod, "select_pick", side_effect=spy_select), \
                             patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                            sched_mod._run_shadow_prediction(
                                {"orchestrator": {"picks_dir": str(d), "models_dir": str(d / "models")}},
                                day_val,
                                "Steven Kwan",
                            )
                        outcome = {"seen_sel": seen_sel}

                    elif path_name in ("poll_midgame_peer", "poll_midgame_score"):
                        def midgame_hook(*_a):
                            if path_name == "poll_midgame_peer":
                                peer = p.load_pick(day_val, d)
                                if peer is not None:
                                    peer.result = "hit"
                                    p.save_pick(peer, d)
                            return [True]

                        polls = iter(["live", "final"])
                        replies = []
                        with patch.object(sched_mod, "_now_et", return_value=FIXED_NOW_ET), \
                             patch.object(sched_mod, "poll_game_result", side_effect=lambda *_a: next(polls)), \
                             patch.object(sched_mod, "_poll_interval_sleep", return_value=None), \
                             patch.object(sched_mod, "_check_hits_midgame", side_effect=midgame_hook), \
                             patch.object(p, "get_game_statuses_detailed", return_value={}), \
                             patch("bts.posting.format_result_reply",
                                   side_effect=lambda r, s: (replies.append((r, s)), f"{r}:{s}")[1]), \
                             patch("bts.posting.reply_to_bluesky", return_value="at://reply/1"), \
                             patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                            ret = sched_mod.run_result_polling(778899, day_val, d)
                        outcome = {"ret": ret, "replies": replies}

                    elif path_name in ("poll_final_peer", "poll_final_void", "poll_final_score"):
                        def resolve_hook(*_a):
                            if path_name == "poll_final_peer":
                                peer = p.load_pick(day_val, d)
                                if peer is not None:
                                    peer.result = "hit"
                                    p.save_pick(peer, d)
                                return {"pick": "hit"}
                            if path_name == "poll_final_void":
                                return {"pick": "void"}
                            return {"pick": "hit"}

                        replies = []
                        with patch.object(sched_mod, "_now_et", return_value=FIXED_NOW_ET), \
                             patch.object(sched_mod, "poll_game_result", return_value="final"), \
                             patch.object(p, "get_game_statuses_detailed", return_value={}), \
                             patch.object(p, "resolve_daily_slot_results", side_effect=resolve_hook), \
                             patch("bts.posting.format_result_reply",
                                   side_effect=lambda r, s: (replies.append((r, s)), f"{r}:{s}")[1]), \
                             patch("bts.posting.reply_to_bluesky", return_value="at://reply/1"), \
                             patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                            ret = sched_mod.run_result_polling(778899, day_val, d)
                        outcome = {"ret": ret, "replies": replies}

                    elif path_name in ("cli_void_default", "cli_void_allow_stale", "cli_score_allow_stale"):
                        slot_res = {"pick": "hit"} if path_name == "cli_score_allow_stale" else {"pick": "void"}
                        allow_stale = path_name != "cli_void_default"
                        with patch.object(cli_mod, "_now_et", return_value=FIXED_NOW_ET), \
                             patch.object(p, "resolve_daily_slot_results", return_value=slot_res):
                            if isinstance(day_val, str) and "\x00" not in day_val:
                                args = ["check-results", "--date", day_val, "--picks-dir", str(d)]
                                if allow_stale:
                                    args.append("--allow-stale-scoring")
                                r = CliRunner().invoke(cli, args)
                                if r.exception is not None and not isinstance(r.exception, SystemExit):
                                    raise r.exception
                                out_buf.write(r.output)
                                outcome = {"exit_code": r.exit_code}
                            else:
                                cli_mod.check_results.callback(
                                    date=day_val,
                                    picks_dir=str(d),
                                    shadow_status_output=None,
                                    wait_deadline_et=None,
                                    wait_interval_min=1,
                                    allow_stale_scoring=allow_stale,
                                )
                                outcome = {"exit_code": 0}

                    elif path_name == "select_pick":
                        with patch.object(strat_mod, "get_game_statuses", return_value={778899: "P"}), \
                             patch.object(strat_mod, "_load_mdp", return_value=None):
                            sel = strat_mod.select_pick(preds_df, day_val, d, streak=13)
                        outcome = {
                            "action": sel.action,
                            "saver": sel.saver_available,
                            "streak": sel.streak,
                        }

                    elif path_name == "update_streak":
                        val = p.update_streak([True], d, date=day_val)
                        outcome = {"ret": val}

                except BaseException as exc:
                    exc_info = {"type": type(exc).__name__, "msg": str(exc)}

            after_hex = _read_bytes_safe(streak_file)
            results[case_id][path_name] = {
                "outcome": outcome,
                "exc": exc_info,
                "stdout": out_buf.getvalue(),
                "stderr": err_buf.getvalue(),
                "before_hex": before_hex,
                "after_hex": after_hex,
                "pick_result": _read_pick_result_safe(d, day_val),
                "calls": observed_calls,
            }
    return results


def run_corrupt_matrix_worker(mode: str, work_root: Path) -> dict:
    import bts.cli as cli_mod
    from bts.cli import cli
    from bts.contest_state import load_decision_streak_state
    import bts.picks as p
    from bts.reconcile_receipt import ReconcileReceipt
    import bts.scheduler as sched_mod
    from click.testing import CliRunner

    payloads = [
        ("brace", b"{"),
        ("non_utf8", b"\xff\xfe"),
        ("empty_bytes", b""),
        ("json_list", b"[]"),
        ("json_null", b"null"),
        ("json_int", b"42"),
        ("json_str", b'"hello"'),
        ("deep_nested", DEEP_JSON),
        ("huge_int", HUGE_INT_JSON),
        ("chmod_000", b'{"season": 2027, "streak": 5, "saver_available": true}'),
        ("is_directory", None),
    ]
    branches = [
        "cli_hit",
        "cli_miss",
        "cli_void",
        "poll_midgame_score",
        "poll_midgame_peer",
        "poll_final_hit",
        "poll_final_void",
        "poll_final_peer",
        "reconcile_incomplete",
        "reconcile_complete",
        "cli_reconcile_incomplete",
        "decision_model_only",
        "decision_with_contest",
    ]

    out = {}
    for pay_id, raw in payloads:
        out[pay_id] = {}
        for br in branches:
            d = work_root / f"c_{mode}_{pay_id}_{br}"
            d.mkdir(parents=True, exist_ok=True)
            sp = d / "streak.json"
            if pay_id == "is_directory":
                sp.mkdir()
            else:
                sp.write_bytes(raw)
                if pay_id == "chmod_000":
                    os.chmod(sp, 0o000)

            before_state = "DIR" if pay_id == "is_directory" else _read_bytes_safe(sp)
            day = "2027-03-24"
            exc_info = None
            ret_val = None

            try:
                if br in ("cli_hit", "cli_miss", "cli_void"):
                    _plant_pick_for_any_date(d, day, result=None)
                    sr = {"pick": br.split("_")[1]}
                    with patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)), \
                         patch.object(p, "resolve_daily_slot_results", return_value=sr):
                        r = CliRunner().invoke(cli, ["check-results", "--date", day, "--picks-dir", str(d)])
                        if r.exception is not None and not isinstance(r.exception, SystemExit):
                            raise r.exception
                        ret_val = {"exit_code": r.exit_code, "output": r.output}

                elif br in ("poll_midgame_score", "poll_midgame_peer"):
                    _plant_pick_for_any_date(
                        d, day, result=("hit" if br == "poll_midgame_peer" else None)
                    )
                    # If peer, only mark resolved inside midgame hook so loop enters midgame
                    if br == "poll_midgame_peer":
                        _plant_pick_for_any_date(d, day, result=None)

                    def mg_hook(*_a):
                        if br == "poll_midgame_peer":
                            peer = p.load_pick(day, d)
                            peer.result = "hit"
                            p.save_pick(peer, d)
                        return [True]

                    polls = iter(["live", "final"])
                    with patch.object(sched_mod, "_now_et", return_value=FIXED_NOW_ET), \
                         patch.object(sched_mod, "poll_game_result", side_effect=lambda *_a: next(polls)), \
                         patch.object(sched_mod, "_poll_interval_sleep", return_value=None), \
                         patch.object(sched_mod, "_check_hits_midgame", side_effect=mg_hook), \
                         patch.object(p, "get_game_statuses_detailed", return_value={}), \
                         patch("bts.posting.format_result_reply", return_value="ok"), \
                         patch("bts.posting.reply_to_bluesky", return_value="at://r/1"):
                        ret_val = sched_mod.run_result_polling(778899, day, d)

                elif br in ("poll_final_hit", "poll_final_void", "poll_final_peer"):
                    _plant_pick_for_any_date(d, day, result=None)

                    def fin_hook(*_a):
                        if br == "poll_final_peer":
                            peer = p.load_pick(day, d)
                            peer.result = "hit"
                            p.save_pick(peer, d)
                            return {"pick": "hit"}
                        return {"pick": "void" if br == "poll_final_void" else "hit"}

                    with patch.object(sched_mod, "_now_et", return_value=FIXED_NOW_ET), \
                         patch.object(sched_mod, "poll_game_result", return_value="final"), \
                         patch.object(p, "get_game_statuses_detailed", return_value={}), \
                         patch.object(p, "resolve_daily_slot_results", side_effect=fin_hook), \
                         patch("bts.posting.format_result_reply", return_value="ok"), \
                         patch("bts.posting.reply_to_bluesky", return_value="at://r/1"):
                        ret_val = sched_mod.run_result_polling(778899, day, d)

                elif br == "reconcile_incomplete":
                    _plant_pick_for_any_date(d, "2027-03-23", result=None)  # unresolved past pick
                    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
                         patch.object(p, "check_hit", return_value=None):
                        ret_val = p.reconcile_results(
                            d, clock=lambda: dt.datetime(2027, 3, 25, 2, 0, tzinfo=ET)
                        )

                elif br == "reconcile_complete":
                    _plant_pick_for_any_date(d, "2027-03-24", result="hit")
                    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
                         patch.object(p, "check_hit", return_value=True):
                        ret_val = p.reconcile_results(
                            d, clock=lambda: dt.datetime(2027, 3, 25, 2, 0, tzinfo=ET)
                        )

                elif br == "cli_reconcile_incomplete":
                    _plant_pick_for_any_date(d, "2027-03-23", result=None)
                    t_rec = dt.datetime(2027, 3, 25, 2, 0, tzinfo=ET)
                    with patch("bts.reconcile_receipt.ReconcileReceipt",
                               side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_rec, lookback_days=kw["lookback_days"])), \
                         patch.object(p, "get_game_statuses_detailed", return_value={}), \
                         patch.object(p, "check_hit", return_value=None):
                        r = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(d)])
                        if r.exception is not None and not isinstance(r.exception, SystemExit):
                            raise r.exception
                        ret_val = {"exit_code": r.exit_code, "output": r.output}

                elif br in ("decision_model_only", "decision_with_contest"):
                    if br == "decision_with_contest":
                        sd = d / "account_state"
                        sd.mkdir()
                        (sd / "contest_streak.json").write_text(json.dumps({
                            "schema_version": "bts_contest_streak_auto_v1",
                            "active_streak": 7,
                            "best_streak": 7,
                            "source": "mlb_bts_profile",
                            "source_date": "2027-03-24",
                        }))
                    st = load_decision_streak_state(
                        d, require_contest_state=False, now=FIXED_NOW_ET
                    )
                    ret_val = {
                        "streak": st.streak,
                        "model_streak": st.model_streak,
                        "model_saver": st.model_saver_available,
                        "has_error": st.model_state_error is not None,
                    }

            except BaseException as exc:
                exc_info = {"type": type(exc).__name__, "msg": str(exc)}

            after_state = "DIR" if sp.is_dir() else _read_bytes_safe(sp)
            pick_res = _read_pick_result_safe(d, day)
            # Clean up chmod 000 so tempdir cleanup works
            if pay_id == "chmod_000" and sp.exists() and not sp.is_dir():
                os.chmod(sp, 0o600)

            out[pay_id][br] = {
                "ret": ret_val,
                "exc": exc_info,
                "before": before_state,
                "after": after_state,
                "preserved": before_state == after_state,
                "pick_result": pick_res,
            }
    return out


def run_rows_a_to_k_worker(mode: str, work_root: Path) -> dict:
    """Run targeted probes for Rows A..K on either base or head."""
    import bts.cli as cli_mod
    from bts.cli import cli
    from bts.contest_state import load_decision_streak_state
    from bts.data.backup import verify_restored_ops_tree
    from bts.health.streak_validation import check as health_check
    import bts.picks as p
    from bts.reconcile_receipt import ReconcileReceipt
    from bts.state.export import export_initial_state
    from bts.state.regenerate import Timeline, compose_state_from_snapshot_and_timeline
    from bts.state.verify import diff_pick_files
    import bts.web as web_mod
    from click.testing import CliRunner

    res = {}

    # Row A: dateless + seasonal (where supported)
    d_a = work_root / f"rowA_{mode}"
    d_a.mkdir()
    (d_a / "streak.json").write_bytes(TAGGED_2026)
    res["A_dateless_2026_file"] = {
        "streak": p.load_streak(d_a),
        "saver": p.load_saver_available(d_a),
        "applied": p.load_applied_dates(d_a),
        "bytes_unchanged": (d_a / "streak.json").read_bytes() == TAGGED_2026,
    }
    if mode == "head":
        res["A_season_2027_on_2026_file"] = {
            "streak": p.load_streak(d_a, season=2027),
            "saver": p.load_saver_available(d_a, season=2027),
            "applied": p.load_applied_dates(d_a, season=2027),
            "bytes_unchanged": (d_a / "streak.json").read_bytes() == TAGGED_2026,
        }
        res["A_season_2026_on_2026_file"] = {
            "streak": p.load_streak(d_a, season=2026),
            "saver": p.load_saver_available(d_a, season=2026),
            "applied": p.load_applied_dates(d_a, season=2026),
        }
        # Bool-tag vs season=1 check (True == 1 trap)
        d_bool = work_root / "rowA_bool"
        d_bool.mkdir()
        (d_bool / "streak.json").write_text('{"season": true, "streak": 44, "saver_available": false, "applied_dates": ["2027-01-01"]}')
        res["A_bool_tag_vs_season_1"] = {
            "streak": p.load_streak(d_bool, season=1),
            "saver": p.load_saver_available(d_bool, season=1),
            "applied": p.load_applied_dates(d_bool, season=1),
        }
        # Positional season must be TypeError (keyword-only)
        pos_errors = []
        for fn in (p.load_streak, p.load_saver_available, p.load_applied_dates):
            try:
                fn(d_a, 2027)
            except TypeError:
                pos_errors.append(True)
        # IntEnum must be TypeError (exact int required)
        enum_errors = []
        for fn in (
            lambda: p.load_streak(d_a, season=SeasonEnum.S2027),
            lambda: p.load_saver_available(d_a, season=SeasonEnum.S2027),
            lambda: p.load_applied_dates(d_a, season=SeasonEnum.S2027),
            lambda: p.save_streak(1, d_a, season=SeasonEnum.S2027),
        ):
            try:
                fn()
            except TypeError:
                enum_errors.append(True)
        res["A_kwonly_and_exact_int"] = {
            "positional_rejected": pos_errors == [True, True, True],
            "intenum_rejected": enum_errors == [True, True, True, True],
        }

    # Row B & D: dateless save_streak and dateless update_streak on a 2026-tagged file
    d_bd = work_root / f"rowBD_{mode}"
    d_bd.mkdir()
    (d_bd / "streak.json").write_bytes(TAGGED_2026)
    p.save_streak(14, d_bd)
    doc_b = json.loads((d_bd / "streak.json").read_text())
    p.update_streak([True], d_bd)
    doc_d = json.loads((d_bd / "streak.json").read_text())
    res["BD_dateless_on_2026_tagged"] = {
        "after_save": {k: doc_b.get(k) for k in ("streak", "saver_available", "applied_dates", "season")},
        "after_update": {k: doc_d.get(k) for k in ("streak", "saver_available", "applied_dates", "season")},
    }

    # Row C: dated update_streak on 2026-tagged file with 2027 date
    d_c = work_root / f"rowC_{mode}"
    d_c.mkdir()
    (d_c / "streak.json").write_bytes(TAGGED_2026)
    c_ret = p.update_streak([True], d_c, date="2027-03-24")
    doc_c = json.loads((d_c / "streak.json").read_text())
    res["C_rollover_hit"] = {
        "ret": c_ret,
        "doc": {k: doc_c.get(k) for k in ("streak", "saver_available", "applied_dates", "season")},
    }

    # Row E: reconcile_results stamps replay_day.year and protects tagless today marker
    d_e = work_root / f"rowE_{mode}"
    d_e.mkdir()
    (d_e / "streak.json").write_bytes(TAGGED_2026)
    _plant_pick_for_any_date(d_e, "2027-03-24", result="hit")
    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True):
        p.reconcile_results(d_e, clock=lambda: dt.datetime(2027, 3, 25, 2, 0, tzinfo=ET))
    doc_e = json.loads((d_e / "streak.json").read_text())

    d_e2 = work_root / f"rowE_marker_{mode}"
    d_e2.mkdir()
    (d_e2 / "streak.json").write_text(json.dumps({
        "streak": 6, "saver_available": True, "applied_dates": ["2027-03-24"],
    }))
    _plant_pick_for_any_date(d_e2, "2027-03-24", result=None)
    before_e2 = (d_e2 / "streak.json").read_bytes()
    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=None):
        p.reconcile_results(d_e2, clock=lambda: dt.datetime(2027, 3, 24, 23, 30, tzinfo=ET))
    res["E_reconcile"] = {
        "replay_doc": {k: doc_e.get(k) for k in ("streak", "saver_available", "applied_dates", "season")},
        "tagless_marker_protected": (d_e2 / "streak.json").read_bytes() == before_e2,
    }

    # Row F: reconcile CLI summary at UTC 2027-01-01 01:00Z (ET 2026) vs 05:05Z (ET 2027)
    d_f = work_root / f"rowF_{mode}"
    d_f.mkdir()
    (d_f / "streak.json").write_bytes(TAGGED_2026)
    _plant_pick_for_any_date(d_f, "2026-12-20", result=None)  # incomplete 2026 so replay is None in 2026
    t_utc_ny = dt.datetime(2027, 1, 1, 1, 0, tzinfo=dt.timezone.utc)  # 2026-12-31 20:00 ET
    with patch("bts.reconcile_receipt.ReconcileReceipt",
               side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_utc_ny, lookback_days=kw["lookback_days"])), \
         patch.object(p, "reconcile_results", return_value=[]):
        rf_2026 = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(d_f)])
    t_et_ny = dt.datetime(2027, 1, 1, 5, 5, tzinfo=dt.timezone.utc)  # 2027-01-01 00:05 ET
    with patch("bts.reconcile_receipt.ReconcileReceipt",
               side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_et_ny, lookback_days=kw["lookback_days"])), \
         patch.object(p, "reconcile_results", return_value=[]):
        rf_2027 = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(d_f)])
    res["F_cli_summary"] = {
        "at_2026_et": rf_2026.output.strip(),
        "at_2027_et": rf_2027.output.strip(),
    }

    # Row G: contest_state._load_model_diagnostics via load_decision_streak_state
    d_g = work_root / f"rowG_{mode}"
    d_g.mkdir()
    (d_g / "streak.json").write_bytes(TAGGED_2026)
    st_2026 = load_decision_streak_state(
        d_g, require_contest_state=False, now=dt.datetime(2027, 1, 1, 1, 0, tzinfo=dt.timezone.utc)
    )
    st_2027 = load_decision_streak_state(
        d_g, require_contest_state=False, now=dt.datetime(2027, 1, 1, 5, 5, tzinfo=dt.timezone.utc)
    )
    res["G_decision_state"] = {
        "et_2026": (st_2026.streak, st_2026.model_streak, st_2026.model_saver_available, st_2026.model_state_error),
        "et_2027": (st_2027.streak, st_2027.model_streak, st_2027.model_saver_available, st_2027.model_state_error),
    }

    # Row H & K: tagless file not rewritten by read; mid-season deploy reads 0 until 02:00 reconcile stamps it
    d_hk = work_root / f"rowHK_{mode}"
    d_hk.mkdir()
    (d_hk / "streak.json").write_bytes(LEGACY_UNTAGGED)
    _plant_pick_for_any_date(d_hk, "2027-05-01", result="hit")
    _plant_pick_for_any_date(d_hk, "2027-05-02", result="hit")
    st_before = load_decision_streak_state(
        d_hk, require_contest_state=False, now=dt.datetime(2027, 5, 2, 18, 0, tzinfo=ET)
    )
    bytes_after_read = (d_hk / "streak.json").read_bytes() == LEGACY_UNTAGGED
    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True):
        p.reconcile_results(d_hk, clock=lambda: dt.datetime(2027, 5, 3, 2, 0, tzinfo=ET))
    st_after = load_decision_streak_state(
        d_hk, require_contest_state=False, now=dt.datetime(2027, 5, 3, 12, 0, tzinfo=ET)
    )
    doc_hk = json.loads((d_hk / "streak.json").read_text())
    res["HK_tagless_and_midseason_deploy"] = {
        "before_reconcile_streak": st_before.streak,
        "read_preserved_bytes": bytes_after_read,
        "after_reconcile_streak": st_after.streak,
        "after_reconcile_doc": {k: doc_hk.get(k) for k in ("streak", "saver_available", "season")},
    }

    # Row I: export, regenerate, verify, health, backup, web on a season-tagged file
    d_i = work_root / f"rowI_{mode}" / "data" / "picks"
    d_i.mkdir(parents=True)
    (d_i / "streak.json").write_text(json.dumps({
        "streak": 7, "saver_available": False, "season": 2027, "applied_dates": ["2027-04-10"],
    }))
    _plant_pick_for_any_date(d_i, "2027-04-10", result="hit")
    (d_i / "account_state").mkdir()
    (d_i / "account_state" / "saver_state.json").write_text(json.dumps({"state": "used", "season": 2027}))
    (d_i / "account_state" / "contest_ledger.jsonl").write_text('{"date":"2027-04-10"}\n')
    (d_i / "decisions" / "2027-04-10").mkdir(parents=True)
    (d_i / "decisions" / "2027-04-10" / "decision.json").write_text('{"date":"2027-04-10","action":"single"}')
    snap_path = work_root / f"rowI_{mode}" / "snap.json"
    snap = export_initial_state(d_i, snap_path)
    regen_dir = work_root / f"rowI_{mode}" / "regen"
    compose_state_from_snapshot_and_timeline(snap_path, Timeline(pick_records=[], final_streak=0, saver_available_at_end=True), regen_dir)
    regen_streak_doc = json.loads((regen_dir / "streak.json").read_text())
    with patch.object(web_mod, "PICKS_DIR", d_i):
        web_val = web_mod.load_streak()
    res["I_untouched_consumers"] = {
        "health_alerts": len(health_check(d_i)),
        "backup_problems": verify_restored_ops_tree(work_root / f"rowI_{mode}"),
        "export_streak": snap["streak_at_cutoff"],
        "export_has_season": "season" in snap,
        "regen_has_season": "season" in regen_streak_doc,
        "verify_clean": diff_pick_files(d_i, regen_dir).is_clean,
        "web_streak": web_val,
    }
    return res


def run_season_of_date_torture_head() -> dict:
    """Torture-test season_of_date directly on HEAD: types, non-int check, 50M-char speed, weird objects."""
    import bts.picks as p

    case_results = {}
    for cid, val in build_date_cases():
        t0 = time.perf_counter()
        out = p.season_of_date(val)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert out is None or type(out) is int, f"Non-int returned for {cid}: {out!r} ({type(out)})"
        case_results[cid] = {"season": out, "type": type(out).__name__, "ms": round(dt_ms, 4)}

    # Giant strings (1M, 10M, 50M chars) — verify no hang / quadratic regex blowup
    perf = {}
    for label, s in [
        ("digits_10M", "2" * 10_000_000),
        ("prefix_dash_10M", "2027-03-" + "2" * 10_000_000),
        ("spaces_50M", " " * 50_000_000),
    ]:
        t0 = time.perf_counter()
        out = p.season_of_date(s)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert out is None, f"Expected None for {label}, got {out!r}"
        assert elapsed_ms < 500.0, f"Too slow on {label}: {elapsed_ms:.1f}ms"
        perf[label] = round(elapsed_ms, 3)

    # Pathological custom objects
    class BadRepr:
        def __str__(self):
            raise RuntimeError("boom str")
        def __repr__(self):
            raise RuntimeError("boom repr")

    assert p.season_of_date(BadRepr()) is None
    return {"cases": case_results, "giant_string_ms": perf}


def compare_base_and_head(base_data: dict, head_data: dict) -> dict:
    """Classify every (case_id, path_name) in the 59 x 11 Row L matrix."""
    torture = head_data["torture"]["cases"]
    summary_counts = {"=": 0, "a": 0, "b": 0, "D": 0, "UNEXPLAINED": 0}
    unexplained = []
    differing = []
    matrix_table = {}

    for cid, _val in build_date_cases():
        season = torture[cid]["season"]
        matrix_table[cid] = {"season_of_date": season, "paths": {}}
        for path_name, b_rec in base_data["date_matrix"][cid].items():
            h_rec = head_data["date_matrix"][cid][path_name]

            # Strip internal call-arg season='NO_ARG' vs season=None when comparing pure '='
            b_cmp = {
                "outcome": b_rec["outcome"],
                "exc": b_rec["exc"],
                "stdout": b_rec["stdout"],
                "stderr": b_rec["stderr"],
                "before_hex": b_rec["before_hex"],
                "after_hex": b_rec["after_hex"],
                "pick_result": b_rec["pick_result"],
                "call_vals": [(c[0], c[2]) for c in b_rec["calls"]],
            }
            h_cmp = {
                "outcome": h_rec["outcome"],
                "exc": h_rec["exc"],
                "stdout": h_rec["stdout"],
                "stderr": h_rec["stderr"],
                "before_hex": h_rec["before_hex"],
                "after_hex": h_rec["after_hex"],
                "pick_result": h_rec["pick_result"],
                "call_vals": [(c[0], c[2]) for c in h_rec["calls"]],
            }

            # Verify read-only paths never modified streak.json
            if path_name in (
                "shadow_real", "poll_midgame_peer", "poll_final_peer",
                "poll_final_void", "cli_void_default", "cli_void_allow_stale", "select_pick",
            ):
                assert h_rec["before_hex"] == h_rec["after_hex"], (
                    f"Read path {path_name} mutated streak.json on {cid}!"
                )

            if b_cmp == h_cmp:
                cls = "="
            elif cid == "none" and path_name == "update_streak":
                # Dateless update_streak: both return 14, head preserves "season": 2026 per Rows B/D
                if (
                    b_rec["outcome"] == {"ret": 14}
                    and h_rec["outcome"] == {"ret": 14}
                    and b_rec["exc"] is None
                    and h_rec["exc"] is None
                ):
                    cls = "D"
                else:
                    cls = "UNEXPLAINED"
            elif season is not None:
                # Readable date: difference must be season-scoped read or write (a)
                # Note: if season == 2026 (none in our table, all parseable are != 2026), it would match TAGGED_2026
                if h_rec["exc"] == b_rec["exc"]:
                    cls = "a"
                else:
                    cls = "UNEXPLAINED"
            else:
                # Season is None!
                # L1..L3 read paths MUST be '='; only paths that call update_streak with date!=None may be 'b'
                writer_paths = {"update_streak", "poll_midgame_score", "poll_final_score", "cli_score_allow_stale"}
                if (
                    path_name in writer_paths
                    and h_rec["exc"] is not None
                    and h_rec["exc"]["type"] == "ValueError"
                    and "cannot determine season" in h_rec["exc"]["msg"]
                    and h_rec["before_hex"] == h_rec["after_hex"]
                    and h_rec["calls"] == []
                ):
                    cls = "b"
                else:
                    cls = "UNEXPLAINED"

            summary_counts[cls] += 1
            matrix_table[cid]["paths"][path_name] = cls
            if cls != "=":
                differing.append({
                    "case": cid,
                    "path": path_name,
                    "season": season,
                    "class": cls,
                    "base_outcome": b_rec["outcome"],
                    "head_outcome": h_rec["outcome"],
                    "base_exc": b_rec["exc"],
                    "head_exc": h_rec["exc"],
                })
            if cls == "UNEXPLAINED":
                unexplained.append({
                    "case": cid,
                    "path": path_name,
                    "base": b_rec,
                    "head": h_rec,
                })

    # Compare corrupt matrix
    corrupt_diffs = []
    for pay_id, branches in base_data["corrupt_matrix"].items():
        for br, b_rec in branches.items():
            h_rec = head_data["corrupt_matrix"][pay_id][br]
            if br == "reconcile_complete":
                # Both should rebuild (except is_directory where os.replace over a dir raises IsADirectoryError at both!)
                if b_rec["exc"] != h_rec["exc"]:
                    corrupt_diffs.append((pay_id, br, "exc_mismatch", b_rec, h_rec))
                elif b_rec["exc"] is None:
                    # Head stamps "season": 2027 on rebuild
                    pass
            else:
                if (
                    b_rec["exc"] != h_rec["exc"]
                    or b_rec["ret"] != h_rec["ret"]
                    or not h_rec["preserved"]
                    or b_rec["pick_result"] != h_rec["pick_result"]
                ):
                    corrupt_diffs.append((pay_id, br, "unexpected_diff", b_rec, h_rec))

    return {
        "summary_counts": summary_counts,
        "unexplained": unexplained,
        "differing_count": len(differing),
        "differing": differing,
        "matrix_table": matrix_table,
        "corrupt_unexpected_diffs": corrupt_diffs,
    }


def worker_main(mode: str, out_file: Path) -> None:
    src_loaded = _configure_worker_imports(mode)
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix=f"tmp_{mode}_") as td:
        work_root = Path(td)
        payload = {
            "mode": mode,
            "picks_file": str(src_loaded),
            "rows_a_to_k": run_rows_a_to_k_worker(mode, work_root),
            "date_matrix": run_date_matrix_worker(mode, work_root),
            "corrupt_matrix": run_corrupt_matrix_worker(mode, work_root),
        }
        if mode == "head":
            payload["torture"] = run_season_of_date_torture_head()
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], Path(sys.argv[3]))
        return

    base_json = REVIEW_DIR / "raw_base.json"
    head_json = REVIEW_DIR / "raw_head.json"
    env = os.environ.copy()
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    for mode, out_p in [("base", base_json), ("head", head_json)]:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", mode, str(out_p)],
            check=True,
            env=env,
        )

    base_data = json.loads(base_json.read_text(encoding="utf-8"))
    head_data = json.loads(head_json.read_text(encoding="utf-8"))
    comp = compare_base_and_head(base_data, head_data)

    report_out = {
        "base_picks": base_data["picks_file"],
        "head_picks": head_data["picks_file"],
        "rows_a_to_k_base": base_data["rows_a_to_k"],
        "rows_a_to_k_head": head_data["rows_a_to_k"],
        "torture_giant_strings_ms": head_data["torture"]["giant_string_ms"],
        "comparison": comp,
    }
    (REVIEW_DIR / "probe_base_vs_head.json").write_text(
        json.dumps(report_out, indent=2), encoding="utf-8"
    )

    lines = [
        f"Base picks.py: {base_data['picks_file']}",
        f"Head picks.py: {head_data['picks_file']}",
        f"Giant string timings (ms): {head_data['torture']['giant_string_ms']}",
        f"Matrix counts (59 cases x 11 paths = 649 pairs): {comp['summary_counts']}",
        f"Unexplained matrix diffs: {len(comp['unexplained'])}",
        f"Unexpected corrupt-state diffs (11 payloads x 13 branches = 143 pairs): {len(comp['corrupt_unexpected_diffs'])}",
        "",
        "Per-case Row L matrix:",
    ]
    header_paths = [
        "shadow_real", "poll_midgame_peer", "poll_midgame_score",
        "poll_final_peer", "poll_final_void", "poll_final_score",
        "cli_void_default", "cli_void_allow_stale", "cli_score_allow_stale",
        "select_pick", "update_streak",
    ]
    lines.append("case | season | " + " | ".join(header_paths))
    for cid, row in comp["matrix_table"].items():
        cells = " | ".join(row["paths"][pn] for pn in header_paths)
        lines.append(f"{cid} | {row['season_of_date']} | {cells}")

    if comp["unexplained"]:
        lines.append("\nUNEXPLAINED DIFFS:")
        lines.append(json.dumps(comp["unexplained"], indent=2))
    if comp["corrupt_unexpected_diffs"]:
        lines.append("\nUNEXPECTED CORRUPT DIFFS:")
        lines.append(json.dumps(comp["corrupt_unexpected_diffs"], indent=2))

    summary_txt = "\n".join(lines) + "\n"
    (REVIEW_DIR / "probe_base_vs_head_summary.txt").write_text(summary_txt, encoding="utf-8")
    print(summary_txt)


if __name__ == "__main__":
    main()
