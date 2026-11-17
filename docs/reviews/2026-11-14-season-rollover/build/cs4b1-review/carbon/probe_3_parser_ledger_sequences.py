"""Probe 3: Parser & trust base-vs-branch matrix (Item 6), ledger counts & all readers (Item 7), and full operator sequence (Item 8)."""
from __future__ import annotations

import ast
import datetime as dt
import gzip
import itertools as it
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ET = ZoneInfo("America/New_York")
NOW = dt.datetime(2027, 7, 15, 12, 0, 0, tzinfo=ET)
REAL_DATETIME = dt.datetime
ID_A = "a" * 32
ID_B = "b" * 32

SWITCH_TOUCHED: list[str] = []
WATCH_ROOT: Path | None = None


def _audit_hook(event: str, args: tuple) -> None:
    if event in {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.getnameinfo",
        "socket.sendto",
        "urllib.Request",
    }:
        raise RuntimeError(f"FORBIDDEN NETWORK AUDIT EVENT: {event} {args!r}")
    w = WATCH_ROOT
    if w is not None and event in {"open", "os.stat"}:
        p = args[0]
        if isinstance(p, (str, os.PathLike)) and "season_contract.json" in str(p) and str(p).startswith(str(w)):
            SWITCH_TOUCHED.append(f"{event}:{p}")


sys.addaudithook(_audit_hook)


def contract_doc(**overrides) -> dict:
    base = {
        "schema_version": "bts_season_contract_v1",
        "season": 2027,
        "status": "automatic",
        "activation_id": ID_A,
        "switched_at": "2027-07-15T16:00:00Z",
        "revoked_at": None,
        "note": None,
    }
    base.update(overrides)
    return base


MANUAL_CASES: dict[str, object] = {
    "absent": None,
    "legacy_unexpired": {
        "schema_version": "bts_contest_streak_manual_v2",
        "active_streak": 5,
        "best_streak": 11,
        "source": "manual_cli",
        "source_date": "2027-07-15",
        "override_expires_at": "2027-07-16T00:00:00Z",
    },
    "legacy_expired": {
        "schema_version": "bts_contest_streak_manual_v2",
        "active_streak": 5,
        "best_streak": 11,
        "source": "manual_cli",
        "source_date": "2027-07-14",
        "override_expires_at": "2027-07-15T00:00:00Z",
    },
    "legacy_no_exp": {"active_streak": 5, "best_streak": 11},
    "manual_with_auto_v1": {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 6,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
    },
    "manual_with_auto_v2_bad_meta": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 6,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
        "season": True,
        "activation_id": "NOT_HEX",
    },
    "manual_with_auto_v2_other_src": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 6,
        "best_streak": 12,
        "source": "manual_cli",
        "source_date": "2027-07-14",
    },
    "dated_today": {
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": "2027-07-15",
        "season": 2027,
        "expires_at": "2027-07-16T03:59:59.999999Z",
        "active_streak": 3,
        "best_streak": 9,
        "saver_state": "active",
        "recorded_at": "2027-07-15T12:00:00Z",
        "source": "manual_cli",
    },
    "dated_yesterday": {
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": "2027-07-14",
        "season": 2027,
        "expires_at": "2027-07-15T03:59:59.999999Z",
        "active_streak": 3,
        "best_streak": 9,
        "saver_state": "active",
        "recorded_at": "2027-07-14T12:00:00Z",
        "source": "manual_cli",
    },
    "invalid_manual": {"active_streak": -1},
}

AUTO_PARSER_CASES: dict[str, object] = {
    "absent": None,
    "v1_valid": {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 7,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
        "recorded_at": "2027-07-15T12:00:00Z",
    },
    "v1_2026": {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 7,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2026-09-28",
        "recorded_at": "2026-09-29T12:00:00Z",
    },
    "v1_undated": {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 7,
        "best_streak": 12,
        "source": "mlb_bts_profile",
    },
    "v1_recorded_only": {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 7,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "recorded_at": "2027-07-15T12:00:00Z",
    },
    "schema_missing": {"active_streak": 7, "best_streak": 12, "source": "mlb_bts_profile", "source_date": "2027-07-14"},
    "schema_null": {"schema_version": None, "active_streak": 7, "best_streak": 12, "source": "mlb_bts_profile", "source_date": "2027-07-14"},
    "schema_int": {"schema_version": 123, "active_streak": 7, "best_streak": 12, "source": "mlb_bts_profile", "source_date": "2027-07-14"},
    "schema_v2_trailing_space": {
        "schema_version": "bts_contest_streak_auto_v2 ",
        "active_streak": 7,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
        "season": True,
        "activation_id": "bad",
    },
    "refused_best58": {"schema_version": "bts_contest_streak_auto_v1", "active_streak": 7, "best_streak": 58, "source_date": "2027-07-14"},
    "not_utf8": b"\xff",
    "not_object": b"[]",
    "v2_valid": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 7,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
        "recorded_at": "2027-07-15T12:00:00Z",
        "season": 2027,
        "activation_id": ID_A,
    },
    "v2_other_season_2027_src": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 7,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
        "recorded_at": "2027-07-15T12:00:00Z",
        "season": 2026,
        "activation_id": ID_B,
    },
    "v2_2026_src": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 7,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2026-09-28",
        "recorded_at": "2026-09-29T12:00:00Z",
        "season": 2027,
        "activation_id": ID_A,
    },
    "v2_other_source": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 7,
        "best_streak": 12,
        "source": "hand_written",
        "source_date": "2027-07-14",
        "season": 2027,
        "activation_id": ID_A,
    },
    "v2_bad_season_bool": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 7,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
        "season": True,
        "activation_id": ID_A,
    },
    "v2_bad_id": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 7,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
        "season": 2027,
        "activation_id": "A" * 32,
    },
    "v2_both_bad": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 7,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
        "season": "2027",
        "activation_id": None,
    },
    "v2_legacy_err_wins_over_season": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 58,
        "best_streak": 58,
        "season": None,
        "activation_id": None,
    },
    "v2_undated_err_wins_over_season": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 7,
        "best_streak": 9,
        "season": None,
        "activation_id": None,
    },
}


def seed_file(path: Path, spec: object) -> None:
    if spec is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(spec, bytes):
        path.write_bytes(spec)
    else:
        path.write_text(json.dumps(spec), encoding="utf-8")


def capture_call(fn, case_str: str) -> dict:
    try:
        val = fn()
    except Exception as exc:
        return {"err": type(exc).__name__, "msg": str(exc).replace(case_str, "<CASE>")}
    if val is None:
        return {"ok": None}
    if isinstance(val, list):
        return {
            "ok": [
                {"level": a.level, "source": a.source, "message": a.message.replace(case_str, "<CASE>")}
                for a in val
            ]
        }
    if hasattr(val, "__dataclass_fields__"):
        d = {}
        for k in val.__dataclass_fields__:
            # Ignore the two new optional fields ONLY when checking base parity on non-v2 inputs (checked separately too!)
            v = getattr(val, k)
            if isinstance(v, Path):
                v = str(v).replace(case_str, "<CASE>")
            elif isinstance(v, (dt.date, dt.datetime)):
                v = v.isoformat()
            d[k] = v
        return {"ok": d}
    return {"ok": val}


def run_parser_worker(src_dir: Path, label: str) -> None:
    sys.path.insert(0, str(src_dir))
    import bts.contest_state as cs
    from bts.health import contest_state as health

    work = HERE / f"_p3_worker_{label}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    out: dict[str, dict] = {}
    global WATCH_ROOT

    # 1. Direct _parse_state_file + classify_best_streak on manual & auto filenames
    for fname, cases in [("contest_streak.manual.json", MANUAL_CASES), ("contest_streak.json", AUTO_PARSER_CASES)]:
        for cname, spec in cases.items():
            if spec is None:
                continue
            for req in (False, True):
                cdir = work / "direct"
                if cdir.exists():
                    shutil.rmtree(cdir)
                p = cdir / "account_state" / fname
                seed_file(p, spec)
                seed_file(cdir / "account_state/season_contract.json", b"\xff")  # corrupt switch must never be read!
                SWITCH_TOUCHED.clear()
                WATCH_ROOT = cdir
                parsed_res = capture_call(lambda: cs._parse_state_file(p, require_dated=req), str(cdir))
                trust_res = None
                if "ok" in parsed_res and parsed_res["ok"] is not None:
                    obj = cs._parse_state_file(p, require_dated=req)
                    trust_res = cs.classify_best_streak(obj, now=NOW, now_year=2027)
                WATCH_ROOT = None
                assert SWITCH_TOUCHED == [], f"switch touched: {SWITCH_TOUCHED}"
                out[f"parse|{fname}|{cname}|req={int(req)}"] = {"parse": parsed_res, "trust": trust_res}

    # 2. Full selection + decision + health matrix: 10 manual x 21 auto x 2 req x 2 switch (corrupt vs valid) = 840 combos!
    for m_name, a_name, req, sw_corrupt in it.product(
        MANUAL_CASES.keys(), AUTO_PARSER_CASES.keys(), [False, True], [False, True]
    ):
        cdir = work / "sel"
        if cdir.exists():
            shutil.rmtree(cdir)
        picks = cdir / "picks"
        picks.mkdir(parents=True)
        seed_file(picks / "account_state/contest_streak.manual.json", MANUAL_CASES[m_name])
        seed_file(picks / "account_state/contest_streak.json", AUTO_PARSER_CASES[a_name])
        seed_file(picks / "account_state/saver_state.json", {"season": 2027, "state": "active"})
        seed_file(picks / "account_state/season_contract.json", b"\xff" if sw_corrupt else contract_doc())

        SWITCH_TOUCHED.clear()
        WATCH_ROOT = cdir
        sel_res = capture_call(lambda: cs.load_contest_streak_state(picks, now=NOW, require_dated_auto=req), str(cdir))
        dec_res = capture_call(lambda: cs.load_decision_streak_state(picks, require_contest_state=req, now=NOW), str(cdir))
        hlth_res = capture_call(lambda: health.check(picks, expected=req, now=NOW), str(cdir))
        WATCH_ROOT = None
        assert SWITCH_TOUCHED == [], f"switch touched during loader/health: {SWITCH_TOUCHED}"

        out[f"matrix|{m_name}|{a_name}|req={int(req)}|sw={int(sw_corrupt)}"] = {
            "sel": sel_res,
            "dec": dec_res,
            "hlth": hlth_res,
        }

    shutil.rmtree(work, ignore_errors=True)
    (HERE / f"probe3_{label}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"PARSER WORKER {label}: {len(out)} records, zero switch touches", flush=True)


def strip_new_contest_fields(rec: dict) -> dict:
    """Return copy of ContestStreakState dict without season/activation_id when both are None."""
    r = json.loads(json.dumps(rec))
    for section in ("parse", "sel"):
        if section in r and isinstance(r[section], dict) and isinstance(r[section].get("ok"), dict):
            ok = r[section]["ok"]
            if ok.get("season") is None and ok.get("activation_id") is None:
                ok.pop("season", None)
                ok.pop("activation_id", None)
    return r


def run_ledger_and_sequences() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    import pandas  # noqa: F401
    from click.testing import CliRunner
    import bts.cli as climod
    import bts.contest_fetch as cf
    import bts.contest_ledger as cl_mod
    import bts.contest_state as cs
    import bts.dm as dm_mod
    import bts.leaderboard.auth as auth
    import bts.saver_state as saver_mod
    import bts.strategy as strategy_mod
    import bts.web as web_mod
    from bts.data.backup import verify_restored_ops_tree
    from bts.health import contest_state as health
    import scripts.audit.boundary_shadow_census as bsc_mod
    from scripts.audit.season_ledger.sources.contest_ledger import parse_contest_ledger
    from scripts.audit.season_ledger.reconcile import account, required_locators, census_problems
    from scripts.audit.season_ledger.contest import entered_rounds, line_round_streaks, slot_history

    class FixedClock(REAL_DATETIME):
        current = NOW

        @classmethod
        def now(cls, tz=None):
            c = cls.current
            return c.astimezone(tz) if tz else c.replace(tzinfo=None)

    dt.datetime = FixedClock
    runner = CliRunner()
    work = HERE / "_p3_main"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # ---- ITEM 7: LEDGER PARTITION & ALL READERS ----
    rounds_map = {1: dt.date(2025, 9, 1), 2: dt.date(2026, 6, 15), 3: dt.date(2027, 7, 14)}
    pred_lists = [
        ([], {}, 0, 0),
        (
            [
                {"roundId": 1, "result": "hit"},
                {"roundId": 2, "result": "not_hit"},
                {"roundId": "2", "result": "void"},
                {"roundId": 3, "result": "miss"},
                {"roundId": 3, "result": "hit"},
                {"result": "hit"},
                {"roundId": None, "result": "void"},
                {"roundId": 999, "result": "not_hit"},
                {"roundId": 3, "result": None},
                {"roundId": 3, "result": "pending"},
                {"roundId": 3, "result": "unknown_status"},
                {"roundId": 3, "result": ""},
                {},
            ],
            {"2025": 1, "2026": 2, "2027": 2},
            3,
            5,
        ),
    ]
    # Add 30 generated combinations
    rng_statuses = ["hit", "not_hit", "miss", "void", None, "pending", "other"]
    rng_rids = [1, 2, 3, 999, None]
    for k in range(30):
        rows = [
            {"result": rng_statuses[(k + j) % len(rng_statuses)], **({"roundId": rng_rids[(k * 3 + j) % len(rng_rids)]} if (k + j) % 5 != 0 else {})}
            for j in range(k % 9)
        ]
        pred_lists.append((rows, None, None, None))

    auth.load_session_cookies = lambda: {"oktaid": "u"}
    auth.extract_uid = lambda c: "u"
    auth.fetch_login_session = lambda **kw: SimpleNamespace(user_id=50311, username="stonehengee", xsid="x")
    climod._fetch_rounds = lambda: rounds_map
    dm_mod.send_dm = Mock()

    partition_checked = 0
    l_picks = work / "ledger_part/picks"
    l_picks.mkdir(parents=True)
    for preds, exp_by_yr, exp_unmapped, exp_unsettled in pred_lists:
        cf.fetch_profile = lambda *a, _p=preds: {"activeStreak": 4, "seasonBestStreak": 12, "predictions": _p}
        res = runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(l_picks)])
        assert res.exit_code == 0, res.output
        line = json.loads((l_picks / "account_state/contest_ledger.jsonl").read_text().splitlines()[-1])
        total = sum(line["settled_by_year"].values()) + line["settled_unmapped"] + line["unsettled"]
        assert total == len(preds), f"partition failed: {total} != {len(preds)} on {line}"
        if exp_by_yr is not None:
            assert line["settled_by_year"] == exp_by_yr
            assert line["settled_unmapped"] == exp_unmapped
            assert line["unsettled"] == exp_unsettled
        partition_checked += 1

    # Feed two-line ledger (with a stable save at streak 12!) WITH and WITHOUT new keys to ALL 8 readers!
    sample_preds = [
        {
            "roundId": 10,
            "result": "hit",
            "streak": 12,
            "streakIncrease": 1,
            "roundPredictions": [{"unitId": 100, "playerId": 200, "number": 1, "result": "hit", "hits": 1, "atBats": 4}],
        },
        {
            "roundId": 11,
            "result": "not_hit",
            "streak": 12,
            "streakIncrease": 0,
            "roundPredictions": [{"unitId": 101, "playerId": 201, "number": 1, "result": "not_hit", "hits": 0, "atBats": 4}],
        },
    ]
    old_lines = [
        {
            "recorded_at": "2027-07-14T12:00:00Z",
            "active_streak": 12,
            "best_streak": 12,
            "source_date": "2027-07-13",
            "predictions": sample_preds,
        },
        {
            "recorded_at": "2027-07-15T12:00:00Z",
            "active_streak": 12,
            "best_streak": 12,
            "source_date": "2027-07-14",
            "predictions": sample_preds,
        },
    ]
    new_extras_unswitched = {
        "season": 2027, "switched": False, "activation_id": None,
        "settled_by_year": {"2027": 2}, "settled_unmapped": 0, "unsettled": 0,
    }
    new_extras_switched = {
        "season": 2027, "switched": True, "activation_id": ID_A,
        "settled_by_year": {"2027": 2}, "settled_unmapped": 0, "unsettled": 0,
    }
    new_lines_1 = [{**old_lines[0], **new_extras_unswitched}, {**old_lines[1], **new_extras_switched}]

    reader_results = {}
    dir_old = work / "readers_old/picks"
    dir_new = work / "readers_new/picks"
    for d, lines_data in [(dir_old, old_lines), (dir_new, new_lines_1)]:
        (d / "account_state").mkdir(parents=True)
        (d / "account_state/saver_state.json").write_text(json.dumps({"season": 2027, "state": "active"}), encoding="utf-8")
        (d / "account_state/contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1", "active_streak": 12, "best_streak": 12,
            "source": "mlb_bts_profile", "source_date": "2027-07-14",
        }), encoding="utf-8")
        (d / "2027-07-14").mkdir(parents=True)
        (d / "2027-07-14/decision.json").write_text(json.dumps({"action": "single", "date": "2027-07-14"}), encoding="utf-8")
        (d / "account_state/contest_ledger.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in lines_data), encoding="utf-8"
        )

    # Reader 1: contest_ledger.parse_latest_ledger + likely_save
    r1_old = cl_mod.parse_latest_ledger(dir_old / "account_state/contest_ledger.jsonl")
    r1_new = cl_mod.parse_latest_ledger(dir_new / "account_state/contest_ledger.jsonl")
    assert r1_old == r1_new and cl_mod.likely_save(r1_old) is True and cl_mod.likely_save(r1_new) is True
    reader_results["contest_ledger.parse_latest_ledger"] = "IDENTICAL (likely_save=True)"

    # Reader 2: web.saver_dashboard_context
    r2_old = web_mod.saver_dashboard_context(dir_old, now=NOW)
    r2_new = web_mod.saver_dashboard_context(dir_new, now=NOW)
    assert r2_old == r2_new and r2_new.nudge is True
    reader_results["web.saver_dashboard_context"] = "IDENTICAL (nudge=True)"

    # Reader 3: data.backup.verify_restored_ops_tree
    r3_old = verify_restored_ops_tree(dir_old.parent)
    r3_new = verify_restored_ops_tree(dir_new.parent)
    assert r3_old == r3_new == []
    reader_results["data.backup.verify_restored_ops_tree"] = "IDENTICAL (problems=[])"

    # Reader 4: boundary_shadow_census.load_ledger + asof_contest
    r4_old = bsc_mod.load_ledger(dir_old / "account_state/contest_ledger.jsonl")
    r4_new = bsc_mod.load_ledger(dir_new / "account_state/contest_ledger.jsonl")
    assert r4_old == r4_new
    assert bsc_mod.asof_contest(r4_old, NOW) == bsc_mod.asof_contest(r4_new, NOW)
    reader_results["boundary_shadow_census.load_ledger"] = "IDENTICAL"

    # Reader 5: p05_tail_policy_audit.main (execute actual AST statements lines 81-84)
    p05_src = (ROOT / "scripts/audit/p05_tail_policy_audit.py").read_text(encoding="utf-8")
    p05_mod = ast.parse(p05_src)
    main_fn = [n for n in p05_mod.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    ledger_stmts = [s for s in main_fn.body if "contest_ledger.jsonl" in ast.unparse(s) or "ledger_last" in ast.unparse(s)]
    compiled_stmts = compile(ast.Module(body=ledger_stmts, type_ignores=[]), "p05_tail_policy_audit.py", "exec")
    ns_old, ns_new = {"picks": dir_old, "json": json, "open": open}, {"picks": dir_new, "json": json, "open": open}
    exec(compiled_stmts, ns_old)
    exec(compiled_stmts, ns_new)
    assert ns_old["contest"] == ns_new["contest"]
    reader_results["p05_tail_policy_audit.main"] = "IDENTICAL"

    # Reader 6, 7, 8: season_ledger parse_contest_ledger, reconcile, compile helpers
    rel_l = "picks/account_state/contest_ledger.jsonl"
    b_old = (dir_old / "account_state/contest_ledger.jsonl").read_bytes()
    b_new = (dir_new / "account_state/contest_ledger.jsonl").read_bytes()
    p_old = parse_contest_ledger(rel_l, b_old)
    p_new = parse_contest_ledger(rel_l, b_new)
    assert p_old.quarantined == p_new.quarantined == []

    def strip_raw_and_hash(rows):
        cleaned = []
        for r in rows:
            c = {k: v for k, v in r.items() if k not in {"content_sha256", "obs_id", "record_raw_json"}}
            if r["row_level"] != "line":
                c["record_raw_json"] = r["record_raw_json"]  # round & slot raw JSON must also be identical!
            cleaned.append(c)
        return cleaned

    assert strip_raw_and_hash(p_old.rows) == strip_raw_and_hash(p_new.rows)
    # Verify line record_raw_json intentionally contains the added keys
    line_raw_new = json.loads([r for r in p_new.rows if r["row_level"] == "line"][-1]["record_raw_json"])
    assert line_raw_new["season"] == 2027 and line_raw_new["switched"] is True and line_raw_new["activation_id"] == ID_A
    reader_results["season_ledger.sources.contest_ledger"] = "TYPED PROJECTIONS IDENTICAL (line record_raw_json/content_sha256/obs_id carry new keys)"

    # Reconcile locators & census
    req_old = required_locators("contest_ledger", b_old)
    req_new = required_locators("contest_ledger", b_new)
    acc_old = account({rel_l: b_old}, {rel_l: "contest_ledger"}, {rel_l: p_old})
    acc_new = account({rel_l: b_new}, {rel_l: "contest_ledger"}, {rel_l: p_new})
    assert req_old == req_new
    assert census_problems({rel_l: b_old}, {rel_l: "contest_ledger"}, acc_old) == []
    assert census_problems({rel_l: b_new}, {rel_l: "contest_ledger"}, acc_new) == []
    reader_results["season_ledger.reconcile"] = "LOCATORS & CENSUS IDENTICAL (zero problems)"

    # Compile helpers (entered_rounds, line_round_streaks, slot_history modulo last_obs_id)
    assert entered_rounds(p_old.rows) == entered_rounds(p_new.rows)
    assert line_round_streaks(p_old.rows) == line_round_streaks(p_new.rows)
    sh_old = [{k: v for k, v in h.items() if k != "last_obs_id"} for h in slot_history(p_old.rows)]
    sh_new = [{k: v for k, v in h.items() if k != "last_obs_id"} for h in slot_history(p_new.rows)]
    assert sh_old == sh_new
    reader_results["season_ledger.compile"] = "ENTERED ROUNDS, STREAKS & SLOT FACTS IDENTICAL (last_obs_id hash-derived)"

    # ---- ITEM 8: OPERATOR SEQUENCE ON ONE DIRECTORY ----
    seq_root = work / "op_seq"
    seq_picks = seq_root / "picks"
    seq_acc = seq_picks / "account_state"
    seq_acc.mkdir(parents=True)

    # Load real checked-in policy artifacts for real policy action verification!
    cand = [{"batter_id": 1, "name": "A", "p": 0.80, "game_pk": 1}, {"batter_id": 2, "name": "B", "p": 0.79, "game_pk": 2}]

    def inspect_step(step_now: dt.datetime) -> dict:
        ds = cs.load_decision_streak_state(seq_picks, require_contest_state=True, now=step_now)
        pol = strategy_mod.resolve_policy_decision(
            candidates=cand,
            streak=ds.streak,
            saver_available=ds.saver_available,
            pick_date=step_now.astimezone(ET).date().isoformat(),
            best_streak=ds.best_streak,
            best_status=ds.best_status,
        )
        al = health.check(seq_picks, expected=True, now=step_now)
        sv = saver_mod.load_saver_state(seq_picks, season=step_now.astimezone(ET).year)
        return {
            "date": step_now.astimezone(ET).date().isoformat(),
            "decision_streak": ds.streak,
            "decision_status": ds.status,
            "best_streak": ds.best_streak,
            "best_status": ds.best_status,
            "saver_available": ds.saver_available,
            "saver_flag_state_in_decision": ds.saver_flag_state,
            "policy_action": pol.action,
            "health_alerts": [(a.level, a.source, a.message[:80]) for a in al],
            "disk_saver": (sv.state, sv.season),
        }

    seq_log: list[tuple[str, dict]] = []

    # Step 1: Seed 2026 auto_v1 (streak=14, best=18) + 2026 saver_state (used)
    auto_p = seq_acc / "contest_streak.json"
    saver_p = seq_acc / "saver_state.json"
    auto_p.write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 14, "best_streak": 18,
        "source": "mlb_bts_profile", "source_date": "2026-09-27",
        "recorded_at": "2026-09-28T12:00:00Z",
        "user_id": 50311, "username": "stonehengee", "saver_available": None,
    }), encoding="utf-8")
    saver_p.write_text(json.dumps({"season": 2026, "state": "used"}), encoding="utf-8")
    orig_auto_bytes = auto_p.read_bytes()
    orig_saver_bytes = saver_p.read_bytes()

    t_0325 = REAL_DATETIME(2027, 3, 25, 10, 0, tzinfo=ET)
    FixedClock.current = t_0325
    s1 = inspect_step(t_0325)
    assert s1["decision_streak"] == 14 and s1["best_status"] == "untrusted" and s1["saver_available"] is False
    assert len(s1["health_alerts"]) == 1 and s1["health_alerts"][0][1] == "saver_state"
    seq_log.append(("1_pre_dated_0325", s1))

    # Step 2: Daily dated observations (03-25, 03-26) + unswitched fetches
    assert runner.invoke(climod.cli, [
        "set-contest-streak", "--picks-dir", str(seq_picks),
        "--date", "2027-03-25", "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
    ]).exit_code == 0
    cf.fetch_profile = lambda *a: {"activeStreak": 14, "seasonBestStreak": 18, "predictions": [{"roundId": 1, "result": "hit"}]}
    climod._fetch_rounds = lambda: {1: dt.date(2026, 9, 27)}
    assert runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(seq_picks)]).exit_code == 0
    assert auto_p.read_bytes() == orig_auto_bytes and saver_p.read_bytes() == orig_saver_bytes
    s2 = inspect_step(t_0325)
    assert s2["decision_streak"] == 0 and s2["best_status"] == "trusted" and s2["saver_available"] is True and s2["health_alerts"] == []
    seq_log.append(("2_dated_0325_plus_unswitched_fetch", s2))

    # Next day 03-26 morning BEFORE entering 03-26 dated obs -> falls back to frozen 2026 auto_v1 (streak 14)!
    t_0326 = REAL_DATETIME(2027, 3, 26, 10, 0, tzinfo=ET)
    FixedClock.current = t_0326
    s2b = inspect_step(t_0326)
    assert s2b["decision_streak"] == 14 and s2b["best_status"] == "untrusted"
    seq_log.append(("2b_0326_before_dated_fallback_to_frozen_2026", s2b))

    # Enter 03-26 dated obs (streak 1) + unswitched fetch now showing 2027 round
    assert runner.invoke(climod.cli, [
        "set-contest-streak", "--picks-dir", str(seq_picks),
        "--date", "2027-03-26", "--streak", "1", "--best-streak", "1", "--saver", "not_earned",
    ]).exit_code == 0
    cf.fetch_profile = lambda *a: {"activeStreak": 1, "seasonBestStreak": 1, "predictions": [{"roundId": 10, "result": "hit"}]}
    climod._fetch_rounds = lambda: {10: dt.date(2027, 3, 25)}
    assert runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(seq_picks)]).exit_code == 0
    assert auto_p.read_bytes() == orig_auto_bytes and saver_p.read_bytes() == orig_saver_bytes
    s2c = inspect_step(t_0326)
    assert s2c["decision_streak"] == 1 and s2c["saver_available"] is True and s2c["health_alerts"] == []
    seq_log.append(("2c_dated_0326_plus_unswitched_2027_fetch", s2c))

    # Step 3: `status` while unswitched on 03-26
    r_st1 = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(seq_picks)])
    assert r_st1.exit_code == 0 and "switch: absent" in r_st1.output and "rows: 2; bad lines: 0" in r_st1.output

    # Step 4: `enable` on 03-26 -> writes switch id1; automatic file still frozen 2026 auto_v1!
    r_en1 = runner.invoke(climod.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(seq_picks), "--note", "verified"])
    assert r_en1.exit_code == 0
    id1 = json.loads((seq_acc / "season_contract.json").read_text())["activation_id"]
    assert auto_p.read_bytes() == orig_auto_bytes

    # Step 5: Fetch failing BEFORE replacing observation (rounds HTTP 503) -> auto_p still frozen 2026!
    import httpx
    climod._fetch_rounds = Mock(side_effect=httpx.HTTPStatusError("503", request=httpx.Request("GET", "https://offline"), response=httpx.Response(503)))
    r_f_fail_pre = runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(seq_picks)])
    assert r_f_fail_pre.exit_code == 2 and auto_p.read_bytes() == orig_auto_bytes

    # Step 6: Successful switched fetch on 03-27 -> replaces auto with auto_v2 (id1, streak=2, best=2) and initializes 2027 saver to not_earned!
    t_0327 = REAL_DATETIME(2027, 3, 27, 10, 0, tzinfo=ET)
    FixedClock.current = t_0327
    cf.fetch_profile = lambda *a: {"activeStreak": 2, "seasonBestStreak": 2, "predictions": [{"roundId": 11, "result": "hit"}]}
    climod._fetch_rounds = lambda: {11: dt.date(2027, 3, 26)}
    assert runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(seq_picks)]).exit_code == 0
    s6 = inspect_step(t_0327)
    assert s6["decision_streak"] == 2 and s6["best_status"] == "trusted" and s6["disk_saver"] == ("not_earned", 2027) and s6["health_alerts"] == []
    seq_log.append(("6_switched_fetch_0327", s6))

    # Step 7: Fetch failing AFTER replacing observation (ledger append OSError -> exit 1, but auto_v2 streak=3 IS on disk!)
    t_0328 = REAL_DATETIME(2027, 3, 28, 10, 0, tzinfo=ET)
    FixedClock.current = t_0328
    cf.fetch_profile = lambda *a: {"activeStreak": 3, "seasonBestStreak": 3, "predictions": [{"roundId": 12, "result": "hit"}]}
    climod._fetch_rounds = lambda: {12: dt.date(2027, 3, 27)}
    orig_popen = Path.open

    def fail_ledger_open(self_p, mode="r", *a, **kw):
        if self_p.name == "contest_ledger.jsonl" and "a" in mode:
            raise OSError("disk full on ledger")
        return orig_popen(self_p, mode, *a, **kw)

    Path.open = fail_ledger_open
    try:
        r_f_fail_post = runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(seq_picks)])
    finally:
        Path.open = orig_popen
    assert r_f_fail_post.exit_code == 1
    assert json.loads(auto_p.read_text())["active_streak"] == 3 and json.loads(auto_p.read_text())["activation_id"] == id1
    s7 = inspect_step(t_0328)
    assert s7["decision_streak"] == 3 and s7["best_status"] == "trusted"
    seq_log.append(("7_fetch_failed_after_obs_replace_0328", s7))

    # Step 8: Hand-edited wholly valid revoke + fetch (ledger only, observation stays streak=3/id1)
    sw_p = seq_acc / "season_contract.json"
    sw_doc = json.loads(sw_p.read_text())
    sw_doc["status"] = "revoked"
    sw_doc["revoked_at"] = "2027-03-28T15:00:00Z"
    sw_p.write_text(json.dumps(sw_doc), encoding="utf-8")
    auto_before_rev_fetch = auto_p.read_bytes()
    saver_before_rev_fetch = saver_p.read_bytes()
    cf.fetch_profile = lambda *a: {"activeStreak": 9, "seasonBestStreak": 10, "predictions": [{"roundId": 13, "result": "hit"}]}
    climod._fetch_rounds = lambda: {13: dt.date(2027, 3, 28)}
    assert runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(seq_picks)]).exit_code == 0
    assert auto_p.read_bytes() == auto_before_rev_fetch and saver_p.read_bytes() == saver_before_rev_fetch
    s8 = inspect_step(t_0328)
    assert s8["decision_streak"] == 3 and s8["disk_saver"] == ("not_earned", 2027)
    seq_log.append(("8_revoked_fetch_frozen_obs", s8))

    # Step 9 & 10: `enable` again -> fresh id2, carries revoked_at; `status` shows identifier matches switch: False!
    r_en2 = runner.invoke(climod.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(seq_picks)])
    assert r_en2.exit_code == 0
    sw2 = json.loads(sw_p.read_text())
    id2 = sw2["activation_id"]
    assert id2 != id1 and sw2["revoked_at"] == "2027-03-28T15:00:00Z"
    r_st2 = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(seq_picks)])
    assert r_st2.exit_code == 0 and "identifier matches switch: False (comparison only)" in r_st2.output

    # Step 10b: Switched fetch after re-enable -> stamps id2 AND promotes 2027 saver not_earned -> active (best=10)!
    assert runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(seq_picks)]).exit_code == 0
    assert json.loads(auto_p.read_text())["activation_id"] == id2
    r_st3 = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(seq_picks)])
    assert "identifier matches switch: True (comparison only)" in r_st3.output
    s10 = inspect_step(t_0328)
    assert s10["decision_streak"] == 9 and s10["disk_saver"] == ("active", 2027) and s10["saver_available"] is True
    seq_log.append(("10_reenabled_switched_fetch_promotes_saver", s10))

    # Step 11: First fetch of January 1, 2028!
    t_jan1 = REAL_DATETIME(2028, 1, 1, 10, 0, tzinfo=ET)
    FixedClock.current = t_jan1
    auto_2027_bytes = auto_p.read_bytes()
    saver_2027_bytes = saver_p.read_bytes()
    cf.fetch_profile = lambda *a: {"activeStreak": 0, "seasonBestStreak": 0, "predictions": [{"roundId": 99, "result": "hit"}]}
    climod._fetch_rounds = lambda: {99: dt.date(2027, 9, 28)}
    r_jan_f = runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(seq_picks)])
    assert r_jan_f.exit_code == 0 and "season 2028 is not switched to automatic" in r_jan_f.output
    assert auto_p.read_bytes() == auto_2027_bytes and saver_p.read_bytes() == saver_2027_bytes
    jan_led = json.loads((seq_acc / "contest_ledger.jsonl").read_text().splitlines()[-1])
    assert jan_led["season"] == 2028 and jan_led["switched"] is False and jan_led["activation_id"] is None
    r_st_jan = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(seq_picks)])
    assert r_st_jan.exit_code == 0 and "switch: other_season" in r_st_jan.output
    s11 = inspect_step(t_jan1)
    assert s11["decision_streak"] == 9 and s11["best_status"] == "untrusted" and s11["disk_saver"] == ("uninitialized", 2027) and s11["saver_available"] is False
    assert any(a[1] == "saver_state" and "stale-season flag, season 2027" in a[2] for a in s11["health_alerts"])
    seq_log.append(("11_jan1_2028_first_fetch_other_season", s11))

    shutil.rmtree(work, ignore_errors=True)
    return {
        "ledger_partition_lists_verified": partition_checked,
        "readers": reader_results,
        "operator_sequence_steps": seq_log,
    }


def analyze_all(main_res: dict) -> None:
    base = json.loads((HERE / "probe3_base.json").read_text(encoding="utf-8"))
    branch = json.loads((HERE / "probe3_branch.json").read_text(encoding="utf-8"))
    assert len(base) == len(branch)

    parser_diffs: dict[str, int] = {}
    unexpected: list[str] = []

    for key, b_val in base.items():
        n_val = branch[key]
        b_norm = strip_new_contest_fields(b_val)
        n_norm = strip_new_contest_fields(n_val)

        if key.startswith("parse|"):
            _, fname, cname, req_part = key.split("|")
            is_exact_auto_v2 = fname == "contest_streak.json" and cname.startswith("v2_")
            is_manual_v2_profile = fname == "contest_streak.manual.json" and cname == "manual_with_auto_v2_bad_meta"
            if not is_exact_auto_v2 and not is_manual_v2_profile:
                if b_norm != n_norm:
                    unexpected.append(f"non_v2 parse diff {key}: base={b_norm} branch={n_norm}")
                else:
                    parser_diffs["unchanged_direct_parse"] = parser_diffs.get("unchanged_direct_parse", 0) + 1
            elif is_manual_v2_profile:
                # Row 7: manual file carrying auto_v2 parses by legacy rules (metadata None!) and gains best trust!
                assert b_norm["parse"] == n_norm["parse"], f"manual v2 parse changed: {b_norm} vs {n_norm}"
                assert n_val["parse"]["ok"]["season"] is None and n_val["parse"]["ok"]["activation_id"] is None
                assert b_norm["trust"] == [12, "untrusted"] and n_norm["trust"] == [12, "trusted"]
                parser_diffs["row7_manual_v2_trust_only"] = parser_diffs.get("row7_manual_v2_trust_only", 0) + 1
            else:
                # Exact auto_v2 at contest_streak.json:
                if cname in {"v2_legacy_err_wins_over_season", "v2_undated_err_wins_over_season"}:
                    if cname == "v2_legacy_err_wins_over_season" or req_part == "req=1":
                        # Existing check error MUST win byte-for-byte over new season check!
                        if b_norm != n_norm:
                            unexpected.append(f"legacy_err_order broken {key}: base={b_norm} branch={n_norm}")
                        parser_diffs["row7_legacy_error_precedes_v2"] = parser_diffs.get("row7_legacy_error_precedes_v2", 0) + 1
                    else:
                        # req=0 on undated: existing checks pass, so new season check fires!
                        assert n_val["parse"]["err"] == "ContestStateError" and "season" in n_val["parse"]["msg"]
                        parser_diffs["row7_v2_field_rejected"] = parser_diffs.get("row7_v2_field_rejected", 0) + 1
                elif cname in {"v2_bad_season_bool", "v2_bad_id", "v2_both_bad"}:
                    expected_fld = "activation_id" if cname == "v2_bad_id" else "season"
                    if not (b_val["parse"].get("ok") and n_val["parse"].get("err") == "ContestStateError" and expected_fld in n_val["parse"]["msg"]):
                        unexpected.append(f"v2_bad_field wrong {key}: {n_val}")
                    parser_diffs["row7_v2_field_rejected"] = parser_diffs.get("row7_v2_field_rejected", 0) + 1
                else:
                    # Valid v2 metadata: exposes season + activation_id; trusted iff source==mlb_bts_profile and source_date in 2027!
                    ok = n_val["parse"]["ok"]
                    assert isinstance(ok["season"], int) and len(ok["activation_id"]) == 32
                    expected_trust = "trusted" if cname in {"v2_valid", "v2_other_season_2027_src"} else "untrusted"
                    assert n_val["trust"] == [12, expected_trust], f"v2 trust wrong {key}: {n_val['trust']}"
                    parser_diffs["row7_v2_valid_metadata_and_trust"] = parser_diffs.get("row7_v2_valid_metadata_and_trust", 0) + 1
        else:
            # matrix|m_name|a_name|req=...|sw=...
            _, m_name, a_name, req_part, sw_part = key.split("|")
            touches_v2 = a_name.startswith("v2_") or m_name == "manual_with_auto_v2_bad_meta"
            if not touches_v2:
                if b_norm != n_norm:
                    unexpected.append(f"non_v2 matrix diff {key}: base={b_norm} branch={n_norm}")
                else:
                    parser_diffs["unchanged_matrix_non_v2"] = parser_diffs.get("unchanged_matrix_non_v2", 0) + 1
            else:
                # Health check never changes on valid v2 (health doesn't use best_status); invalid selected v2 becomes CRITICAL!
                if b_norm == n_norm:
                    parser_diffs["unchanged_matrix_v2_bypassed_or_untrusted"] = parser_diffs.get("unchanged_matrix_v2_bypassed_or_untrusted", 0) + 1
                else:
                    parser_diffs["row7_matrix_v2_expected_diff"] = parser_diffs.get("row7_matrix_v2_expected_diff", 0) + 1

    summary = {
        "total_parser_and_matrix_records_per_tree": len(branch),
        "parser_buckets": parser_diffs,
        "unexpected_diffs": unexpected,
        **main_res,
    }
    (HERE / "probe3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert not unexpected, f"UNEXPECTED PARSER DIFFS: {unexpected[:10]}"
    print("PROBE 3 SUMMARY:", json.dumps({k: v for k, v in summary.items() if k != "operator_sequence_steps"}, indent=2), flush=True)
    print(f"OPERATOR SEQUENCE STEPS VERIFIED: {len(main_res['operator_sequence_steps'])}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_parser_worker(Path(sys.argv[1]), sys.argv[2])
    else:
        for sdir, lbl in [(HERE / "base_tree/src", "base"), (ROOT / "src", "branch")]:
            subprocess.run([sys.executable, __file__, str(sdir), lbl], check=True, cwd=ROOT)
        m_res = run_ledger_and_sequences()
        analyze_all(m_res)
