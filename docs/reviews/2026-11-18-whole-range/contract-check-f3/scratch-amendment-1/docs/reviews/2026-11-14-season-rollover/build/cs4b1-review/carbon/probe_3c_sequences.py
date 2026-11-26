"""Probe 3c: Complete ledger readers, operator sequence with real MDP policy context, and parser matrix analysis."""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
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


sys.addaudithook(_audit_hook)

from probe_3_parser_ledger_sequences import analyze_all  # noqa: E402


def run_all() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    import ast
    import pandas  # noqa: F401
    import httpx
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
    work = HERE / "_p3c_main"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

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
        assert total == len(preds)
        if exp_by_yr is not None:
            assert line["settled_by_year"] == exp_by_yr
            assert line["settled_unmapped"] == exp_unmapped
            assert line["unsettled"] == exp_unsettled
        partition_checked += 1

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
        {"recorded_at": "2027-07-14T12:00:00Z", "active_streak": 12, "best_streak": 12, "source_date": "2027-07-13", "predictions": sample_preds},
        {"recorded_at": "2027-07-15T12:00:00Z", "active_streak": 12, "best_streak": 12, "source_date": "2027-07-14", "predictions": sample_preds},
    ]
    new_lines_1 = [
        {**old_lines[0], "season": 2027, "switched": False, "activation_id": None, "settled_by_year": {"2027": 2}, "settled_unmapped": 0, "unsettled": 0},
        {**old_lines[1], "season": 2027, "switched": True, "activation_id": ID_A, "settled_by_year": {"2027": 2}, "settled_unmapped": 0, "unsettled": 0},
    ]

    reader_results = {}
    dir_old = work / "readers_old/picks"
    dir_new = work / "readers_new/picks"
    for d, lines_data in [(dir_old, old_lines), (dir_new, new_lines_1)]:
        (d / "account_state").mkdir(parents=True)
        (d / "streak.json").write_text(json.dumps({"season": 2027, "streak": 0}), encoding="utf-8")
        (d / "account_state/saver_state.json").write_text(json.dumps({"season": 2027, "state": "active"}), encoding="utf-8")
        (d / "account_state/contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1", "active_streak": 12, "best_streak": 12,
            "source": "mlb_bts_profile", "source_date": "2027-07-14",
        }), encoding="utf-8")
        (d / "2027-07-14").mkdir(parents=True)
        (d / "2027-07-14/decision.json").write_text(json.dumps({"action": "single", "date": "2027-07-14"}), encoding="utf-8")
        (d / "account_state/contest_ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines_data), encoding="utf-8")

    assert cl_mod.parse_latest_ledger(dir_old / "account_state/contest_ledger.jsonl") == cl_mod.parse_latest_ledger(dir_new / "account_state/contest_ledger.jsonl")
    reader_results["src/bts/contest_ledger.py::parse_latest_ledger"] = "IDENTICAL (likely_save=True)"
    assert web_mod.saver_dashboard_context(dir_old, now=NOW) == web_mod.saver_dashboard_context(dir_new, now=NOW)
    reader_results["src/bts/web.py::saver_dashboard_context"] = "IDENTICAL (nudge=True)"
    assert verify_restored_ops_tree(dir_old.parent) == verify_restored_ops_tree(dir_new.parent) == []
    reader_results["src/bts/data/backup.py::verify_restored_ops_tree"] = "IDENTICAL (problems=[])"
    assert bsc_mod.load_ledger(dir_old / "account_state/contest_ledger.jsonl") == bsc_mod.load_ledger(dir_new / "account_state/contest_ledger.jsonl")
    reader_results["scripts/audit/boundary_shadow_census.py::load_ledger"] = "IDENTICAL"

    p05_src = (ROOT / "scripts/audit/p05_tail_policy_audit.py").read_text(encoding="utf-8")
    p05_mod = ast.parse(p05_src)
    main_fn = [n for n in p05_mod.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    ledger_stmts = [s for s in main_fn.body if "contest_ledger.jsonl" in ast.unparse(s) or "ledger_last" in ast.unparse(s)]
    compiled_stmts = compile(ast.Module(body=ledger_stmts, type_ignores=[]), "p05_tail_policy_audit.py", "exec")
    ns_old, ns_new = {"picks": dir_old, "json": json, "open": open}, {"picks": dir_new, "json": json, "open": open}
    exec(compiled_stmts, ns_old)
    exec(compiled_stmts, ns_new)
    assert ns_old["contest"] == ns_new["contest"]
    reader_results["scripts/audit/p05_tail_policy_audit.py::main"] = "IDENTICAL"

    rel_l = "picks/account_state/contest_ledger.jsonl"
    b_old = (dir_old / "account_state/contest_ledger.jsonl").read_bytes()
    b_new = (dir_new / "account_state/contest_ledger.jsonl").read_bytes()
    p_old, p_new = parse_contest_ledger(rel_l, b_old), parse_contest_ledger(rel_l, b_new)
    assert p_old.quarantined == p_new.quarantined == []

    def strip_raw_and_hash(rows):
        return [
            {
                **{k: v for k, v in r.items() if k not in {"content_sha256", "obs_id", "record_raw_json"}},
                **({"record_raw_json": r["record_raw_json"]} if r["row_level"] != "line" else {}),
            }
            for r in rows
        ]

    assert strip_raw_and_hash(p_old.rows) == strip_raw_and_hash(p_new.rows)
    reader_results["scripts/audit/season_ledger/sources/contest_ledger.py::parse_contest_ledger"] = "TYPED PROJECTIONS IDENTICAL (line record_raw_json/content_sha256/obs_id carry new keys)"
    assert required_locators("contest_ledger", b_old) == required_locators("contest_ledger", b_new)
    assert census_problems({rel_l: b_new}, {rel_l: "contest_ledger"}, account({rel_l: b_new}, {rel_l: "contest_ledger"}, {rel_l: p_new})) == []
    reader_results["scripts/audit/season_ledger/reconcile.py"] = "LOCATORS & CENSUS IDENTICAL"
    assert entered_rounds(p_old.rows) == entered_rounds(p_new.rows)
    assert line_round_streaks(p_old.rows) == line_round_streaks(p_new.rows)
    assert [{k: v for k, v in h.items() if k != "last_obs_id"} for h in slot_history(p_old.rows)] == [
        {k: v for k, v in h.items() if k != "last_obs_id"} for h in slot_history(p_new.rows)
    ]
    reader_results["scripts/audit/season_ledger/compile.py"] = "ENTERED ROUNDS, STREAKS & SLOT FACTS IDENTICAL"

    # Operator sequence on one directory
    seq_root = work / "op_seq"
    seq_picks = seq_root / "picks"
    seq_acc = seq_picks / "account_state"
    seq_acc.mkdir(parents=True)
    mdp_bundle = strategy_mod._load_mdp(ROOT / "data/models") if hasattr(strategy_mod, "_load_mdp") else None

    def inspect_step(step_now: dt.datetime) -> dict:
        ds = cs.load_decision_streak_state(seq_picks, require_contest_state=True, now=step_now)
        ctx = strategy_mod.DecisionContext(
            primary_p=0.82,
            second_p=0.80,
            has_diff_game=True,
            date=step_now.astimezone(ET).date().isoformat(),
            allow_double=ds.allow_double,
            mdp=mdp_bundle,
            best_streak=ds.best_streak,
            best_status=ds.best_status,
        )
        pol = strategy_mod.resolve_policy_decision(ctx, ds.streak, ds.saver_available)
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
            "health_alerts": [(a.level, a.source, a.message[:90]) for a in al],
            "disk_saver": [sv.state, sv.season],
        }

    seq_log: list[tuple[str, dict]] = []
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

    t_0326 = REAL_DATETIME(2027, 3, 26, 10, 0, tzinfo=ET)
    FixedClock.current = t_0326
    s2b = inspect_step(t_0326)
    assert s2b["decision_streak"] == 14 and s2b["best_status"] == "untrusted"
    seq_log.append(("2b_0326_before_dated_fallback_to_frozen_2026", s2b))

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

    r_st1 = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(seq_picks)])
    assert r_st1.exit_code == 0 and "switch: absent" in r_st1.output and "rows: 2; bad lines: 0" in r_st1.output

    r_en1 = runner.invoke(climod.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(seq_picks), "--note", "verified"])
    assert r_en1.exit_code == 0
    id1 = json.loads((seq_acc / "season_contract.json").read_text())["activation_id"]
    assert auto_p.read_bytes() == orig_auto_bytes

    climod._fetch_rounds = Mock(side_effect=httpx.HTTPStatusError("503", request=httpx.Request("GET", "https://offline"), response=httpx.Response(503)))
    r_f_fail_pre = runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(seq_picks)])
    assert r_f_fail_pre.exit_code == 2 and auto_p.read_bytes() == orig_auto_bytes

    t_0327 = REAL_DATETIME(2027, 3, 27, 10, 0, tzinfo=ET)
    FixedClock.current = t_0327
    cf.fetch_profile = lambda *a: {"activeStreak": 2, "seasonBestStreak": 2, "predictions": [{"roundId": 11, "result": "hit"}]}
    climod._fetch_rounds = lambda: {11: dt.date(2027, 3, 26)}
    assert runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(seq_picks)]).exit_code == 0
    s6 = inspect_step(t_0327)
    assert s6["decision_streak"] == 2 and s6["best_status"] == "trusted" and s6["disk_saver"] == ["not_earned", 2027] and s6["health_alerts"] == []
    seq_log.append(("6_switched_fetch_0327", s6))

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
    assert s8["decision_streak"] == 3 and s8["disk_saver"] == ["not_earned", 2027]
    seq_log.append(("8_revoked_fetch_frozen_obs", s8))

    r_en2 = runner.invoke(climod.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(seq_picks)])
    assert r_en2.exit_code == 0
    sw2 = json.loads(sw_p.read_text())
    id2 = sw2["activation_id"]
    assert id2 != id1 and sw2["revoked_at"] == "2027-03-28T15:00:00Z"
    r_st2 = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(seq_picks)])
    assert r_st2.exit_code == 0 and "identifier matches switch: False (comparison only)" in r_st2.output

    assert runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(seq_picks)]).exit_code == 0
    assert json.loads(auto_p.read_text())["activation_id"] == id2
    r_st3 = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(seq_picks)])
    assert "identifier matches switch: True (comparison only)" in r_st3.output
    s10 = inspect_step(t_0328)
    assert s10["decision_streak"] == 9 and s10["disk_saver"] == ["active", 2027] and s10["saver_available"] is True
    seq_log.append(("10_reenabled_switched_fetch_promotes_saver", s10))

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
    assert s11["decision_streak"] == 9 and s11["best_status"] == "untrusted" and s11["disk_saver"] == ["uninitialized", 2027] and s11["saver_available"] is False
    assert any(a[1] == "saver_state" and "stale-season flag, season 2027" in a[2] for a in s11["health_alerts"])
    seq_log.append(("11_jan1_2028_first_fetch_other_season", s11))

    shutil.rmtree(work, ignore_errors=True)
    return {
        "ledger_partition_lists_verified": partition_checked,
        "readers": reader_results,
        "operator_sequence_steps": seq_log,
    }


if __name__ == "__main__":
    m_res = run_all()
    analyze_all(m_res)
