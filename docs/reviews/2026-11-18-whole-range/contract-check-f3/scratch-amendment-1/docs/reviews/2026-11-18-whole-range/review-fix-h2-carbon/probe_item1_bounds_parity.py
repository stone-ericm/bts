"""Item 1 adversarial probe: verify ONLY the bound moved from 57 to 400 across S1..S6."""
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
BASE_SRC = OUT_DIR / "base_tree" / "src"
HEAD_SRC = REPO / "src"
ET = ZoneInfo("America/New_York")
REAL_DATETIME = dt.datetime

MISSING = "__MISSING__"
GRID = [
    -(10**20), -2, -1, 0, 1, 9, 10, 15, 16, 56, 57, 58, 59, 100, 399, 400, 401, 402, 570, 10**100,
    False, True, 0.0, 57.0, 58.0, 400.0,
    "0", "57", "58", "400", "",
    None, MISSING, [57], {"a": 1},
]

SMALL_GRID = [
    -1, 0, 9, 10, 15, 16, 56, 57, 58, 100, 399, 400, 401, 10**50,
    False, True, 57.0, "57", None, MISSING, [1], {"k": 1},
]


def _refuse_net(*args, **kwargs):
    raise OSError("offline probe: network refused")


class _Clock(REAL_DATETIME):
    at: REAL_DATETIME = REAL_DATETIME(2027, 7, 15, 12, 0, tzinfo=ET)

    @classmethod
    def now(cls, tz=None):
        return cls.at.astimezone(tz) if tz else cls.at.astimezone(ET).replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return cls.at.astimezone(dt.timezone.utc).replace(tzinfo=None)


@contextmanager
def pinned_offline(at: REAL_DATETIME):
    import bts.cli
    import bts.contest_fetch
    import bts.contest_state
    import bts.picks
    import bts.saver_state
    import bts.season_contract

    _Clock.at = at
    with ExitStack() as stack:
        stack.enter_context(patch.object(dt, "datetime", _Clock))
        for name, mod in list(sys.modules.items()):
            if (name == "bts" or name.startswith("bts.")) and getattr(mod, "datetime", None) is REAL_DATETIME:
                stack.enter_context(patch.object(mod, "datetime", _Clock))
        for m in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, m, _refuse_net))
        stack.enter_context(patch.object(socket, "getaddrinfo", _refuse_net))
        yield


def _put(d: dict, key: str, val):
    out = dict(d)
    if val == MISSING:
        out.pop(key, None)
    else:
        out[key] = val
    return out


def run_worker(side: str) -> dict:
    import bts
    from click.testing import CliRunner
    from bts import cli, contest_fetch, contest_state, picks
    from bts.leaderboard import auth

    now = REAL_DATETIME(2027, 7, 15, 12, 0, tzinfo=ET)
    results: dict = {
        "side": side,
        "bts_file": str(Path(bts.__file__).resolve()),
        "contest_state_file": str(Path(contest_state.__file__).resolve()),
        "target_streak": contest_state.TARGET_STREAK,
        "max_season_streak": getattr(contest_state, "MAX_SEASON_STREAK", None),
    }

    with pinned_offline(now), tempfile.TemporaryDirectory(prefix=f"h2c-item1-{side}-") as tmp:
        root = Path(tmp)

        # --- S1: _validate_dated_observation (live + historical) ---
        base_dated = {
            "schema_version": "bts_contest_streak_manual_v3",
            "effective_date": "2027-07-15",
            "season": 2027,
            "expires_at": "2027-07-15T23:59:59.999999-04:00",
            "active_streak": 12,
            "best_streak": 12,
            "saver_state": "used",
            "recorded_at": "2027-07-15T14:00:00Z",
            "source": "manual_cli",
            "reason": "check",
        }
        dummy_manual = Path("/synthetic/account_state/contest_streak.manual.json")
        s1 = {}
        for idx_a, a in enumerate(GRID):
            for idx_b, b in enumerate(GRID):
                payload = _put(_put(base_dated, "active_streak", a), "best_streak", b)
                for hist in (False, True):
                    key = f"{idx_a},{idx_b},{int(hist)}"
                    try:
                        st = contest_state._validate_dated_observation(payload, dummy_manual, historical=hist)
                        s1[key] = ["ok", st.streak, st.best_streak, st.saver_state, st.schema_version]
                    except Exception as exc:
                        s1[key] = ["err", type(exc).__name__, str(exc)]

        # S1 pairwise ordering with checks 1..4 and 7..10 + huge int file read
        s1_order = {}
        bad_others = {
            1: [],
            2: _put(base_dated, "effective_date", "20270715"),
            3: _put(base_dated, "season", 2026),
            4: _put(base_dated, "expires_at", "2027-07-15T23:59:59Z"),
            7: _put(base_dated, "saver_state", "bad"),
            8: _put(base_dated, "recorded_at", "not-iso"),
            9: _put(base_dated, "source", 123),
            10: _put(base_dated, "reason", 123),
        }
        for chk, base_bad in bad_others.items():
            for a, b in ((58, 58), (401, 401), (12, 401), (-1, 12)):
                payload = base_bad if chk == 1 else _put(_put(base_bad, "active_streak", a), "best_streak", b)
                for hist in (False, True):
                    try:
                        st = contest_state._validate_dated_observation(payload, dummy_manual, historical=hist)
                        s1_order[f"{chk},{a},{b},{int(hist)}"] = ["ok", st.streak, st.best_streak]
                    except Exception as exc:
                        s1_order[f"{chk},{a},{b},{int(hist)}"] = ["err", type(exc).__name__, str(exc)]

        huge_dir = root / "huge" / "account_state"
        huge_dir.mkdir(parents=True)
        for fname in ("contest_streak.manual.json", "contest_streak.json"):
            fpath = huge_dir / fname
            fpath.write_text('{"active_streak": ' + ("9" * 5000) + "}", encoding="utf-8")
            for hist in (False, True):
                try:
                    contest_state._read_state_file(fpath, require_dated=False, historical=hist)
                    s1_order[f"huge,{fname},{int(hist)}"] = ["ok"]
                except Exception as exc:
                    s1_order[f"huge,{fname},{int(hist)}"] = ["err", type(exc).__name__, str(exc)[:120]]
        results["s1"] = s1
        results["s1_order"] = s1_order

        # --- S2: _read_state_file on contest_streak.json + legacy manual control ---
        s2 = {}
        s2_dir = root / "s2" / "account_state"
        s2_dir.mkdir(parents=True)
        auto_path = s2_dir / "contest_streak.json"
        base_auto = {
            "schema_version": "bts_contest_streak_auto_v2",
            "active_streak": 12,
            "best_streak": 12,
            "source_date": "2027-07-14",
            "recorded_at": "2027-07-15T14:00:00Z",
            "source": "mlb_bts_profile",
            "season": 2027,
            "activation_id": "a" * 32,
        }
        for idx_a, a in enumerate(SMALL_GRID):
            for idx_b, b in enumerate(SMALL_GRID):
                payload = _put(_put(base_auto, "active_streak", a), "best_streak", b)
                auto_path.write_text(json.dumps(payload), encoding="utf-8")
                for hist in (False, True):
                    key = f"{idx_a},{idx_b},{int(hist)}"
                    try:
                        st = contest_state._read_state_file(auto_path, require_dated=True, historical=hist)
                        s2[key] = ["ok", st.streak, st.best_streak, st.season, st.activation_id]
                    except Exception as exc:
                        # Normalize temp root in path
                        msg = str(exc).replace(str(auto_path), "<AUTO_PATH>")
                        s2[key] = ["err", type(exc).__name__, msg]
        # Legacy manual v2 control at 58 and 401
        man_path = s2_dir / "contest_streak.manual.json"
        for a, b in ((0, 0), (57, 57), (58, 58), (13, 58), (401, 401), (13, 401), (13, 10), (-1, 5)):
            man_path.write_text(json.dumps({
                "schema_version": "bts_contest_streak_manual_v2",
                "active_streak": a, "best_streak": b,
                "source_date": "2027-07-15", "override_expires_at": "2027-07-16T00:00:00Z",
            }), encoding="utf-8")
            for hist in (False, True):
                try:
                    st = contest_state._read_state_file(man_path, require_dated=False, historical=hist)
                    s2[f"legacy,{a},{b},{int(hist)}"] = ["ok", st.streak, st.best_streak]
                except Exception as exc:
                    msg = str(exc).replace(str(man_path), "<MAN_PATH>")
                    s2[f"legacy,{a},{b},{int(hist)}"] = ["err", type(exc).__name__, msg]
        results["s2"] = s2

        # --- S3: classify_best_streak ---
        s3 = {}
        prov_cases = {
            "dated_active": dict(
                source="manual_cli", source_date=dt.date(2027, 7, 15),
                schema_version="bts_contest_streak_manual_v3", effective_date=dt.date(2027, 7, 15),
                override_expires_at=REAL_DATETIME(2027, 7, 15, 23, 59, 59, 999999, tzinfo=ET),
            ),
            "dated_expired": dict(
                source="manual_cli", source_date=dt.date(2027, 7, 14),
                schema_version="bts_contest_streak_manual_v3", effective_date=dt.date(2027, 7, 14),
                override_expires_at=REAL_DATETIME(2027, 7, 14, 23, 59, 59, 999999, tzinfo=ET),
            ),
            "auto_v2": dict(
                source="mlb_bts_profile", source_date=dt.date(2027, 7, 14),
                schema_version="bts_contest_streak_auto_v2", effective_date=None, override_expires_at=None,
            ),
            "auto_v1": dict(
                source="mlb_bts_profile", source_date=dt.date(2027, 7, 15),
                schema_version="bts_contest_streak_auto_v1", effective_date=None, override_expires_at=None,
            ),
            "auto_wrong_season": dict(
                source="mlb_bts_profile", source_date=dt.date(2026, 7, 15),
                schema_version="bts_contest_streak_auto_v2", effective_date=None, override_expires_at=None,
            ),
            "auto_future": dict(
                source="mlb_bts_profile", source_date=dt.date(2027, 7, 16),
                schema_version="bts_contest_streak_auto_v2", effective_date=None, override_expires_at=None,
            ),
            "auto_no_date": dict(
                source="mlb_bts_profile", source_date=None,
                schema_version="bts_contest_streak_auto_v2", effective_date=None, override_expires_at=None,
            ),
            "auto_wrong_source": dict(
                source="other", source_date=dt.date(2027, 7, 14),
                schema_version="bts_contest_streak_auto_v2", effective_date=None, override_expires_at=None,
            ),
            "manual_v2_unexpired": dict(
                source="manual_cli", source_date=dt.date(2027, 7, 15),
                schema_version="bts_contest_streak_manual_v2", effective_date=None,
                override_expires_at=now + dt.timedelta(hours=2),
            ),
            "manual_v2_expired": dict(
                source="manual_cli", source_date=dt.date(2027, 7, 15),
                schema_version="bts_contest_streak_manual_v2", effective_date=None,
                override_expires_at=now,
            ),
            "no_schema": dict(
                source="mlb_bts_profile", source_date=dt.date(2027, 7, 15),
                schema_version=None, effective_date=None, override_expires_at=None,
            ),
        }
        for pname, pkw in prov_cases.items():
            for s_val in (0, 12, 57, 58, 100, 400, 401):
                for b_val in (-2, -1, 0, 11, 12, 57, 58, 100, 399, 400, 401, 10**20, None):
                    c_obj = contest_state.ContestStreakState(
                        streak=s_val, saver_available=None, recorded_at="2027-07-15T14:00:00Z",
                        path=dummy_manual, best_streak=b_val, **pkw,
                    )
                    val, status = contest_state.classify_best_streak(c_obj, now=now, now_year=2027)
                    s3[f"{pname},{s_val},{b_val}"] = [val, status]
        s3["none"] = list(contest_state.classify_best_streak(None, now=now, now_year=2027))
        results["s3"] = s3

        # --- S4: contest_fetch.validate_fetch + real CLI fetch ---
        s4 = {}
        for idx_a, a in enumerate(GRID):
            for idx_b, b in enumerate(GRID):
                payload = _put(_put({"activeStreak": 0, "seasonBestStreak": 0}, "activeStreak", a), "seasonBestStreak", b)
                try:
                    contest_fetch.validate_fetch(payload)
                    s4[f"{idx_a},{idx_b}"] = ["ok"]
                except Exception as exc:
                    s4[f"{idx_a},{idx_b}"] = ["err", type(exc).__name__, str(exc)]
        results["s4"] = s4

        s4_cli = {}
        runner = CliRunner()
        for switched in (False, True):
            for dry in (False, True):
                for idx_c, (a, b) in enumerate([
                    (0, 0), (57, 57), (58, 58), (58, 57), (12, 60), (400, 400),
                    (400, 401), (401, 400), (401, 401), (-1, 10), (True, 10),
                ]):
                    case_dir = root / f"s4_cli_{int(switched)}_{int(dry)}_{idx_c}" / "picks"
                    case_dir.mkdir(parents=True)
                    if switched:
                        res_en = runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(case_dir)])
                        assert res_en.exit_code == 0
                    session = SimpleNamespace(user_id=42, username="acct", xsid="x")
                    prof = {"activeStreak": a, "seasonBestStreak": b, "predictions": [{"roundId": 1, "result": "hit"}]}
                    sent_dms = []
                    with ExitStack() as st:
                        st.enter_context(patch.object(auth, "load_session_cookies", lambda: {}))
                        st.enter_context(patch.object(auth, "extract_uid", lambda c: 42))
                        st.enter_context(patch.object(auth, "fetch_login_session", lambda **kw: session))
                        st.enter_context(patch.object(contest_fetch, "fetch_profile", lambda *args: prof))
                        st.enter_context(patch.object(cli, "_fetch_rounds", lambda: {1: dt.date(2027, 7, 14)}))
                        st.enter_context(patch("bts.dm.send_dm", side_effect=lambda r, m: (sent_dms.append((r, m)) or "dm-1")))
                        argv = ["fetch-contest-streak", "--picks-dir", str(case_dir), "--expected-username", "acct", "--dm-recipient", "rec"]
                        if dry:
                            argv.append("--dry-run")
                        r = runner.invoke(cli.cli, argv)
                    obs_p = case_dir / "account_state" / "contest_streak.json"
                    led_p = case_dir / "account_state" / "contest_ledger.jsonl"
                    sav_p = case_dir / "account_state" / "saver_state.json"
                    st_p = case_dir.parent / "health_state" / "contest_streak_fetch_status.json"
                    s4_cli[f"{int(switched)},{int(dry)},{idx_c}"] = {
                        "exit": r.exit_code,
                        "out": r.output.replace(str(case_dir), "<PICKS>"),
                        "obs": json.loads(obs_p.read_text())["active_streak"] if obs_p.exists() else None,
                        "best": json.loads(obs_p.read_text())["best_streak"] if obs_p.exists() else None,
                        "ledger_lines": len(led_p.read_text().splitlines()) if led_p.exists() else 0,
                        "saver_exists": sav_p.exists(),
                        "status_err": json.loads(st_p.read_text()).get("last_error") if st_p.exists() else None,
                        "dms": sent_dms,
                    }
        results["s4_cli"] = s4_cli

        # --- S5: CLI set-contest-streak (dated and undated) ---
        s5 = {}
        cli_cases = [
            # dated valid / boundary / invalid
            ["--date", "2027-07-15", "--streak", "0", "--best-streak", "0", "--saver", "not_earned"],
            ["--date", "2027-07-15", "--streak", "57", "--best-streak", "57", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "58", "--best-streak", "58", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "12", "--best-streak", "58", "--saver", "active"],
            ["--date", "2027-07-15", "--streak", "58", "--best-streak", "400", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "400", "--best-streak", "400", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "400", "--best-streak", "401", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "401", "--best-streak", "401", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "401", "--best-streak", "400", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "58", "--best-streak", "57", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "12", "--best-streak", "401", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "-1", "--best-streak", "10", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "-1", "--best-streak", "401", "--saver", "used"],
            ["--date", "2027-07-14", "--streak", "58", "--best-streak", "58", "--saver", "used"],
            ["--date", "2027-07-16", "--streak", "401", "--best-streak", "401", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "58", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "58", "--best-streak", "58"],
            ["--date", "2027-07-15", "--streak", "58", "--best-streak", "58", "--saver", "used", "--ttl-hours", "5"],
            ["--date", "2027-07-15", "--streak", "12.0", "--best-streak", "12", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "abc", "--best-streak", "12", "--saver", "used"],
            # undated legacy controls
            ["--streak", "58", "--best-streak", "58"],
            ["--streak", "401", "--best-streak", "401"],
            ["--streak", "58", "--best-streak", "57"],
            ["--streak", "-1"],
        ]
        for idx_c, extra in enumerate(cli_cases):
            cdir = root / f"s5_{idx_c}"
            cdir.mkdir()
            r_exc = runner.invoke(cli.cli, ["set-contest-streak", "--picks-dir", str(cdir), *extra], standalone_mode=False)
            cdir2 = root / f"s5_run_{idx_c}"
            cdir2.mkdir()
            r_std = runner.invoke(cli.cli, ["set-contest-streak", "--picks-dir", str(cdir2), *extra])
            written = sorted(str(p.relative_to(cdir2)) for p in cdir2.rglob("*") if p.is_file())
            man_file = cdir2 / "account_state" / "contest_streak.manual.json"
            parsed_man = json.loads(man_file.read_text()) if man_file.exists() else None
            if parsed_man:
                parsed_man.pop("recorded_at", None)
                parsed_man.pop("override_expires_at", None)
            s5[str(idx_c)] = {
                "extra": extra,
                "exc_type": type(r_exc.exception).__name__ if r_exc.exception is not None else None,
                "exit": r_std.exit_code,
                "out": r_std.output.replace(str(cdir2), "<PICKS>"),
                "written": written,
                "manual": parsed_man,
            }
        results["s5"] = s5

        # --- S6: picks._valid_chosen_on ---
        s6 = {}
        # 1) Non-dated kinds across GRID
        for kind in ("automatic", "legacy_manual", "model", "unknown", None, True):
            for idx_s, s_val in enumerate(GRID):
                for req in (True, False, 1):
                    for dated_field in (None, {"effective_date": "2027-07-15"}):
                        rec = _put({
                            "schema": "bts_pick_chosen_on_v1",
                            "required": req,
                            "kind": kind,
                            "streak": 0,
                            "saver_available": True,
                            "dated": dated_field,
                        }, "streak", s_val)
                        out = picks._valid_chosen_on(rec)
                        s6[f"nondated,{kind},{idx_s},{req},{dated_field is not None}"] = (out is not None)

        # 2) Dated kind across SMALL_GRID x SMALL_GRID
        for idx_a, a in enumerate(SMALL_GRID):
            for idx_b, b in enumerate(SMALL_GRID):
                for saver_st, sav_bool in (("used", False), ("active", True), ("not_earned", False), ("not_earned", True), ("bad", False)):
                    dated_sub = _put(_put({
                        "effective_date": "2027-07-15",
                        "active_streak": 0,
                        "best_streak": 0,
                        "saver_state": saver_st,
                    }, "active_streak", a), "best_streak", b)
                    rec = _put({
                        "schema": "bts_pick_chosen_on_v1",
                        "required": True,
                        "kind": "dated",
                        "streak": 0,
                        "saver_available": sav_bool,
                        "dated": dated_sub,
                    }, "streak", a)
                    out = picks._valid_chosen_on(rec)
                    s6[f"dated,{idx_a},{idx_b},{saver_st},{int(sav_bool)}"] = (out is not None)
                # Also mismatch outer streak != active_streak when both int
                if type(a) is int and a >= 0:
                    dated_sub = {
                        "effective_date": "2027-07-15",
                        "active_streak": a,
                        "best_streak": b if b != MISSING else 0,
                        "saver_state": "used",
                    }
                    rec = {
                        "schema": "bts_pick_chosen_on_v1",
                        "required": True,
                        "kind": "dated",
                        "streak": a + 1,
                        "saver_available": False,
                        "dated": dated_sub,
                    }
                    s6[f"dated_mismatch,{idx_a},{idx_b}"] = (picks._valid_chosen_on(rec) is not None)
        results["s6"] = s6

    return results


def _is_exact_int(v) -> bool:
    return type(v) is int


def compare_sides(base: dict, head: dict) -> list[str]:
    problems: list[str] = []

    if base["target_streak"] != 57 or head["target_streak"] != 57:
        problems.append(f"TARGET_STREAK changed: base={base['target_streak']} head={head['target_streak']}")
    if base["max_season_streak"] is not None or head["max_season_streak"] != 400:
        problems.append(f"MAX_SEASON_STREAK unexpected: base={base['max_season_streak']} head={head['max_season_streak']}")
    if base["bts_file"] == head["bts_file"]:
        problems.append(f"base and head imported same bts.__file__: {base['bts_file']}")

    # --- Compare S1 ---
    s1_changed = 0
    for key, b_res in base["s1"].items():
        h_res = head["s1"][key]
        ia, ib, ih = (int(x) for x in key.split(","))
        a, b, hist = GRID[ia], GRID[ib], bool(ih)
        in_new_valid = _is_exact_int(a) and _is_exact_int(b) and 0 <= a <= b <= 400 and (a > 57 or b > 57)
        in_high_a_bad_b = _is_exact_int(a) and 58 <= a <= 400 and not in_new_valid
        if in_new_valid:
            s1_changed += 1
            if b_res[0] != "err":
                problems.append(f"S1 base unexpectedly accepted {a}/{b} hist={hist}: {b_res}")
            if h_res != ["ok", a, b, "used", "bts_contest_streak_manual_v3"]:
                problems.append(f"S1 head did not accept {a}/{b} hist={hist}: {h_res}")
        elif in_high_a_bad_b:
            s1_changed += 1
            # Base rejected active at check 5; head accepts active at check 5 and rejects best at check 6
            if hist:
                if b_res != ["err", "_HistoricalContestStateError", "The dated manual contest streak file failed validation for active_streak."]:
                    problems.append(f"S1 hist base unexpected on {a}/{b}: {b_res}")
                if h_res != ["err", "_HistoricalContestStateError", "The dated manual contest streak file failed validation for best_streak."]:
                    problems.append(f"S1 hist head unexpected on {a}/{b}: {h_res}")
            else:
                if b_res[0] != "err" or "check 5, active_streak: must be an integer from 0 to 57" not in b_res[2]:
                    problems.append(f"S1 base unexpected on {a}/{b}: {b_res}")
                if h_res[0] != "err" or "check 6, best_streak: must be an integer from active_streak to 400" not in h_res[2]:
                    problems.append(f"S1 head unexpected on {a}/{b}: {h_res}")
        else:
            norm_h = list(h_res)
            if norm_h[0] == "err":
                norm_h[2] = norm_h[2].replace("from 0 to 400", "from 0 to 57").replace("from active_streak to 400", "from active_streak to 57")
            if b_res != norm_h:
                problems.append(f"S1 parity mismatch at {a!r}/{b!r} hist={hist}: base={b_res} head={h_res}")

    for key, b_res in base["s1_order"].items():
        h_res = head["s1_order"][key]
        if key.startswith("58,58") or ",58,58," in key:
            parts = key.split(",")
            chk, hist = int(parts[0]), bool(int(parts[3]))
            if chk < 5:
                # Checks 1..4 MUST fire before check 5 on BOTH base and head!
                if b_res != h_res:
                    problems.append(f"S1 check {chk}<5 order mismatch: base={b_res} head={h_res}")
            else:
                # Checks 7..10: base failed at check 5 (58>57), head reaches check 7..10!
                if not hist and f"check {chk}," not in h_res[2]:
                    problems.append(f"S1 head did not reach check {chk} on 58/58: {h_res}")
        else:
            norm_h = list(h_res)
            if len(norm_h) >= 3 and norm_h[0] == "err":
                norm_h[2] = norm_h[2].replace("from 0 to 400", "from 0 to 57").replace("from active_streak to 400", "from active_streak to 57")
            if b_res != norm_h:
                problems.append(f"S1 order mismatch at {key}: base={b_res} head={h_res}")

    # --- Compare S2 ---
    s2_changed = 0
    for key, b_res in base["s2"].items():
        h_res = head["s2"][key]
        if key.startswith("legacy,"):
            if b_res != h_res:
                problems.append(f"S2 legacy manual changed at {key}: base={b_res} head={h_res}")
            continue
        ia, ib, ih = (int(x) for x in key.split(","))
        a, b, hist = SMALL_GRID[ia], SMALL_GRID[ib], bool(ih)
        b_int = b if _is_exact_int(b) else None
        in_new_valid = (
            _is_exact_int(a) and 0 <= a <= 400
            and (b_int is None or a <= b_int <= 400)
            and (a > 57 or (b_int is not None and b_int > 57))
        )
        if in_new_valid:
            s2_changed += 1
            if b_res[0] != "err":
                problems.append(f"S2 base unexpectedly accepted {a}/{b} hist={hist}: {b_res}")
            if h_res != ["ok", a, b_int, 2027, "a" * 32]:
                problems.append(f"S2 head unexpected on {a}/{b} hist={hist}: {h_res}")
        else:
            if b_res != h_res:
                problems.append(f"S2 parity mismatch at {a!r}/{b!r} hist={hist}: base={b_res} head={h_res}")

    # --- Compare S3 ---
    s3_changed = 0
    for key, b_res in base["s3"].items():
        h_res = head["s3"][key]
        if key == "none":
            if b_res != h_res:
                problems.append(f"S3 None mismatch: {b_res} vs {h_res}")
            continue
        pname, s_str, b_str = key.split(",")
        s_val = int(s_str)
        b_val = None if b_str == "None" else int(b_str)
        if b_val is not None and 58 <= b_val <= 400 and 0 <= s_val <= b_val:
            s3_changed += 1
            if b_res != [b_val, "untrusted"]:
                problems.append(f"S3 base unexpected at {key}: {b_res}")
            ref_status = base["s3"][f"{pname},{min(s_val, 57)},57"][1]
            if h_res != [b_val, ref_status]:
                problems.append(f"S3 head unexpected at {key}: got {h_res}, expected {[b_val, ref_status]}")
        else:
            if b_res != h_res:
                problems.append(f"S3 parity mismatch at {key}: base={b_res} head={h_res}")

    # --- Compare S4 ---
    s4_changed = 0
    for key, b_res in base["s4"].items():
        h_res = head["s4"][key]
        ia, ib = (int(x) for x in key.split(","))
        a, b = GRID[ia], GRID[ib]
        if _is_exact_int(a) and _is_exact_int(b) and 0 <= a <= b <= 400 and b > 57:
            s4_changed += 1
            if b_res != ["err", "ContestFetchError", "seasonBestStreak must be <= 57"]:
                problems.append(f"S4 base unexpected at {a}/{b}: {b_res}")
            if h_res != ["ok"]:
                problems.append(f"S4 head unexpected at {a}/{b}: {h_res}")
        else:
            norm_h = list(h_res)
            if _is_exact_int(a) and _is_exact_int(b) and 0 <= a <= b and b > 400 and norm_h[0] == "err":
                norm_h[2] = norm_h[2].replace("<= 400", "<= 57")
            if b_res != norm_h:
                problems.append(f"S4 parity mismatch at {a!r}/{b!r}: base={b_res} head={h_res}")

    s4_cli_pairs = [
        (0, 0), (57, 57), (58, 58), (58, 57), (12, 60), (400, 400),
        (400, 401), (401, 400), (401, 401), (-1, 10), (True, 10),
    ]
    for key, b_res in base["s4_cli"].items():
        h_res = head["s4_cli"][key]
        sw, dr, ic = (int(x) for x in key.split(","))
        a, b = s4_cli_pairs[ic]
        if 0 <= a <= b <= 400 and b > 57:
            if b_res["exit"] != 2 or h_res["exit"] != 0:
                problems.append(f"S4 CLI {key} ({a}/{b}) unexpected exits: base={b_res['exit']} head={h_res['exit']}")
            if not dr and sw and (h_res["obs"], h_res["best"], h_res["ledger_lines"]) != (a, b, 1):
                problems.append(f"S4 CLI {key} head switched write wrong: {h_res}")
            if not dr and not sw and (h_res["obs"], h_res["ledger_lines"]) != (None, 1):
                problems.append(f"S4 CLI {key} head unswitched write wrong: {h_res}")
            if dr and (h_res["obs"], h_res["ledger_lines"]) != (None, 0):
                problems.append(f"S4 CLI {key} head dry-run wrote files: {h_res}")
        else:
            norm_h = json.loads(json.dumps(h_res).replace("<= 400", "<= 57"))
            if b_res != norm_h:
                problems.append(f"S4 CLI parity mismatch at {key} ({a}/{b}): base={b_res} head={h_res}")

    # --- Compare S5 ---
    s5_changed = 0
    for key, b_res in base["s5"].items():
        h_res = head["s5"][key]
        extra = b_res["extra"]
        is_dated_high_valid = (
            "--date" in extra and extra[1] == "2027-07-15"
            and "--best-streak" in extra and "--saver" in extra and "--ttl-hours" not in extra
        )
        if is_dated_high_valid:
            try:
                sv = int(extra[extra.index("--streak") + 1])
                bv = int(extra[extra.index("--best-streak") + 1])
                is_dated_high_valid = (0 <= sv <= bv <= 400 and bv > 57)
            except ValueError:
                is_dated_high_valid = False
        if is_dated_high_valid:
            s5_changed += 1
            if b_res["exit"] != 2 or b_res["written"] != [] or "at most 57" not in b_res["out"]:
                problems.append(f"S5 base unexpected at {extra}: {b_res}")
            if h_res["exit"] != 0 or h_res["written"] != [
                "account_state/contest_manual_requests.jsonl",
                "account_state/contest_streak.manual.json",
            ]:
                problems.append(f"S5 head unexpected at {extra}: {h_res}")
        else:
            norm_h = json.loads(json.dumps(h_res).replace("at most 400", "at most 57"))
            if b_res != norm_h:
                problems.append(f"S5 parity mismatch at {extra}: base={b_res} head={h_res}")

    # --- Compare S6 ---
    s6_changed = 0
    for key, b_ok in base["s6"].items():
        h_ok = head["s6"][key]
        if key.startswith("nondated,"):
            _, kind_s, idx_s, req_s, has_dated_s = key.split(",")
            s_val = GRID[int(idx_s)]
            expect_newly_valid = (
                kind_s == "automatic" and req_s in ("True", "False") and has_dated_s == "False"
                and _is_exact_int(s_val) and 58 <= s_val <= 400
            )
        elif key.startswith("dated,"):
            _, ia_s, ib_s, saver_st, sav_b = key.split(",")
            a, b = SMALL_GRID[int(ia_s)], SMALL_GRID[int(ib_s)]
            sav_bool = bool(int(sav_b))
            valid_saver = (
                (saver_st == "active" and sav_bool is True)
                or (saver_st == "used" and sav_bool is False)
                or (saver_st == "not_earned" and _is_exact_int(a) and sav_bool == (a < 10))
            )
            expect_newly_valid = (
                _is_exact_int(a) and _is_exact_int(b) and 0 <= a <= b <= 400 and b >= 58 and valid_saver
            )
        else:
            expect_newly_valid = False

        if expect_newly_valid:
            s6_changed += 1
            if b_ok is not False or h_ok is not True:
                problems.append(f"S6 newly-valid mismatch at {key}: base={b_ok} head={h_ok}")
        else:
            if b_ok != h_ok:
                problems.append(f"S6 parity mismatch at {key}: base={b_ok} head={h_ok}")

    print(
        f"counts: s1={len(base['s1'])} (changed={s1_changed}), "
        f"s2={len(base['s2'])} (changed={s2_changed}), "
        f"s3={len(base['s3'])} (changed={s3_changed}), "
        f"s4={len(base['s4'])}+{len(base['s4_cli'])} (changed={s4_changed}), "
        f"s5={len(base['s5'])} (changed={s5_changed}), "
        f"s6={len(base['s6'])} (changed={s6_changed})"
    )
    return problems


def check_diff_lines() -> list[str]:
    problems = []
    proc = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--name-only", "36fd3e35", "b12e144c", "--", "src"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    changed_files = sorted(proc.stdout.strip().splitlines())
    expected_files = [
        "src/bts/cli.py",
        "src/bts/contest_fetch.py",
        "src/bts/contest_state.py",
        "src/bts/picks.py",
    ]
    if changed_files != expected_files:
        problems.append(f"unexpected src files changed: {changed_files}")

    # Fresh-import order test: import contest_fetch first, then picks first
    for mod in ("bts.contest_fetch", "bts.picks", "bts.contest_state", "bts.cli"):
        env = dict(os.environ, PYTHONPATH=str(HEAD_SRC))
        subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c", f"import {mod}"],
            env=env, check=True, timeout=15, capture_output=True,
        )
    return problems


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        out = run_worker(sys.argv[2])
        sys.stdout.write(json.dumps(out))
        return 0

    log_lines = []
    problems = check_diff_lines()

    side_data = {}
    for side, src_path in (("base", BASE_SRC), ("head", HEAD_SRC)):
        env = dict(os.environ, PYTHONPATH=str(src_path))
        proc = subprocess.run(
            [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "--worker", side],
            env=env, capture_output=True, text=True, check=True, timeout=60,
        )
        side_data[side] = json.loads(proc.stdout)
        line = f"{side} bts.__file__={side_data[side]['bts_file']} MAX_SEASON_STREAK={side_data[side]['max_season_streak']}"
        print(line)
        log_lines.append(line)

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        problems.extend(compare_sides(side_data["base"], side_data["head"]))
    summary = buf.getvalue().strip()
    print(summary)
    log_lines.append(summary)
    log_lines.append(f"problems={len(problems)}")
    for p in problems[:50]:
        print("PROBLEM:", p)
        log_lines.append(f"PROBLEM: {p}")

    (OUT_DIR / "item1_bounds_parity.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
