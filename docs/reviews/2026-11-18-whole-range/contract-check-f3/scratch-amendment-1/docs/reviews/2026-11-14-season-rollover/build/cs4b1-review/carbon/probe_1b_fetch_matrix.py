"""Probe 1b: Base vs branch 5,400-case fetch matrix, audit-hooked I/O order, write-failure prefixes, and single-clock boundaries."""
from __future__ import annotations

import datetime as dt
import hashlib
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

AUDIT_EVENTS: list[tuple[str, str, str]] = []
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
    if w is None:
        return
    w_str = str(w)
    if event == "open":
        path_arg, mode, flags = args[0], args[1], args[2]
        if isinstance(path_arg, (str, os.PathLike)):
            p_str = str(path_arg)
            if p_str.startswith(w_str):
                rel = os.path.relpath(p_str, w_str)
                AUDIT_EVENTS.append(("open", rel, f"mode={mode}|flags={flags}"))
    elif event in {"os.mkdir", "os.unlink", "os.remove"}:
        p_str = str(args[0])
        if p_str.startswith(w_str):
            AUDIT_EVENTS.append((event, os.path.relpath(p_str, w_str), ""))
    elif event == "os.rename":
        src, dst = str(args[0]), str(args[1])
        if src.startswith(w_str) or dst.startswith(w_str):
            s_rel = os.path.relpath(src, w_str) if src.startswith(w_str) else src
            d_rel = os.path.relpath(dst, w_str) if dst.startswith(w_str) else dst
            AUDIT_EVENTS.append(("os.rename", f"{s_rel}->{d_rel}", ""))


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


SWITCH_CASES: dict[str, object] = {
    "absent": None,
    "c1_nonobj": b"[]",
    "c2_schema": contract_doc(schema_version="wrong"),
    "c3_bool_season": contract_doc(season=True),
    "c4_status": contract_doc(status="pending"),
    "c5_id": contract_doc(activation_id="A" * 32),
    "c6_switched": contract_doc(switched_at=None),
    "c7_revoked": contract_doc(revoked_at="2027-07-15T12:00:00"),
    "c8_note": contract_doc(note=123),
    "pair_c2_c8_other_yr": contract_doc(schema_version=None, season=2026, note=[]),
    "pair_c3_c5": contract_doc(season=False, activation_id="bad"),
    "pair_c6_c7": contract_doc(switched_at=None, revoked_at="naive"),
    "other_season": contract_doc(season=2026),
    "revoked": contract_doc(status="revoked", revoked_at="2027-07-14T16:00:00Z"),
    "automatic": contract_doc(),
}

AUTO_CASES: dict[str, object] = {
    "absent": None,
    "auto_v1": {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 3,
        "best_streak": 6,
        "source": "mlb_bts_profile",
        "source_date": "2026-09-28",
        "recorded_at": "2026-09-29T12:00:00Z",
        "user_id": 50311,
        "username": "stonehengee",
        "saver_available": None,
    },
    "auto_v2_match": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 4,
        "best_streak": 8,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-13",
        "recorded_at": "2027-07-14T12:00:00Z",
        "user_id": 50311,
        "username": "stonehengee",
        "saver_available": None,
        "season": 2027,
        "activation_id": ID_A,
    },
    "auto_v2_other_id": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 4,
        "best_streak": 8,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-13",
        "recorded_at": "2027-07-14T12:00:00Z",
        "user_id": 50311,
        "username": "stonehengee",
        "saver_available": None,
        "season": 2027,
        "activation_id": ID_B,
    },
    "auto_v2_other_season": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 4,
        "best_streak": 8,
        "source": "mlb_bts_profile",
        "source_date": "2026-09-28",
        "recorded_at": "2026-09-29T12:00:00Z",
        "user_id": 50311,
        "username": "stonehengee",
        "saver_available": None,
        "season": 2026,
        "activation_id": ID_A,
    },
    "auto_v2_bad_season": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 4,
        "best_streak": 8,
        "season": True,
        "activation_id": ID_A,
    },
    "auto_v2_bad_id": {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 4,
        "best_streak": 8,
        "season": 2027,
        "activation_id": "Z" * 32,
    },
    "refused_best58": {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 4,
        "best_streak": 58,
    },
    "not_utf8": b"\xff\xfe",
    "not_object": b"[1, 2, 3]",
    "other_account_user": {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 4,
        "best_streak": 8,
        "user_id": 50311,
        "username": "other_person",
    },
    "other_account_uid": {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 4,
        "best_streak": 8,
        "user_id": 99999,
        "username": "stonehengee",
    },
}

RESPONSE_CASES: list[str] = [
    "valid_settled",
    "valid_unsettled",
    "pred_empty",
    "pred_missing",
    "pred_null",
    "pred_nonlist",
    "pred_nonobj",
    "rounds_empty",
    "rounds_503",
    "rounds_429",
    "profile_null",
    "best_57",
    "best_58",
    "active_gt_best",
    "bool_streak",
]


def seed_file(path: Path, spec: object) -> None:
    if spec is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(spec, bytes):
        path.write_bytes(spec)
    else:
        path.write_text(json.dumps(spec), encoding="utf-8")


def tree_digest(root: Path, case_str: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root))
            raw = p.read_bytes().replace(case_str.encode("utf-8"), b"<CASE>")
            out[rel] = hashlib.sha256(raw).hexdigest()[:16] + f":{len(raw)}"
            if p.suffix in {".json", ".jsonl"}:
                try:
                    out[rel + "#text"] = raw.decode("utf-8")
                except UnicodeDecodeError:
                    out[rel + "#hex"] = raw.hex()
    return out


def run_worker(src_dir: Path, label: str) -> None:
    sys.path.insert(0, str(src_dir))
    import builtins
    import pandas  # noqa: F401
    import httpx
    from click.testing import CliRunner
    import bts.cli as climod
    import bts.contest_fetch as cf
    import bts.dm as dm_mod
    import bts.leaderboard.auth as auth
    import bts.saver_state as saver_mod
    import bts.util as util_mod

    class FixedClock(REAL_DATETIME):
        current = NOW

        @classmethod
        def now(cls, tz=None):
            c = cls.current
            return c.astimezone(tz) if tz else c.replace(tzinfo=None)

    dt.datetime = FixedClock

    runner = CliRunner()
    work = HERE / f"_work_{label}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    def configure_response(resp_kind: str):
        rounds_calls = []

        def fake_rounds():
            rounds_calls.append(1)
            if resp_kind == "rounds_empty":
                return {}
            if resp_kind == "rounds_503":
                raise httpx.HTTPStatusError(
                    "503",
                    request=httpx.Request("GET", "https://offline/rounds"),
                    response=httpx.Response(503),
                )
            if resp_kind == "rounds_429":
                raise httpx.HTTPStatusError(
                    "429",
                    request=httpx.Request("GET", "https://offline/rounds"),
                    response=httpx.Response(429),
                )
            return {1: dt.date(2027, 7, 14)}

        def fake_profile(uid, cookies, xsid):
            if resp_kind == "profile_null":
                return None
            if resp_kind == "pred_missing":
                return {"activeStreak": 7, "seasonBestStreak": 9}
            if resp_kind == "pred_null":
                return {"activeStreak": 7, "seasonBestStreak": 9, "predictions": None}
            if resp_kind == "pred_nonlist":
                return {"activeStreak": 7, "seasonBestStreak": 9, "predictions": {"a": 1}}
            if resp_kind == "pred_nonobj":
                return {"activeStreak": 7, "seasonBestStreak": 9, "predictions": [{"roundId": 1, "result": "hit"}, "bad"]}
            if resp_kind == "pred_empty":
                return {"activeStreak": 7, "seasonBestStreak": 9, "predictions": []}
            if resp_kind == "valid_unsettled":
                return {"activeStreak": 7, "seasonBestStreak": 9, "predictions": [{"roundId": 1, "result": None}]}
            if resp_kind == "best_57":
                return {"activeStreak": 7, "seasonBestStreak": 57, "predictions": [{"roundId": 1, "result": "hit"}]}
            if resp_kind == "best_58":
                return {"activeStreak": 7, "seasonBestStreak": 58, "predictions": [{"roundId": 1, "result": "hit"}]}
            if resp_kind == "active_gt_best":
                return {"activeStreak": 10, "seasonBestStreak": 9, "predictions": [{"roundId": 1, "result": "hit"}]}
            if resp_kind == "bool_streak":
                return {"activeStreak": True, "seasonBestStreak": 9, "predictions": [{"roundId": 1, "result": "hit"}]}
            return {"activeStreak": 7, "seasonBestStreak": 9, "predictions": [{"roundId": 1, "result": "hit"}]}

        auth.load_session_cookies = lambda: {"oktaid": "u"}
        auth.extract_uid = lambda c: "u"
        auth.fetch_login_session = lambda **kw: SimpleNamespace(user_id=50311, username="stonehengee", xsid="x")
        cf.fetch_profile = fake_profile
        climod._fetch_rounds = fake_rounds
        httpx.get = Mock(side_effect=AssertionError("uncaptured httpx.get"))
        return rounds_calls

    records: dict[str, dict] = {}
    io_violations: list[str] = []

    for sw_name, auto_name, resp_kind, dry in it.product(
        SWITCH_CASES.keys(), AUTO_CASES.keys(), RESPONSE_CASES, [False, True]
    ):
        case_dir = work / "case"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        picks = case_dir / "picks"
        picks.mkdir(parents=True)
        seed_file(picks / "account_state" / "season_contract.json", SWITCH_CASES[sw_name])
        seed_file(picks / "account_state" / "contest_streak.json", AUTO_CASES[auto_name])

        before_state = tree_digest(case_dir, str(case_dir))
        dms: list[tuple[str, str]] = []
        dm_mod.send_dm = lambda h, m, _dms=dms: _dms.append((h, m.replace(str(case_dir), "<CASE>")))
        rounds_calls = configure_response(resp_kind)

        global WATCH_ROOT
        AUDIT_EVENTS.clear()
        WATCH_ROOT = case_dir
        args = ["fetch-contest-streak", "--picks-dir", str(picks), "--expected-username", "stonehengee", "--dm-recipient", "op.bsky"]
        if dry:
            args.append("--dry-run")
        res = runner.invoke(climod.cli, args)
        WATCH_ROOT = None

        after_state = tree_digest(case_dir, str(case_dir))
        events_copy = list(AUDIT_EVENTS)

        if label == "branch":
            is_auto_switch = sw_name == "automatic"
            auto_writes = [
                ev for ev in events_copy
                if (ev[0] == "open" and ev[1].endswith("contest_streak.json") and ("mode=w" in ev[2] or "mode=a" in ev[2] or "+" in ev[2]))
                or (ev[0] == "os.rename" and ev[1].endswith("->picks/account_state/contest_streak.json"))
            ]
            saver_touches = [
                ev for ev in events_copy
                if any(s in ev[1] for s in ("saver_state.json", "saver_state.lock", "saver_transitions.jsonl"))
            ]
            if (not is_auto_switch or dry) and auto_writes:
                io_violations.append(f"auto_write on {sw_name}|{auto_name}|{resp_kind}|dry={dry}: {auto_writes}")
            if (not is_auto_switch or dry) and saver_touches:
                io_violations.append(f"saver_touch on {sw_name}|{auto_name}|{resp_kind}|dry={dry}: {saver_touches}")
            if not is_auto_switch or dry:
                before_auto = before_state.get("picks/account_state/contest_streak.json")
                after_auto = after_state.get("picks/account_state/contest_streak.json")
                if before_auto != after_auto:
                    io_violations.append(f"auto_bytes_changed on {sw_name}|{auto_name}|{resp_kind}|dry={dry}")

        key = f"{sw_name}|{auto_name}|{resp_kind}|dry={int(dry)}"
        records[key] = {
            "exit": res.exit_code,
            "exc_type": type(res.exception).__name__ if res.exception and not isinstance(res.exception, SystemExit) else None,
            "out": res.output.replace(str(case_dir), "<CASE>"),
            "dms": dms,
            "rounds_called": bool(rounds_calls),
            "before": {k: v for k, v in before_state.items() if "#" not in k},
            "after": {k: v for k, v in after_state.items() if "#" not in k},
            "auto_text": after_state.get("picks/account_state/contest_streak.json#text"),
            "status_text": after_state.get("health_state/contest_streak_fetch_status.json#text"),
            "ledger_text": after_state.get("picks/account_state/contest_ledger.jsonl#text"),
            "saver_text": after_state.get("picks/account_state/saver_state.json#text"),
            "write_order": [
                ev[1] for ev in events_copy
                if ev[0] == "os.rename" or (ev[0] == "open" and ("mode=a" in ev[2] or "saver_state.lock" in ev[1]))
            ],
        }

    # Pairwise response-failure precedence checks
    pairwise: dict[str, dict] = {}
    pair_specs = [
        ("profile_null_vs_rounds503", lambda: None, "503"),
        ("profile_null_vs_rounds429", lambda: None, "429"),
        ("pred_null_vs_rounds503", lambda: {"activeStreak": 7, "seasonBestStreak": 9, "predictions": None}, "503"),
        ("pred_null_vs_rounds429", lambda: {"activeStreak": 7, "seasonBestStreak": 9, "predictions": None}, "429"),
        ("pred_missing_vs_empty_rounds", lambda: {"activeStreak": 7, "seasonBestStreak": 9}, "empty"),
        ("pred_null_vs_empty_rounds", lambda: {"activeStreak": 7, "seasonBestStreak": 9, "predictions": None}, "empty"),
        ("pred_nonlist_vs_empty_rounds", lambda: {"activeStreak": 7, "seasonBestStreak": 9, "predictions": "str"}, "empty"),
        ("pred_nonobj_vs_empty_rounds", lambda: {"activeStreak": 7, "seasonBestStreak": 9, "predictions": [123]}, "empty"),
        ("bad_roundid_vs_best58", lambda: {"activeStreak": 7, "seasonBestStreak": 58, "predictions": [{"roundId": "abc", "result": "hit"}]}, "ok"),
        ("active60_gt_best59_vs_cap57", lambda: {"activeStreak": 60, "seasonBestStreak": 59, "predictions": []}, "ok"),
        ("active58_eq_best58_vs_cap57", lambda: {"activeStreak": 58, "seasonBestStreak": 58, "predictions": []}, "ok"),
        ("bool_best_streak", lambda: {"activeStreak": 0, "seasonBestStreak": True, "predictions": []}, "ok"),
    ]
    for p_name, prof_fn, rnd_mode in pair_specs:
        case_dir = work / "pair"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        picks = case_dir / "picks"
        picks.mkdir(parents=True)
        rc = []

        def rfn(_mode=rnd_mode, _rc=rc):
            _rc.append(1)
            if _mode == "empty":
                return {}
            if _mode in {"503", "429"}:
                code = int(_mode)
                raise httpx.HTTPStatusError(_mode, request=httpx.Request("GET", "https://offline"), response=httpx.Response(code))
            return {1: dt.date(2027, 7, 14)}

        cf.fetch_profile = lambda *a, _f=prof_fn: _f()
        climod._fetch_rounds = rfn
        dms = []
        dm_mod.send_dm = lambda h, m, _d=dms: _d.append(m.replace(str(case_dir), "<CASE>"))
        res = runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(picks), "--dm-recipient", "op.bsky"])
        st_file = case_dir / "health_state/contest_streak_fetch_status.json"
        st = json.loads(st_file.read_text()) if st_file.exists() else {}
        pairwise[p_name] = {
            "exit": res.exit_code,
            "out": res.output.replace(str(case_dir), "<CASE>"),
            "rounds_called": bool(rc),
            "category": st.get("last_error_category"),
            "last_error": (st.get("last_error") or "").replace(str(case_dir), "<CASE>"),
        }

    # Write-failure prefix matrix (Item 2): fail each write stage in each switch state
    prefix_cases: dict[str, dict] = {}
    if label == "branch":
        configure_response("valid_settled")
        for sw_name in ["absent", "other_season", "revoked", "automatic", "c8_note"]:
            for fail_stage in ["obs", "status", "ledger_open", "ledger_partial", "saver_lock", "saver_state", "saver_audit"]:
                case_dir = work / "prefix"
                if case_dir.exists():
                    shutil.rmtree(case_dir)
                picks = case_dir / "picks"
                picks.mkdir(parents=True)
                seed_file(picks / "account_state/season_contract.json", SWITCH_CASES[sw_name])
                seed_file(picks / "account_state/contest_streak.json", AUTO_CASES["auto_v1"])
                before_auto = (picks / "account_state/contest_streak.json").read_bytes()
                dms = []
                dm_mod.send_dm = lambda h, m, _d=dms: _d.append(m)

                orig_atomic = climod._atomic_write_json
                orig_popen = Path.open
                orig_bopen = builtins.open
                orig_awrite = saver_mod.atomic_write_text

                class PartialLedger:
                    def __init__(self, stream):
                        self.stream = stream
                    def __enter__(self):
                        return self
                    def write(self, text):
                        self.stream.write(text[:12])
                        self.stream.flush()
                        raise OSError("partial ledger write")
                    def __exit__(self, *a):
                        self.stream.close()

                def f_atomic(p, obj, _fs=fail_stage):
                    if _fs == "obs" and p.name == "contest_streak.json":
                        raise OSError("fail obs")
                    if _fs == "status" and p.name == "contest_streak_fetch_status.json":
                        raise OSError("fail status")
                    return orig_atomic(p, obj)

                def f_popen(self_p, mode="r", *a, _fs=fail_stage, **kw):
                    if self_p.name == "contest_ledger.jsonl" and "a" in mode:
                        if _fs == "ledger_open":
                            raise OSError("fail ledger open")
                        if _fs == "ledger_partial":
                            return PartialLedger(orig_popen(self_p, mode, *a, **kw))
                    return orig_popen(self_p, mode, *a, **kw)

                def f_bopen(file, mode="r", *a, _fs=fail_stage, **kw):
                    name = Path(file).name if isinstance(file, (str, os.PathLike)) else ""
                    if _fs == "saver_lock" and name == "saver_state.lock":
                        raise OSError("fail saver lock")
                    if _fs == "saver_audit" and name == "saver_transitions.jsonl":
                        raise OSError("fail saver audit")
                    return orig_bopen(file, mode, *a, **kw)

                def f_awrite(p, text, _fs=fail_stage):
                    if _fs == "saver_state" and Path(p).name == "saver_state.json":
                        raise OSError("fail saver state")
                    return orig_awrite(p, text)

                climod._atomic_write_json = f_atomic
                Path.open = f_popen
                builtins.open = f_bopen
                saver_mod.atomic_write_text = f_awrite
                try:
                    res = runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(picks), "--dm-recipient", "op.bsky"])
                finally:
                    climod._atomic_write_json = orig_atomic
                    Path.open = orig_popen
                    builtins.open = orig_bopen
                    saver_mod.atomic_write_text = orig_awrite
                    util_mod.atomic_write_text = orig_awrite

                auto_after = (picks / "account_state/contest_streak.json").read_bytes()
                st_exists = (case_dir / "health_state/contest_streak_fetch_status.json").exists()
                led_p = picks / "account_state/contest_ledger.jsonl"
                led_val = led_p.read_text() if led_p.exists() else None
                sav_exists = (picks / "account_state/saver_state.json").exists()
                prefix_cases[f"{sw_name}|{fail_stage}"] = {
                    "exit": res.exit_code,
                    "dms": len(dms),
                    "auto_changed": auto_after != before_auto,
                    "status_exists": st_exists,
                    "ledger_len": len(led_val) if led_val is not None else None,
                    "saver_exists": sav_exists,
                }

    # Year-boundary advancing clock checks
    clock_cases: dict[str, dict] = {}
    t_dec = REAL_DATETIME(2027, 12, 31, 23, 59, 59, 500000, tzinfo=ET)
    t_jan = REAL_DATETIME(2028, 1, 1, 0, 0, 0, 500000, tzinfo=ET)
    for advance_at in ["auth", "between_profile_rounds", "obs_write", "status_write", "ledger_write"]:
        for src_yr in [2027, 2028, None]:
            case_dir = work / "clock"
            if case_dir.exists():
                shutil.rmtree(case_dir)
            picks = case_dir / "picks"
            picks.mkdir(parents=True)
            if label == "branch":
                seed_file(picks / "account_state" / "season_contract.json", contract_doc(season=2027))

            FixedClock.current = t_dec
            earn_calls = []
            orig_earn = saver_mod.maybe_auto_earn_saver

            def spy_earn(*a, **kw):
                earn_calls.append(kw.get("season"))
                return orig_earn(*a, **kw)

            saver_mod.maybe_auto_earn_saver = spy_earn

            def c_load(_adv=advance_at):
                if _adv == "auth":
                    FixedClock.current = t_jan
                return {"oktaid": "u"}

            def c_prof(*a, _adv=advance_at, _sy=src_yr):
                if _adv == "between_profile_rounds":
                    FixedClock.current = t_jan
                preds = [] if _sy is None else [{"roundId": 1, "result": "hit"}]
                return {"activeStreak": 5, "seasonBestStreak": 7, "predictions": preds}

            def c_rnd(_sy=src_yr):
                d = dt.date(2027, 12, 31) if _sy in (2027, None) else dt.date(2028, 1, 1)
                return {1: d}

            orig_atomic = climod._atomic_write_json

            def c_atomic(p, obj, _adv=advance_at, _orig=orig_atomic):
                if _adv == "obs_write" and p.name == "contest_streak.json":
                    FixedClock.current = t_jan
                if _adv == "status_write" and p.name == "contest_streak_fetch_status.json":
                    FixedClock.current = t_jan
                return _orig(p, obj)

            orig_open = Path.open

            def c_open(self_p, mode="r", *a, _adv=advance_at, _orig=orig_open, **kw):
                if _adv == "ledger_write" and self_p.name == "contest_ledger.jsonl" and "a" in mode:
                    FixedClock.current = t_jan
                return _orig(self_p, mode, *a, **kw)

            auth.load_session_cookies = c_load
            cf.fetch_profile = c_prof
            climod._fetch_rounds = c_rnd
            climod._atomic_write_json = c_atomic
            Path.open = c_open
            try:
                res = runner.invoke(climod.cli, ["fetch-contest-streak", "--picks-dir", str(picks)])
            finally:
                climod._atomic_write_json = orig_atomic
                Path.open = orig_open
                saver_mod.maybe_auto_earn_saver = orig_earn
                FixedClock.current = NOW

            auto_p = picks / "account_state/contest_streak.json"
            led_p = picks / "account_state/contest_ledger.jsonl"
            st_p = case_dir / "health_state/contest_streak_fetch_status.json"
            clock_cases[f"{advance_at}|src={src_yr}"] = {
                "exit": res.exit_code,
                "auto": json.loads(auto_p.read_text()) if auto_p.exists() else None,
                "ledger": json.loads(led_p.read_text().splitlines()[-1]) if led_p.exists() else None,
                "status": json.loads(st_p.read_text()) if st_p.exists() else None,
                "earn_seasons": earn_calls,
            }

    shutil.rmtree(work, ignore_errors=True)
    out_file = HERE / f"probe1b_{label}.json"
    out_file.write_text(
        json.dumps(
            {
                "count": len(records),
                "io_violations": io_violations,
                "records": records,
                "pairwise": pairwise,
                "prefixes": prefix_cases,
                "clocks": clock_cases,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"WORKER {label}: {len(records)} matrix cases, {len(pairwise)} pairwise, "
        f"{len(prefix_cases)} prefixes, {len(clock_cases)} clock cases, io_violations={len(io_violations)}",
        flush=True,
    )


def analyze() -> None:
    base = json.loads((HERE / "probe1b_base.json").read_text(encoding="utf-8"))
    branch = json.loads((HERE / "probe1b_branch.json").read_text(encoding="utf-8"))
    assert base["count"] == branch["count"] == 5400
    assert branch["io_violations"] == [], f"IO violations on branch: {branch['io_violations'][:5]}"

    diff_buckets: dict[str, int] = {}
    unexpected: list[str] = []

    for key, b_rec in base["records"].items():
        sw_name, auto_name, resp_kind, dry_part = key.split("|")
        dry = dry_part == "dry=1"
        n_rec = branch["records"][key]

        if auto_name in {"not_utf8", "not_object"}:
            if not (b_rec["exit"] == n_rec["exit"] == 1 and b_rec["dms"] == n_rec["dms"] == []):
                unexpected.append(f"prior_crash mismatch {key}: base={b_rec['exit']} branch={n_rec['exit']}")
            diff_buckets["unchanged_prior_crash"] = diff_buckets.get("unchanged_prior_crash", 0) + 1
            continue

        if auto_name in {"other_account_user", "other_account_uid"}:
            if not (b_rec["exit"] == n_rec["exit"] == 2 and len(b_rec["dms"]) == len(n_rec["dms"]) == 1):
                unexpected.append(f"identity_guard mismatch {key}: base={b_rec} branch={n_rec}")
            diff_buckets["unchanged_identity_guard"] = diff_buckets.get("unchanged_identity_guard", 0) + 1
            continue

        already_rejected_on_base = {"pred_null", "pred_nonobj", "rounds_503", "rounds_429", "profile_null", "active_gt_best", "bool_streak"}
        newly_rejected_row3 = {"pred_missing", "pred_nonlist", "rounds_empty", "best_58"}

        if resp_kind in already_rejected_on_base:
            if not (b_rec["exit"] == n_rec["exit"] == 2 and len(n_rec["dms"]) == 1 and n_rec["ledger_text"] is None):
                unexpected.append(f"existing_reject mismatch {key}: base={b_rec} branch={n_rec}")
            bucket = "row3_pred_null_msg" if resp_kind == "pred_null" else "unchanged_early_reject"
            diff_buckets[bucket] = diff_buckets.get(bucket, 0) + 1
            continue

        if resp_kind in newly_rejected_row3:
            if not (b_rec["exit"] == 0 and n_rec["exit"] == 2 and len(n_rec["dms"]) == 1 and n_rec["ledger_text"] is None):
                unexpected.append(f"row3_new_reject mismatch {key}: base_exit={b_rec['exit']} branch={n_rec}")
            if n_rec["before"].get("picks/account_state/contest_streak.json") != n_rec["after"].get("picks/account_state/contest_streak.json"):
                unexpected.append(f"row3_modified_auto {key}")
            diff_buckets["row3_newly_rejected"] = diff_buckets.get("row3_newly_rejected", 0) + 1
            continue

        assert resp_kind in {"valid_settled", "valid_unsettled", "pred_empty", "best_57"}

        is_corrupt_switch = sw_name.startswith("c") or sw_name.startswith("pair_")
        if is_corrupt_switch:
            if not (n_rec["exit"] == 2 and len(n_rec["dms"]) == 1 and n_rec["ledger_text"] is None):
                unexpected.append(f"corrupt_switch mismatch {key}: {n_rec}")
            expected_check = {
                "c1_nonobj": "check 1:", "c2_schema": "check 2:", "c3_bool_season": "check 3:",
                "c4_status": "check 4:", "c5_id": "check 5:", "c6_switched": "check 6:",
                "c7_revoked": "check 7:", "c8_note": "check 8:",
                "pair_c2_c8_other_yr": "check 2:", "pair_c3_c5": "check 3:", "pair_c6_c7": "check 6:",
            }[sw_name]
            if expected_check not in n_rec["out"]:
                unexpected.append(f"corrupt_switch wrong check {key}: out={n_rec['out']!r}")
            diff_buckets["row4_5_corrupt_switch"] = diff_buckets.get("row4_5_corrupt_switch", 0) + 1
            continue

        if dry:
            if not (n_rec["exit"] == 0 and n_rec["dms"] == [] and n_rec["before"] == n_rec["after"]):
                unexpected.append(f"dry_run mismatch {key}: {n_rec}")
            expected_phrase = "season 2027 automatic; would write" if sw_name == "automatic" else "season 2027 not switched to automatic; would record reading in ledger and fetch status"
            if expected_phrase not in n_rec["out"]:
                unexpected.append(f"dry_run phrase mismatch {key}: {n_rec['out']!r}")
            diff_buckets["row5_dry_run_valid"] = diff_buckets.get("row5_dry_run_valid", 0) + 1
            continue

        if sw_name == "automatic":
            if n_rec["exit"] != 0 or n_rec["dms"] != []:
                unexpected.append(f"auto_success mismatch {key}: {n_rec}")
            auto_doc = json.loads(n_rec["auto_text"])
            led_doc = json.loads(n_rec["ledger_text"].strip())
            if not (
                auto_doc["schema_version"] == "bts_contest_streak_auto_v2"
                and auto_doc["season"] == 2027
                and auto_doc["activation_id"] == ID_A
                and auto_doc["recorded_at"] == "2027-07-15T16:00:00Z"
            ):
                unexpected.append(f"auto_doc stamp wrong {key}: {auto_doc}")
            if not (
                led_doc["season"] == 2027
                and led_doc["switched"] is True
                and led_doc["activation_id"] == ID_A
                and led_doc["recorded_at"] == "2027-07-15T16:00:00Z"
            ):
                unexpected.append(f"ledger_doc stamp wrong {key}: {led_doc}")
            wo = n_rec["write_order"]
            has_settled_2027 = resp_kind in {"valid_settled", "best_57"}
            if not (
                len(wo) >= 3
                and wo[0].endswith("->picks/account_state/contest_streak.json")
                and wo[1].endswith("->health_state/contest_streak_fetch_status.json")
                and wo[2] == "picks/account_state/contest_ledger.jsonl"
            ):
                unexpected.append(f"auto write_order wrong {key}: {wo}")
            if has_settled_2027:
                if resp_kind == "valid_settled" and n_rec["saver_text"] is None:
                    unexpected.append(f"saver not initialized on valid_settled {key}")
                if resp_kind == "best_57" and n_rec["saver_text"] is not None:
                    unexpected.append(f"cold best_57 initialized saver unexpectedly {key}")
            else:
                if len(wo) != 3 or n_rec["saver_text"] is not None:
                    unexpected.append(f"unsettled touched saver {key}: {wo}")
            diff_buckets["row4_6_automatic_write"] = diff_buckets.get("row4_6_automatic_write", 0) + 1
        else:
            assert sw_name in {"absent", "other_season", "revoked"}
            if n_rec["exit"] != 0 or n_rec["dms"] != []:
                unexpected.append(f"unswitched mismatch {key}: {n_rec}")
            led_doc = json.loads(n_rec["ledger_text"].strip())
            if not (
                led_doc["season"] == 2027
                and led_doc["switched"] is False
                and led_doc["activation_id"] is None
                and led_doc["recorded_at"] == "2027-07-15T16:00:00Z"
            ):
                unexpected.append(f"unswitched ledger wrong {key}: {led_doc}")
            wo = n_rec["write_order"]
            if not (
                len(wo) == 2
                and wo[0].endswith("->health_state/contest_streak_fetch_status.json")
                and wo[1] == "picks/account_state/contest_ledger.jsonl"
            ):
                unexpected.append(f"unswitched write_order wrong {key}: {wo}")
            if "season 2027 is not switched to automatic; reading recorded in the ledger and the fetch status" not in n_rec["out"]:
                unexpected.append(f"unswitched output wrong {key}: {n_rec['out']!r}")
            diff_buckets["row4_6_unswitched_ledger_status_only"] = diff_buckets.get("row4_6_unswitched_ledger_status_only", 0) + 1

    # Verify pairwise failure precedence between base and branch
    bp, np = base["pairwise"], branch["pairwise"]
    assert bp["profile_null_vs_rounds503"] == np["profile_null_vs_rounds503"]
    assert not np["profile_null_vs_rounds503"]["rounds_called"] and np["profile_null_vs_rounds503"]["category"] == "actionable"
    assert bp["profile_null_vs_rounds429"] == np["profile_null_vs_rounds429"]
    assert not np["profile_null_vs_rounds429"]["rounds_called"] and np["profile_null_vs_rounds429"]["category"] == "actionable"
    assert bp["pred_null_vs_rounds503"] == np["pred_null_vs_rounds503"]
    assert np["pred_null_vs_rounds503"]["rounds_called"] and np["pred_null_vs_rounds503"]["category"] == "transient"
    assert bp["pred_null_vs_rounds429"] == np["pred_null_vs_rounds429"]
    assert np["pred_null_vs_rounds429"]["rounds_called"] and np["pred_null_vs_rounds429"]["category"] == "rate_limited"
    assert bp["pred_missing_vs_empty_rounds"]["exit"] == 0 and "predictions key is missing" in np["pred_missing_vs_empty_rounds"]["last_error"]
    assert "predictions must be a list" in np["pred_null_vs_empty_rounds"]["last_error"]
    assert bp["pred_nonlist_vs_empty_rounds"]["exit"] == 0 and "predictions must be a list" in np["pred_nonlist_vs_empty_rounds"]["last_error"]
    assert "predictions must contain only objects" in np["pred_nonobj_vs_empty_rounds"]["last_error"]
    assert "invalid literal for int()" in np["bad_roundid_vs_best58"]["last_error"]
    assert ">= activeStreak" in np["active60_gt_best59_vs_cap57"]["last_error"]
    assert "<= 57" in np["active58_eq_best58_vs_cap57"]["last_error"]
    assert "seasonBestStreak must be an integer" in np["bool_best_streak"]["last_error"]

    # Verify write-failure prefixes on branch!
    pf = branch["prefixes"]
    # automatic:
    assert pf["automatic|obs"] == {"exit": 1, "dms": 0, "auto_changed": False, "status_exists": False, "ledger_len": None, "saver_exists": False}
    assert pf["automatic|status"] == {"exit": 1, "dms": 0, "auto_changed": True, "status_exists": False, "ledger_len": None, "saver_exists": False}
    assert pf["automatic|ledger_open"] == {"exit": 1, "dms": 0, "auto_changed": True, "status_exists": True, "ledger_len": None, "saver_exists": False}
    assert pf["automatic|ledger_partial"] == {"exit": 1, "dms": 0, "auto_changed": True, "status_exists": True, "ledger_len": 12, "saver_exists": False}
    assert pf["automatic|saver_lock"]["exit"] == 1 and pf["automatic|saver_lock"]["dms"] == 0 and pf["automatic|saver_lock"]["auto_changed"] and pf["automatic|saver_lock"]["status_exists"] and pf["automatic|saver_lock"]["ledger_len"] > 50 and not pf["automatic|saver_lock"]["saver_exists"]
    assert pf["automatic|saver_state"]["exit"] == 1 and pf["automatic|saver_state"]["dms"] == 0 and pf["automatic|saver_state"]["auto_changed"] and not pf["automatic|saver_state"]["saver_exists"]
    assert pf["automatic|saver_audit"]["exit"] == 0 and pf["automatic|saver_audit"]["dms"] == 0 and pf["automatic|saver_audit"]["auto_changed"] and pf["automatic|saver_audit"]["saver_exists"]
    # unswitched (absent, other_season, revoked):
    for sw in ["absent", "other_season", "revoked"]:
        assert pf[f"{sw}|obs"]["exit"] == 0 and not pf[f"{sw}|obs"]["auto_changed"] and pf[f"{sw}|obs"]["status_exists"] and pf[f"{sw}|obs"]["ledger_len"] > 50
        assert pf[f"{sw}|status"] == {"exit": 1, "dms": 0, "auto_changed": False, "status_exists": False, "ledger_len": None, "saver_exists": False}
        assert pf[f"{sw}|ledger_open"] == {"exit": 1, "dms": 0, "auto_changed": False, "status_exists": True, "ledger_len": None, "saver_exists": False}
        assert pf[f"{sw}|ledger_partial"] == {"exit": 1, "dms": 0, "auto_changed": False, "status_exists": True, "ledger_len": 12, "saver_exists": False}
        for s_stage in ["saver_lock", "saver_state", "saver_audit"]:
            assert pf[f"{sw}|{s_stage}"]["exit"] == 0 and not pf[f"{sw}|{s_stage}"]["auto_changed"] and not pf[f"{sw}|{s_stage}"]["saver_exists"]
    # corrupt switch (c8_note):
    assert pf["c8_note|status"] == {"exit": 1, "dms": 1, "auto_changed": False, "status_exists": False, "ledger_len": None, "saver_exists": False}
    assert pf["c8_note|obs"] == {"exit": 2, "dms": 1, "auto_changed": False, "status_exists": True, "ledger_len": None, "saver_exists": False}
    assert pf["c8_note|ledger_open"] == {"exit": 2, "dms": 1, "auto_changed": False, "status_exists": True, "ledger_len": None, "saver_exists": False}

    # Verify clock boundary cases on branch vs base
    for ckey, crec in branch["clocks"].items():
        adv, src_part = ckey.split("|")
        assert crec["exit"] == 0, f"clock exit failed {ckey}: {crec}"
        assert crec["auto"]["season"] == 2027
        assert crec["ledger"]["season"] == 2027
        assert crec["auto"]["recorded_at"] == "2028-01-01T04:59:59.500000Z"
        assert crec["ledger"]["recorded_at"] == "2028-01-01T04:59:59.500000Z"
        # Independent status timestamp reflects Jan 1 after clock advances at/before status_write
        if adv in {"auth", "between_profile_rounds", "obs_write"}:
            assert crec["status"]["last_success_at"] == "2028-01-01T05:00:00.500000Z"
        if src_part == "src=2027":
            assert crec["earn_seasons"] == [2027], f"expected earn(2027) on {ckey}: {crec}"
        else:
            assert crec["earn_seasons"] == [], f"expected no earn on {ckey}: {crec}"

    assert base["clocks"]["auth|src=2028"]["earn_seasons"] == [2028]
    assert base["clocks"]["auth|src=2027"]["earn_seasons"] == []

    summary = {
        "total_matrix_cases_per_tree": branch["count"],
        "total_runs": branch["count"] * 2,
        "io_violations": branch["io_violations"],
        "unexpected_diffs": unexpected,
        "buckets": diff_buckets,
        "prefix_cases_verified": len(pf),
        "pairwise_cases_verified": len(np),
        "clock_cases_verified": len(branch["clocks"]),
    }
    (HERE / "probe1b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert not unexpected, f"UNEXPECTED DIFFS ({len(unexpected)}): {unexpected[:10]}"
    print("PROBE 1B SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_worker(Path(sys.argv[1]), sys.argv[2])
    else:
        log_path = HERE / "probe1b.log"
        with log_path.open("w", encoding="utf-8") as log_fh:
            for sdir, lbl in [(HERE / "base_tree/src", "base"), (ROOT / "src", "branch")]:
                p = subprocess.run([sys.executable, __file__, str(sdir), lbl], capture_output=True, text=True, cwd=ROOT)
                log_fh.write(p.stdout + p.stderr)
                print(p.stdout, end="", flush=True)
                if p.returncode != 0:
                    print(p.stderr, file=sys.stderr, flush=True)
                    sys.exit(p.returncode)
        analyze()
