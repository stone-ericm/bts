"""Probe 1: Base vs branch fetch matrix, audit-hooked I/O order, failure prefixes, and single-clock boundaries."""
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

# Install process-wide network denial + file open/write audit hook
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
            # Also decode key JSON fields for exact semantic inspection
            if p.suffix in {".json", ".jsonl"}:
                try:
                    out[rel + "#text"] = raw.decode("utf-8")
                except UnicodeDecodeError:
                    out[rel + "#hex"] = raw.hex()
    return out


def run_worker(src_dir: Path, label: str) -> None:
    sys.path.insert(0, str(src_dir))
    import pandas  # noqa: F401 - import before datetime patching
    import httpx
    from click.testing import CliRunner
    import bts.cli as climod
    import bts.contest_fetch as cf
    import bts.dm as dm_mod
    import bts.leaderboard.auth as auth
    import bts.saver_state as saver_mod

    class FixedClock(REAL_DATETIME):
        current = NOW

        @classmethod
        def now(cls, tz=None):
            c = cls.current
            return c.astimezone(tz) if tz else c.replace(tzinfo=None)

    dt.datetime = FixedClock  # patch datetime module globally

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
            # valid_settled and rounds_*
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

    # Run full matrix: 15 switch_states x 12 auto_files x 15 responses x 2 dry_run = 5,400 cases!
    # Wait: optimize speed while keeping 100% coverage! Let's run all 5,400 cases!
    idx = 0
    for sw_name, auto_name, resp_kind, dry in it.product(
        SWITCH_CASES.keys(), AUTO_CASES.keys(), RESPONSE_CASES, [False, True]
    ):
        idx += 1
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

        # On branch, check strict non-write invariants for automatic file and saver state!
        if label == "branch":
            is_auto_switch = sw_name == "automatic"
            # Check if contest_streak.json or temp replacement targeting contest_streak.json ever occurred when not automatic or dry
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
        st = json.loads((case_dir / "health_state/contest_streak_fetch_status.json").read_text())
        pairwise[p_name] = {
            "exit": res.exit_code,
            "out": res.output.replace(str(case_dir), "<CASE>"),
            "rounds_called": bool(rc),
            "category": st.get("last_error_category"),
            "last_error": st.get("last_error", "").replace(str(case_dir), "<CASE>"),
        }

    # Year-boundary advancing clock checks (Dec 31 23:59:59.5 ET -> Jan 1 00:00:00.5 ET)
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
    out_file = HERE / f"probe1_{label}.json"
    out_file.write_text(
        json.dumps(
            {"count": len(records), "io_violations": io_violations, "records": records, "pairwise": pairwise, "clocks": clock_cases},
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"WORKER {label}: {len(records)} matrix cases, {len(pairwise)} pairwise, {len(clock_cases)} clock cases, io_violations={len(io_violations)}", flush=True)


def analyze() -> None:
    base = json.loads((HERE / "probe1_base.json").read_text(encoding="utf-8"))
    branch = json.loads((HERE / "probe1_branch.json").read_text(encoding="utf-8"))
    assert base["count"] == branch["count"] == 5400
    assert branch["io_violations"] == [], f"IO violations on branch: {branch['io_violations'][:5]}"

    diff_buckets: dict[str, int] = {}
    unexpected: list[str] = []

    for key, b_rec in base["records"].items():
        sw_name, auto_name, resp_kind, dry_part = key.split("|")
        dry = dry_part == "dry=1"
        n_rec = branch["records"][key]

        # Classify every single case mathematically against Rows 3, 4, 5, 6!
        # 1) Prior automatic file guards:
        if auto_name in {"not_utf8", "not_object"}:
            # Row 5: prior non-UTF8 / non-object crashes exit 1 with no alert on both base and branch
            if not (b_rec["exit"] == n_rec["exit"] == 1 and b_rec["dms"] == n_rec["dms"] == []):
                unexpected.append(f"prior_crash mismatch {key}: base={b_rec['exit']} branch={n_rec['exit']}")
            diff_buckets["unchanged_prior_crash"] = diff_buckets.get("unchanged_prior_crash", 0) + 1
            continue

        if auto_name in {"other_account_user", "other_account_uid"}:
            # Row 4/5: prior identity guard precedes profile and switch in every state, exits 2 with DM
            if not (b_rec["exit"] == n_rec["exit"] == 2 and len(b_rec["dms"]) == len(n_rec["dms"]) == 1):
                unexpected.append(f"identity_guard mismatch {key}: base={b_rec} branch={n_rec}")
            diff_buckets["unchanged_identity_guard"] = diff_buckets.get("unchanged_identity_guard", 0) + 1
            continue

        # 2) Response failures:
        already_rejected_on_base = {"pred_null", "pred_nonobj", "rounds_503", "rounds_429", "profile_null", "active_gt_best", "bool_streak"}
        newly_rejected_row3 = {"pred_missing", "pred_nonlist", "rounds_empty", "best_58"}

        if resp_kind in already_rejected_on_base:
            if not (b_rec["exit"] == n_rec["exit"] == 2 and len(n_rec["dms"]) == 1 and n_rec["ledger_text"] is None):
                unexpected.append(f"existing_reject mismatch {key}: base={b_rec} branch={n_rec}")
            # Note: for pred_null with valid rounds, base fails inside derive_source_date ('NoneType' not iterable),
            # while branch fails at explicit check ('profile predictions must be a list') — both exit 2 actionable DM!
            bucket = "row3_pred_null_msg" if resp_kind == "pred_null" else "unchanged_early_reject"
            diff_buckets[bucket] = diff_buckets.get(bucket, 0) + 1
            continue

        if resp_kind in newly_rejected_row3:
            # Base exited 0 (or wrote observation); branch MUST exit 2, write status, send 1 DM, write NO ledger, leave auto untouched!
            if not (b_rec["exit"] == 0 and n_rec["exit"] == 2 and len(n_rec["dms"]) == 1 and n_rec["ledger_text"] is None):
                unexpected.append(f"row3_new_reject mismatch {key}: base_exit={b_rec['exit']} branch={n_rec}")
            if n_rec["before"].get("picks/account_state/contest_streak.json") != n_rec["after"].get("picks/account_state/contest_streak.json"):
                unexpected.append(f"row3_modified_auto {key}")
            diff_buckets["row3_newly_rejected"] = diff_buckets.get("row3_newly_rejected", 0) + 1
            continue

        # Remaining responses are valid: valid_settled, valid_unsettled, pred_empty, best_57
        assert resp_kind in {"valid_settled", "valid_unsettled", "pred_empty", "best_57"}

        # 3) Switch state evaluation on valid reading:
        is_corrupt_switch = sw_name.startswith("c") or sw_name.startswith("pair_")
        if is_corrupt_switch:
            # Row 4 & Row 5: corrupt switch calls _fail (exit 2, status, 1 DM, no obs/ledger/saver), even on --dry-run!
            if not (n_rec["exit"] == 2 and len(n_rec["dms"]) == 1 and n_rec["ledger_text"] is None):
                unexpected.append(f"corrupt_switch mismatch {key}: {n_rec}")
            # Check pairwise check numbers:
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

        # Non-dry valid run with non-corrupt switch:
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
            # Verify write order: contest_streak.json -> contest_streak_fetch_status.json -> contest_ledger.jsonl -> (saver if settled)
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
                # best=9 initializes saver_state.json to not_earned; best=57 on cold stays uninitialized (no saver_state.json write, only lock+audit)
                if resp_kind == "valid_settled" and n_rec["saver_text"] is None:
                    unexpected.append(f"saver not initialized on valid_settled {key}")
                if resp_kind == "best_57" and n_rec["saver_text"] is not None:
                    unexpected.append(f"cold best_57 initialized saver unexpectedly {key}")
            else:
                if len(wo) != 3 or n_rec["saver_text"] is not None:
                    unexpected.append(f"unsettled touched saver {key}: {wo}")
            diff_buckets["row4_6_automatic_write"] = diff_buckets.get("row4_6_automatic_write", 0) + 1
        else:
            # absent, other_season, revoked
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

    # Verify pairwise failure precedence between base and branch!
    bp, np = base["pairwise"], branch["pairwise"]
    assert bp["profile_null_vs_rounds503"] == np["profile_null_vs_rounds503"]
    assert not np["profile_null_vs_rounds503"]["rounds_called"] and np["profile_null_vs_rounds503"]["category"] == "actionable"
    assert bp["profile_null_vs_rounds429"] == np["profile_null_vs_rounds429"]
    assert not np["profile_null_vs_rounds429"]["rounds_called"] and np["profile_null_vs_rounds429"]["category"] == "actionable"
    assert bp["pred_null_vs_rounds503"] == np["pred_null_vs_rounds503"]
    assert np["pred_null_vs_rounds503"]["rounds_called"] and np["pred_null_vs_rounds503"]["category"] == "transient"
    assert bp["pred_null_vs_rounds429"] == np["pred_null_vs_rounds429"]
    assert np["pred_null_vs_rounds429"]["rounds_called"] and np["pred_null_vs_rounds429"]["category"] == "rate_limited"
    assert "predictions key is missing" in np["pred_missing_vs_empty_rounds"]["last_error"]
    assert "predictions must be a list" in np["pred_null_vs_empty_rounds"]["last_error"]
    assert "predictions must be a list" in np["pred_nonlist_vs_empty_rounds"]["last_error"]
    assert "predictions must contain only objects" in np["pred_nonobj_vs_empty_rounds"]["last_error"]
    assert "invalid literal for int()" in np["bad_roundid_vs_best58"]["last_error"]
    assert ">= activeStreak" in np["active60_gt_best59_vs_cap57"]["last_error"]
    assert "<= 57" in np["active58_eq_best58_vs_cap57"]["last_error"]

    # Verify clock boundary cases on branch vs base!
    for ckey, crec in branch["clocks"].items():
        adv, src_part = ckey.split("|")
        assert crec["exit"] == 0, f"clock exit failed {ckey}: {crec}"
        # Entry was 2027-12-31 23:59:59.500000 ET == 2028-01-01T04:59:59.500000Z -> season MUST be 2027 in ALL phases!
        assert crec["auto"]["season"] == 2027, f"auto season wrong {ckey}: {crec}"
        assert crec["ledger"]["season"] == 2027, f"ledger season wrong {ckey}: {crec}"
        assert crec["auto"]["recorded_at"] == "2028-01-01T04:59:59.500000Z"
        assert crec["ledger"]["recorded_at"] == "2028-01-01T04:59:59.500000Z"
        if src_part == "src=2027":
            assert crec["earn_seasons"] == [2027], f"expected earn(2027) on {ckey}: {crec}"
        else:
            assert crec["earn_seasons"] == [], f"expected no earn on {ckey}: {crec}"

    # Verify base actually got the Dec->Jan clock boundary WRONG when clock advanced before saver (confirming RED)!
    assert base["clocks"]["auth|src=2028"]["earn_seasons"] == [2028]
    assert base["clocks"]["auth|src=2027"]["earn_seasons"] == []

    summary = {
        "total_matrix_cases_per_tree": branch["count"],
        "total_runs": branch["count"] * 2,
        "io_violations": branch["io_violations"],
        "unexpected_diffs": unexpected,
        "buckets": diff_buckets,
        "pairwise_branch": np,
        "clock_check": "PASS (all 15 advancing-clock cases stamped season=2027 and gated saver on 2027; base flipped to 2028)",
    }
    (HERE / "probe1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Clean up large raw JSONs only? No, keep compact or leave them? Let's check size; 5400 records is ~4MB, fine to keep!
    assert not unexpected, f"UNEXPECTED DIFFS ({len(unexpected)}): {unexpected[:10]}"
    print("PROBE 1 SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_worker(Path(sys.argv[1]), sys.argv[2])
    else:
        subprocess.run([sys.executable, __file__, str(HERE / "base_tree/src"), "base"], check=True, cwd=ROOT)
        subprocess.run([sys.executable, __file__, str(ROOT / "src"), "branch"], check=True, cwd=ROOT)
        analyze()
