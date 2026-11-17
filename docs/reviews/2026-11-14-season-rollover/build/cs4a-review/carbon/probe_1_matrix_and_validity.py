"""Adversarial Probe 1: Base-vs-Branch Matrix (Item 1) & Validity Definition (Item 3)."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[3]
REVIEW_DIR = REPO_ROOT / ".codex-review" / "cs4a-review"
BASE_SRC = REVIEW_DIR / "base_tree" / "src"
BRANCH_SRC = REPO_ROOT / "src"

ET = ZoneInfo("America/New_York")
NOW_ISO = "2027-07-15T12:00:00-04:00"
SCHEMA = "bts_contest_streak_manual_v3"


def install_no_network_hook() -> None:
    forbidden = {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.getnameinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.sendto",
    }

    def hook(event: str, args: tuple) -> None:
        if event in forbidden:
            raise RuntimeError(f"Network access forbidden in probe: {event} {args}")

    sys.addaudithook(hook)


def eod_iso(day: str, as_utc_z: bool = False) -> str:
    d = dt.date.fromisoformat(day)
    eod = dt.datetime.combine(d, dt.time.max, ET)
    if as_utc_z:
        return eod.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return eod.isoformat()


def valid_v3(day: str = "2027-07-15", **overrides) -> dict:
    d = dt.date.fromisoformat(day)
    base = {
        "schema_version": SCHEMA,
        "effective_date": day,
        "season": d.year,
        "expires_at": eod_iso(day),
        "active_streak": 12,
        "best_streak": 14,
        "saver_state": "not_earned",
        "recorded_at": "2027-07-15T16:00:00Z",
        "source": "manual_cli",
    }
    base.update(overrides)
    return base


def build_manual_cases() -> dict[str, tuple[str, object]]:
    """Map case_name -> (kind, payload) where kind is 'absent'|'json'|'raw_str'|'raw_bytes'|'dir'|'unreadable'|'dangling_symlink'."""
    cases: dict[str, tuple[str, object]] = {
        "absent": ("absent", None),
        # Legacy variants
        "legacy_v2_unexpired": ("json", {
            "schema_version": "bts_contest_streak_manual_v2",
            "active_streak": 7, "best_streak": 9, "source": "manual_cli",
            "source_date": "2027-07-15", "recorded_at": "2027-07-15T15:00:00Z",
            "override_expires_at": "2027-07-16T15:00:00Z",
        }),
        "legacy_v2_expired": ("json", {
            "schema_version": "bts_contest_streak_manual_v2",
            "active_streak": 7, "best_streak": 9, "source": "manual_cli",
            "source_date": "2027-07-14", "recorded_at": "2027-07-14T15:00:00Z",
            "override_expires_at": "2027-07-15T15:00:00Z",  # 11:00 ET < 12:00 ET
        }),
        "legacy_v2_exact_now_expiry": ("json", {
            "schema_version": "bts_contest_streak_manual_v2",
            "active_streak": 7, "best_streak": 9, "source": "manual_cli",
            "source_date": "2027-07-15", "recorded_at": "2027-07-14T16:00:00Z",
            "override_expires_at": "2027-07-15T16:00:00Z",  # == NOW
        }),
        "legacy_no_expiry": ("json", {
            "schema_version": "bts_contest_streak_manual_v1",
            "active_streak": 5, "source_date": "2027-07-15",
        }),
        "legacy_schema_missing_v3_shaped": ("json", {
            k: v for k, v in valid_v3(override_expires_at="2099-01-01T00:00:00Z").items()
            if k != "schema_version"
        }),
        "legacy_schema_null_v3_shaped": ("json", valid_v3(
            schema_version=None, override_expires_at="2099-01-01T00:00:00Z"
        )),
        "legacy_schema_unknown": ("json", valid_v3(
            schema_version="bts_contest_streak_manual_v4", override_expires_at="2099-01-01T00:00:00Z"
        )),
        "legacy_schema_trailing_space": ("json", valid_v3(
            schema_version=SCHEMA + " ", override_expires_at="2099-01-01T00:00:00Z"
        )),
        "legacy_schema_uppercase": ("json", valid_v3(
            schema_version=SCHEMA.upper(), override_expires_at="2099-01-01T00:00:00Z"
        )),
        "legacy_schema_int": ("json", valid_v3(
            schema_version=3, override_expires_at="2099-01-01T00:00:00Z"
        )),
        "legacy_streak_58_best_bad": ("json", {
            "schema_version": "bts_contest_streak_manual_v2",
            "active_streak": 58, "best_streak": "not_int",
            "override_expires_at": "2099-01-01T00:00:00Z",
        }),
        # Dated v3 valid variants
        "v3_today_not_earned_12": ("json", valid_v3("2027-07-15", active_streak=12, best_streak=14, saver_state="not_earned")),
        "v3_today_active_7": ("json", valid_v3("2027-07-15", active_streak=7, best_streak=10, saver_state="active")),
        "v3_today_used_7": ("json", valid_v3("2027-07-15", active_streak=7, best_streak=10, saver_state="used")),
        "v3_today_not_earned_7": ("json", valid_v3("2027-07-15", active_streak=7, best_streak=7, saver_state="not_earned")),
        "v3_today_ignored_legacy_keys": ("json", valid_v3(
            "2027-07-15", expires_at=eod_iso("2027-07-15", as_utc_z=True),
            source="", reason="",
            override_expires_at="2000-01-01T00:00:00Z",
            source_date="1999-01-01", streak="garbage", saver_available="bad_type", username=[1, 2],
        )),
        "v3_yesterday": ("json", valid_v3("2027-07-14")),
        "v3_yesterday_future_legacy_exp": ("json", valid_v3("2027-07-14", override_expires_at="2099-01-01T00:00:00Z")),
        "v3_tomorrow": ("json", valid_v3("2027-07-16")),
        "v3_tomorrow_future_legacy_exp": ("json", valid_v3(
            "2027-07-16", override_expires_at="2099-01-01T00:00:00Z", saver_available="not_bool"
        )),
        # Dated v3 failing checks 2..10
        "v3_fail_c2_date": ("json", valid_v3(effective_date="2027-7-15")),
        "v3_fail_c3_season": ("json", valid_v3(season=2026)),
        "v3_fail_c4_expires": ("json", valid_v3(expires_at="2027-07-15T23:59:59.999999")),
        "v3_fail_c5_active": ("json", valid_v3(active_streak=58, best_streak=58)),
        "v3_fail_c6_best": ("json", valid_v3(active_streak=12, best_streak=11)),
        "v3_fail_c7_saver": ("json", valid_v3(saver_state="uninitialized")),
        "v3_fail_c8_recorded": ("json", valid_v3(recorded_at="2027-07-15T16:00:00")),
        "v3_fail_c9_source": ("json", valid_v3(source=None)),
        "v3_fail_c10_reason": ("json", valid_v3(reason=None)),
        # Check 1 / unreadable / non-object / symlink
        "c1_malformed_json": ("raw_str", "{bad json"),
        "c1_utf8_bom": ("raw_bytes", b"\xef\xbb\xbf{}"),
        "c1_bad_utf8": ("raw_bytes", b"\xff\xfe"),
        "c1_array": ("raw_str", "[1, 2, 3]"),
        "c1_5000_digits": ("raw_str", '{"active_streak": ' + ("9" * 5000) + "}"),
        "c1_deep_nesting": ("raw_str", "[" * 2000 + "]" * 2000),
        "c1_directory": ("dir", None),
        "c1_unreadable": ("unreadable", "{}"),
        "dangling_symlink": ("dangling_symlink", None),
    }
    return cases


AUTO_CASES = {
    "absent": None,
    "valid_2027": {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 4, "best_streak": 8, "source": "mlb_bts_profile",
        "source_date": "2027-07-14", "recorded_at": "2027-07-15T10:00:00Z",
    },
    "auto_with_v3_schema": valid_v3("2027-07-14", active_streak=6, best_streak=8),
    "undated_no_recorded": {"active_streak": 3},
    "refused_58": {"active_streak": 58, "source_date": "2027-07-14"},
    "corrupt": "{",
}

FLAG_CASES = {
    "absent": None,
    "not_earned_2027": {"state": "not_earned", "season": 2027},
    "active_2027": {"state": "active", "season": 2027},
    "used_2027": {"state": "used", "season": 2027},
    "stale_2026": {"state": "active", "season": 2026},
    "corrupt": "{",
}


def write_manual(state_dir: Path, kind: str, payload: object) -> Path | None:
    p = state_dir / "contest_streak.manual.json"
    if kind == "absent":
        return None
    state_dir.mkdir(parents=True, exist_ok=True)
    if kind == "json":
        p.write_text(json.dumps(payload), encoding="utf-8")
    elif kind == "raw_str":
        p.write_text(str(payload), encoding="utf-8")
    elif kind == "raw_bytes":
        p.write_bytes(payload)  # type: ignore[arg-type]
    elif kind == "dir":
        p.mkdir()
    elif kind == "unreadable":
        p.write_text(str(payload), encoding="utf-8")
        p.chmod(0o000)
        return p
    elif kind == "dangling_symlink":
        p.symlink_to(state_dir / "does_not_exist.json")
    return None


def write_auto(state_dir: Path, payload: object) -> None:
    if payload is None:
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    p = state_dir / "contest_streak.json"
    if isinstance(payload, str):
        p.write_text(payload, encoding="utf-8")
    else:
        p.write_text(json.dumps(payload), encoding="utf-8")


def write_flag(state_dir: Path, payload: object) -> None:
    if payload is None:
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    p = state_dir / "saver_state.json"
    if isinstance(payload, str):
        p.write_text(payload, encoding="utf-8")
    else:
        p.write_text(json.dumps(payload), encoding="utf-8")


def serialize_reader(st) -> dict | None:
    if st is None:
        return None
    return {
        "streak": st.streak,
        "best_streak": st.best_streak,
        "saver_available": st.saver_available,
        "source": st.source,
        "source_date": st.source_date.isoformat() if st.source_date else None,
        "recorded_at": st.recorded_at,
        "path_name": st.path.name,
        "override_expires_at": st.override_expires_at.astimezone(dt.timezone.utc).isoformat() if st.override_expires_at else None,
        "schema_version": st.schema_version,
        "effective_date": getattr(st, "effective_date", None).isoformat() if getattr(st, "effective_date", None) else None,
        "saver_state": getattr(st, "saver_state", None),
    }


def serialize_decision(ds) -> dict:
    return {
        "streak": ds.streak,
        "saver_available": ds.saver_available,
        "allow_double": ds.allow_double,
        "source": ds.source,
        "status": ds.status,
        "contest_streak": ds.contest_streak,
        "contest_saver_available": ds.contest_saver_available,
        "contest_source_date": ds.contest_source_date.isoformat() if ds.contest_source_date else None,
        "best_streak": ds.best_streak,
        "best_status": ds.best_status,
        "saver_flag_state": ds.saver_flag_state,
    }


def run_worker(mode: str, out_file: Path) -> None:
    install_no_network_hook()
    import bts.contest_state as cs
    from bts.health.contest_state import check as health_check

    now = dt.datetime.fromisoformat(NOW_ISO)
    manual_cases = build_manual_cases()
    results: dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        idx = 0
        for m_name, (m_kind, m_payload) in manual_cases.items():
            for a_name, a_payload in AUTO_CASES.items():
                # Reader depends only on (manual, auto, require_dated_auto)
                idx += 1
                case_dir = root / f"c_{idx}"
                case_dir.mkdir()
                state_dir = case_dir / "account_state"
                chmod_p = write_manual(state_dir, m_kind, m_payload)
                write_auto(state_dir, a_payload)
                try:
                    reader_out = {}
                    for req in (False, True):
                        try:
                            r = cs.load_contest_streak_state(case_dir, now=now, require_dated_auto=req)
                            reader_out[str(req)] = {"ok": serialize_reader(r)}
                        except Exception as exc:
                            reader_out[str(req)] = {"err": f"{type(exc).__name__}: {exc}".replace(str(case_dir), "<DIR>")}

                    for f_name, f_payload in FLAG_CASES.items():
                        flag_file = state_dir / "saver_state.json"
                        if flag_file.exists():
                            flag_file.unlink()
                        write_flag(state_dir, f_payload)

                        dec_out = {}
                        for req in (False, True):
                            try:
                                d = cs.load_decision_streak_state(case_dir, require_contest_state=req, now=now)
                                dec_out[str(req)] = {"ok": serialize_decision(d)}
                            except Exception as exc:
                                dec_out[str(req)] = {"err": f"{type(exc).__name__}: {exc}".replace(str(case_dir), "<DIR>")}

                        hlth_out = {}
                        for exp in (False, True):
                            try:
                                alerts = health_check(case_dir, expected=exp, now=now)
                                hlth_out[str(exp)] = {
                                    "ok": [
                                        {
                                            "level": a.level,
                                            "source": a.source,
                                            "message": a.message.replace(str(case_dir), "<DIR>"),
                                        }
                                        for a in alerts
                                    ]
                                }
                            except Exception as exc:
                                hlth_out[str(exp)] = {"err": f"{type(exc).__name__}: {exc}".replace(str(case_dir), "<DIR>")}

                        key = f"{m_name}|{a_name}|{f_name}"
                        results[key] = {
                            "reader": reader_out,
                            "decision": dec_out,
                            "health": hlth_out,
                        }
                finally:
                    if chmod_p is not None:
                        chmod_p.chmod(0o644)

    out_file.write_text(json.dumps(results), encoding="utf-8")


def run_validity_deep_checks() -> dict:
    """Exhaustive Item 3 validity checks on Branch vs Base."""
    install_no_network_hook()
    sys.path.insert(0, str(BRANCH_SRC))
    import bts.contest_state as cs

    now_today = dt.datetime.fromisoformat(NOW_ISO)
    now_other = dt.datetime(2028, 3, 25, 12, 0, tzinfo=ET)

    summary = {
        "single_failures_tested": 0,
        "pair_failures_tested": 0,
        "odd_values_tested": 0,
        "timestamp_variants_tested": 0,
        "date_variants_tested": 0,
        "auto_parity_cases": 0,
        "failures": [],
    }

    # 1. Single-field failures for checks 2..10 across multiple bad forms
    bad_forms_by_check: dict[int, tuple[str, list[object]]] = {
        2: ("effective_date", [
            "__MISSING__", None, 20270715, True, [], {"d": "2027-07-15"},
            "2027-7-15", "2027-07-5", "20270715", "2027-W28-4", "2027W284",
            "2027-07-15T00:00:00", "2027-07-15 ", " 2027-07-15", "2027-02-29", "not-a-date", "",
        ]),
        3: ("season", [
            "__MISSING__", None, True, False, 2027.0, "2027", 2026, 2028, -2027, [], {}, float("nan"), float("inf"),
        ]),
        4: ("expires_at", [
            "__MISSING__", None, 123, [], {}, "", "Z",
            "2027-07-15T23:59:59.999999",        # naive
            "2027-07-16T03:59:59.999999",        # naive UTC
            "2027-07-15T23:59:59-04:00",         # missing microseconds
            "2027-07-16T03:59:59Z",              # missing microseconds UTC
            "2027-07-15T23:59:59.999998-04:00",  # 1us early
            "2027-07-16T00:00:00.000000-04:00",  # 1us late
            "2027-07-15T23:59:59.999999-05:00",  # wrong DST offset (EST on July day)
            "2027-07-16T03:59:59.999999+00:00Z", # double tz
            "2027-07-16Z",                       # date + Z
        ]),
        5: ("active_streak", [
            "__MISSING__", None, True, False, 0.0, 1.0, 12.0, -0.0, -1, 58, 100, "12", [], {}, float("nan"), float("inf"),
        ]),
        6: ("best_streak", [
            "__MISSING__", None, True, False, 14.0, -1, 11, 58, 99, "14", [], {}, float("nan"),
        ]),
        7: ("saver_state", [
            "__MISSING__", None, True, 0, "", "uninitialized", "ACTIVE", "not_earned ", "available", [], {},
        ]),
        8: ("recorded_at", [
            "__MISSING__", None, 123, [], {}, "", "Z",
            "2027-07-15", "2027-07-15T16:00:00", "2027-07-15T16:00:00.123456", "2027-07-15T25:00:00Z",
        ]),
        9: ("source", [
            "__MISSING__", None, True, 0, 1.5, [], {"s": "manual_cli"}, float("nan"),
        ]),
        10: ("reason", [
            None, True, False, 0, 1.5, [], {"r": "text"}, float("nan"),
        ]),
    }

    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        mpath = tdir / "account_state" / "contest_streak.manual.json"
        mpath.parent.mkdir(parents=True)

        for chk_num, (field, bad_vals) in bad_forms_by_check.items():
            for val in bad_vals:
                obj = valid_v3()
                if val == "__MISSING__":
                    del obj[field]
                else:
                    obj[field] = val
                mpath.write_text(json.dumps(obj, allow_nan=True), encoding="utf-8")
                for probe_now in (now_today, now_other):
                    summary["single_failures_tested"] += 1
                    try:
                        cs.load_contest_streak_state(tdir, now=probe_now)
                        summary["failures"].append(f"Expected check {chk_num} failure for {field}={val!r}, got success")
                    except cs.ContestStateError as exc:
                        msg = str(exc)
                        if f"check {chk_num}," not in msg or field not in msg or str(mpath) not in msg:
                            summary["failures"].append(
                                f"Wrong error for check {chk_num} {field}={val!r}: {msg}"
                            )

        # Special year=1 True check: on 0001-01-01, effective_date.year == 1 == True!
        obj_y1 = valid_v3("0001-01-01", season=True)
        mpath.write_text(json.dumps(obj_y1), encoding="utf-8")
        summary["odd_values_tested"] += 1
        try:
            cs.load_contest_streak_state(tdir, now=now_today)
            summary["failures"].append("Accepted season=True on 0001-01-01!")
        except cs.ContestStateError as exc:
            if "check 3" not in str(exc):
                summary["failures"].append(f"Wrong error on 0001-01-01 season=True: {exc}")

        # 2. All C(9, 2) = 36 pairs (i, j) for 2 <= i < j <= 10, on today and other date
        rep_bad = {
            2: ("effective_date", "20270715"),
            3: ("season", 2026),
            4: ("expires_at", "2027-07-15T23:59:59.999999"),
            5: ("active_streak", 58),
            6: ("best_streak", 11),
            7: ("saver_state", "uninitialized"),
            8: ("recorded_at", "2027-07-15T16:00:00"),
            9: ("source", None),
            10: ("reason", None),
        }
        checks = sorted(rep_bad.keys())
        for idx_i, i in enumerate(checks):
            for j in checks[idx_i + 1:]:
                fi, vi = rep_bad[i]
                fj, vj = rep_bad[j]
                obj = valid_v3(**{fi: vi, fj: vj})
                mpath.write_text(json.dumps(obj), encoding="utf-8")
                for probe_now in (now_today, now_other):
                    summary["pair_failures_tested"] += 1
                    try:
                        cs.load_contest_streak_state(tdir, now=probe_now)
                        summary["failures"].append(f"Pair ({i},{j}) unexpectedly succeeded")
                    except cs.ContestStateError as exc:
                        msg = str(exc)
                        if f"check {i}," not in msg or fi not in msg:
                            summary["failures"].append(f"Pair ({i},{j}) reported wrong check: {msg}")

        # 3. Valid boundary & odd JSON encodings that MUST succeed or fail cleanly
        # -0 for active_streak and best_streak
        raw_neg_zero = (
            '{"schema_version":"bts_contest_streak_manual_v3","effective_date":"2027-07-15",'
            '"season":2027,"expires_at":"2027-07-15T23:59:59.999999-04:00",'
            '"active_streak":-0,"best_streak":-0,"saver_state":"not_earned",'
            '"recorded_at":"2027-07-15T16:00:00Z","source":""}'
        )
        mpath.write_text(raw_neg_zero, encoding="utf-8")
        summary["odd_values_tested"] += 1
        st = cs.load_contest_streak_state(tdir, now=now_today)
        if st is None or st.streak != 0 or st.best_streak != 0 or type(st.streak) is not int:
            summary["failures"].append(f"-0 integer failed: {st}")

        # 57/57 boundary
        mpath.write_text(json.dumps(valid_v3(active_streak=57, best_streak=57, saver_state="used")), encoding="utf-8")
        summary["odd_values_tested"] += 1
        st57 = cs.load_contest_streak_state(tdir, now=now_today)
        if st57 is None or st57.streak != 57 or st57.best_streak != 57:
            summary["failures"].append(f"57/57 failed: {st57}")

        # 5,000-digit integer inside an IGNORED extra key of a v3 file -> Check 1 ContestStateError
        raw_5000_extra = (
            '{"schema_version":"bts_contest_streak_manual_v3","effective_date":"2027-07-15",'
            '"season":2027,"expires_at":"2027-07-15T23:59:59.999999-04:00",'
            '"active_streak":1,"best_streak":1,"saver_state":"not_earned",'
            '"recorded_at":"2027-07-15T16:00:00Z","source":"x","extra":' + ("9" * 5000) + "}"
        )
        mpath.write_text(raw_5000_extra, encoding="utf-8")
        summary["odd_values_tested"] += 1
        try:
            cs.load_contest_streak_state(tdir, now=now_today)
            summary["failures"].append("5000-digit int in extra key did not raise ContestStateError")
        except cs.ContestStateError:
            pass

        # Duplicate keys in JSON: first invalid, second valid -> last wins (valid); first valid, second invalid -> fails
        dup_valid_wins = (
            '{"schema_version":"bts_contest_streak_manual_v3","effective_date":"2027-07-15",'
            '"season":2027,"expires_at":"2027-07-15T23:59:59.999999-04:00",'
            '"active_streak":999,"active_streak":5,"best_streak":5,"saver_state":"active",'
            '"recorded_at":"2027-07-15T16:00:00Z","source":"x"}'
        )
        mpath.write_text(dup_valid_wins, encoding="utf-8")
        summary["odd_values_tested"] += 1
        st_dup = cs.load_contest_streak_state(tdir, now=now_today)
        if st_dup is None or st_dup.streak != 5:
            summary["failures"].append(f"dup_valid_wins unexpected: {st_dup}")

        dup_invalid_wins = (
            '{"schema_version":"bts_contest_streak_manual_v3","effective_date":"2027-07-15",'
            '"season":2027,"expires_at":"2027-07-15T23:59:59.999999-04:00",'
            '"active_streak":5,"active_streak":999,"best_streak":5,"saver_state":"active",'
            '"recorded_at":"2027-07-15T16:00:00Z","source":"x"}'
        )
        mpath.write_text(dup_invalid_wins, encoding="utf-8")
        summary["odd_values_tested"] += 1
        try:
            cs.load_contest_streak_state(tdir, now=now_today)
            summary["failures"].append("dup_invalid_wins unexpectedly succeeded")
        except cs.ContestStateError as exc:
            if "check 5" not in str(exc):
                summary["failures"].append(f"dup_invalid_wins wrong check: {exc}")

        # Valid timestamp offset spellings for expires_at & recorded_at
        valid_exp_spellings = [
            "2027-07-15T23:59:59.999999-04:00",
            "2027-07-16T03:59:59.999999Z",
            "2027-07-16T03:59:59.999999+00:00",
            "2027-07-16T03:59:59.999999-00:00",
            "2027-07-16T05:59:59.999999+02:00",
            "2027-07-16T09:29:59.999999+05:30",
        ]
        valid_rec_spellings = [
            "2027-07-15T16:00:00Z",
            "2027-07-15T16:00:00.123456Z",
            "2027-07-15T12:00:00-04:00",
            "2027-07-15T12:00:00.000001-04:00",
            "2027-07-15T16:00:00+00:00",
        ]
        for exp_s in valid_exp_spellings:
            for rec_s in valid_rec_spellings:
                summary["timestamp_variants_tested"] += 1
                mpath.write_text(json.dumps(valid_v3(expires_at=exp_s, recorded_at=rec_s)), encoding="utf-8")
                st_ts = cs.load_contest_streak_state(tdir, now=now_today)
                if st_ts is None or st_ts.recorded_at != rec_s:
                    summary["failures"].append(f"Timestamp spelling failed: {exp_s} / {rec_s}")

        # DST expires_at offset checks on 2027-03-14 (EDT -04:00 at EOD) and 2027-11-07 (EST -05:00 at EOD)
        for dst_day, good_off, bad_off in [
            ("2027-03-14", "-04:00", "-05:00"),
            ("2027-11-07", "-05:00", "-04:00"),
            ("2028-02-29", "-05:00", "-04:00"),
        ]:
            summary["date_variants_tested"] += 2
            mpath.write_text(json.dumps(valid_v3(dst_day, expires_at=f"{dst_day}T23:59:59.999999{good_off}")), encoding="utf-8")
            st_dst = cs.load_contest_streak_state(tdir, now=dt.datetime.fromisoformat(f"{dst_day}T12:00:00{good_off}"))
            if st_dst is None or st_dst.effective_date != dt.date.fromisoformat(dst_day):
                summary["failures"].append(f"DST good offset failed for {dst_day}")
            mpath.write_text(json.dumps(valid_v3(dst_day, expires_at=f"{dst_day}T23:59:59.999999{bad_off}")), encoding="utf-8")
            try:
                cs.load_contest_streak_state(tdir, now=dt.datetime.fromisoformat(f"{dst_day}T12:00:00{good_off}"))
                summary["failures"].append(f"DST bad offset {bad_off} accepted on {dst_day}")
            except cs.ContestStateError as exc:
                if "check 4" not in str(exc):
                    summary["failures"].append(f"DST bad offset wrong check on {dst_day}: {exc}")

    return summary


def compare_base_and_branch(base_data: dict, branch_data: dict) -> dict:
    assert set(base_data.keys()) == set(branch_data.keys()), "Key mismatch between base and branch!"
    counts = {
        "total_cells": len(base_data),
        "identical_cells": 0,
        "changed_cells": 0,
        "by_manual_group": {},
        "unexpected_differences": [],
    }

    for key, base_val in sorted(base_data.items()):
        branch_val = branch_data[key]
        m_name, a_name, f_name = key.split("|")

        if m_name.startswith(("absent", "legacy_", "c1_", "dangling_")):
            group = "preservation_legacy_absent_c1"
        elif m_name.startswith("v3_today"):
            group = "v3_today"
        elif m_name.startswith(("v3_yesterday", "v3_tomorrow")):
            group = "v3_other_day"
        elif m_name.startswith("v3_fail_"):
            group = "v3_invalid_property"
        else:
            group = "other"

        g_stat = counts["by_manual_group"].setdefault(group, {"total": 0, "identical": 0, "changed": 0})
        g_stat["total"] += 1

        if base_val == branch_val:
            counts["identical_cells"] += 1
            g_stat["identical"] += 1
        else:
            counts["changed_cells"] += 1
            g_stat["changed"] += 1

        # Strict semantic verification of EVERY cell!
        if group == "preservation_legacy_absent_c1":
            # Every single reader, decision, and health call MUST be 100% identical to base!
            if base_val != branch_val:
                counts["unexpected_differences"].append({
                    "key": key, "reason": "Preservation cell differed from base!",
                    "base": base_val, "branch": branch_val,
                })
        elif group == "v3_other_day":
            # Row 8 & Row 10: a valid dated observation NOT in effect must equal the absent-manual outcome!
            absent_key = f"absent|{a_name}|{f_name}"
            expected_val = branch_data[absent_key]
            if branch_val != expected_val:
                counts["unexpected_differences"].append({
                    "key": key, "reason": "Out-of-date v3 did not match absent-manual outcome!",
                    "expected": expected_val, "branch": branch_val,
                })
        elif group == "v3_invalid_property":
            # Row 6 & Row 10: must raise ContestStateError naming check number on reader & decision, and CRITICAL on health!
            chk_token = m_name.split("_")[2]  # e.g. c2 -> check 2
            chk_num = chk_token[1:]
            for req in ("False", "True"):
                r_err = branch_val["reader"][req].get("err", "")
                d_err = branch_val["decision"][req].get("err", "")
                if f"check {chk_num}," not in r_err or f"check {chk_num}," not in d_err:
                    counts["unexpected_differences"].append({
                        "key": key, "reason": f"Missing check {chk_num} in reader/decision error",
                        "branch": branch_val,
                    })
            for exp in ("False", "True"):
                h_ok = branch_val["health"][exp].get("ok", [])
                if len(h_ok) != 1 or h_ok[0]["level"] != "CRITICAL" or f"check {chk_num}," not in h_ok[0]["message"]:
                    counts["unexpected_differences"].append({
                        "key": key, "reason": f"Health did not return single CRITICAL with check {chk_num}",
                        "branch": branch_val["health"][exp],
                    })
        elif group == "v3_today":
            # Rows 7, 8, 9, 10, 11: today's v3 must win regardless of auto & flag!
            for req in ("False", "True"):
                r_ok = branch_val["reader"][req].get("ok")
                d_ok = branch_val["decision"][req].get("ok")
                if not r_ok or r_ok["path_name"] != "contest_streak.manual.json" or r_ok["effective_date"] != "2027-07-15":
                    counts["unexpected_differences"].append({"key": key, "reason": "Reader did not select today v3", "branch": branch_val})
                if not d_ok or d_ok["source"] != "contest" or d_ok["status"] != "fresh" or d_ok["best_status"] != "trusted":
                    counts["unexpected_differences"].append({"key": key, "reason": "Decision not fresh/trusted contest", "branch": branch_val})
                expected_saver = r_ok["saver_state"] == "active" or (r_ok["saver_state"] == "not_earned" and r_ok["streak"] < 10)
                if d_ok["saver_available"] != expected_saver or d_ok["saver_flag_state"] != r_ok["saver_state"]:
                    counts["unexpected_differences"].append({"key": key, "reason": "Decision saver mismatch", "branch": branch_val})

    return counts


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        run_worker("worker", Path(sys.argv[2]))
        return

    base_json = REVIEW_DIR / "probe1_base.json"
    branch_json = REVIEW_DIR / "probe1_branch.json"

    for label, src_dir, out_json in [("base", BASE_SRC, base_json), ("branch", BRANCH_SRC, branch_json)]:
        env = dict(os.environ, PYTHONPATH=str(src_dir), PYTHONDONTWRITEBYTECODE="1", TZ="America/New_York")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(out_json)],
            env=env, check=True,
        )

    base_data = json.loads(base_json.read_text(encoding="utf-8"))
    branch_data = json.loads(branch_json.read_text(encoding="utf-8"))
    matrix_summary = compare_base_and_branch(base_data, branch_data)
    validity_summary = run_validity_deep_checks()

    report = {
        "matrix": matrix_summary,
        "validity": validity_summary,
    }
    out_summary = REVIEW_DIR / "probe1_summary.json"
    out_summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if matrix_summary["unexpected_differences"] or validity_summary["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
